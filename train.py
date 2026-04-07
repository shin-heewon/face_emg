"""
감정 분류 모델 학습 스크립트

Usage:
  python train.py --backbone densenet121 --output_dir output/densenet121
  python train.py --backbone efficientnet_b0 --use_clahe --use_edge --output_dir output/efficientnet_b0_enhanced
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EmotionDataset, SAMPLE_EMOTIONS
from model import EmotionClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',   default='New_sample_3')
    p.add_argument('--backbone',    default='efficientnet_b0',
                   choices=['efficientnet_b0', 'densenet121', 'densenet169',
                            'resnet18', 'resnet50'])
    p.add_argument('--epochs',      type=int, default=30)
    p.add_argument('--batch_size',  type=int, default=32)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--image_size',  type=int, default=224)
    p.add_argument('--val_ratio',   type=float, default=0.2)
    p.add_argument('--output_dir',  default=None)   # None → output/{backbone}/
    p.add_argument('--num_workers', type=int, default=0)
    # 전처리 옵션
    p.add_argument('--use_clahe',   action='store_true', help='CLAHE 히스토그램 평활화')
    p.add_argument('--use_edge',    action='store_true', help='엣지 채널 추가 (4ch)')
    p.add_argument('--use_align',   action='store_true', help='mediapipe 얼굴 정렬')
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, leave=False, desc='  train'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (model(imgs).detach().argmax(1) == labels).sum().item() if False else \
                   (imgs.shape[0] - (model(imgs).detach().argmax(1) != labels).sum().item()) # reuse forward
        total += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, leave=False, desc='  val  '):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def main():
    args = parse_args()

    # output_dir 자동 분기
    if args.output_dir is None:
        suffix = ''
        if args.use_clahe or args.use_edge or args.use_align:
            parts = []
            if args.use_clahe:  parts.append('clahe')
            if args.use_edge:   parts.append('edge')
            if args.use_align:  parts.append('align')
            suffix = '_' + '_'.join(parts)
        args.output_dir = os.path.join('output', args.backbone + suffix)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = 4 if args.use_edge else 3

    print(f'Device: {device}')
    print(f'Backbone: {args.backbone}  |  in_channels: {in_channels}')
    print(f'전처리: clahe={args.use_clahe}, edge={args.use_edge}, align={args.use_align}')
    print(f'Output: {args.output_dir}')

    ds_kwargs = dict(
        data_root=args.data_root, emotions=SAMPLE_EMOTIONS,
        val_ratio=args.val_ratio, image_size=args.image_size,
        use_clahe=args.use_clahe, use_edge=args.use_edge, use_align=args.use_align,
    )
    train_ds = EmotionDataset(split='train', augment=True,  **ds_kwargs)
    val_ds   = EmotionDataset(split='val',   augment=False, **ds_kwargs)

    print(f'Train: {len(train_ds)}장  |  Val: {len(val_ds)}장')
    print('클래스별:', train_ds.class_counts())

    counts = train_ds.class_counts()
    class_weights = torch.tensor(
        [1.0 / counts.get(e, 1) for e in SAMPLE_EMOTIONS], dtype=torch.float32
    ).to(device)
    class_weights = class_weights / class_weights.sum() * len(SAMPLE_EMOTIONS)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type=='cuda'))
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    model = EmotionClassifier(
        len(SAMPLE_EMOTIONS), backbone=args.backbone,
        pretrained=True, in_channels=in_channels,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # 학습 루프 (forward 중복 방지)
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, leave=False, desc='  train'):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(labels)
            tc += (logits.detach().argmax(1) == labels).sum().item()
            tt += len(labels)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f'Epoch {epoch:3d}/{args.epochs} '
            f'| train loss {tl/tt:.4f} acc {tc/tt:.3f} '
            f'| val loss {val_loss:.4f} acc {val_acc:.3f} '
            f'| {time.time()-t0:.1f}s'
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'backbone': args.backbone,
                'num_classes': len(SAMPLE_EMOTIONS),
                'emotions': SAMPLE_EMOTIONS,
                'in_channels': in_channels,
                'use_clahe': args.use_clahe,
                'use_edge': args.use_edge,
                'use_align': args.use_align,
                'state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f'  => Best saved (val_acc={val_acc:.3f})')

    print(f'\n완료. Best val acc: {best_val_acc:.3f} → {args.output_dir}/best_model.pth')


if __name__ == '__main__':
    main()
