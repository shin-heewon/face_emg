"""
학습된 모델 평가 스크립트

Usage:
  python evaluate.py --checkpoint output/best_model.pth --data_root New_sample_3
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import EmotionDataset, SAMPLE_EMOTIONS
from model import EmotionClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  required=True)
    p.add_argument('--data_root',   default='New_sample_3')
    p.add_argument('--split',       default='val', choices=['val', 'all'])
    p.add_argument('--batch_size',  type=int, default=32)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--save_cm',     default='output/confusion_matrix.png')
    return p.parse_args()


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc='평가 중'):
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel('예측')
    ax.set_ylabel('실제')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f'Confusion matrix 저장: {save_path}')


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 체크포인트 로드
    ckpt = torch.load(args.checkpoint, map_location=device)
    emotions = ckpt.get('emotions', SAMPLE_EMOTIONS)
    num_classes = ckpt.get('num_classes', len(emotions))
    backbone = ckpt.get('backbone', 'efficientnet_b0')

    model = EmotionClassifier(num_classes, backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt['state_dict'])

    # 데이터셋
    ds = EmotionDataset(
        args.data_root, emotions=emotions,
        split=args.split, augment=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    preds, labels = predict_all(model, loader, device)
    acc = (preds == labels).mean()

    print(f'\n전체 정확도: {acc:.4f}  ({int(acc*len(labels))}/{len(labels)})\n')
    print(classification_report(
        labels, preds,
        target_names=emotions,
        digits=3,
    ))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    import os; os.makedirs(os.path.dirname(args.save_cm), exist_ok=True)
    plot_confusion_matrix(cm, emotions, args.save_cm)


if __name__ == '__main__':
    main()
