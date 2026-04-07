"""
모델 비교 스크립트

Usage:
  python compare.py --data_root New_sample_3

output/*/best_model.pth 를 자동 탐색해 동일 val set 기준 비교표와 차트를 생성.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EmotionDataset, SAMPLE_EMOTIONS
from model import EmotionClassifier


@torch.no_grad()
def eval_model(ckpt_path: str, data_root: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    emotions    = ckpt.get('emotions', SAMPLE_EMOTIONS)
    backbone    = ckpt.get('backbone', 'efficientnet_b0')
    num_classes = ckpt.get('num_classes', len(emotions))
    in_channels = ckpt.get('in_channels', 3)
    use_clahe   = ckpt.get('use_clahe', False)
    use_edge    = ckpt.get('use_edge', False)
    use_align   = ckpt.get('use_align', False)

    model = EmotionClassifier(num_classes, backbone, pretrained=False,
                              in_channels=in_channels).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    ds = EmotionDataset(
        data_root, emotions=emotions, split='val', augment=False,
        use_clahe=use_clahe, use_edge=use_edge, use_align=use_align,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    preds, labels = [], []
    for imgs, lbls in tqdm(loader, desc=f'  {backbone}', leave=False):
        logits = model(imgs.to(device))
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(lbls.numpy())

    preds  = np.array(preds)
    labels = np.array(labels)
    acc    = (preds == labels).mean()
    f1_per = f1_score(labels, preds, average=None, labels=list(range(len(emotions))))

    # 모델 이름 구성
    parts = [backbone]
    if use_clahe: parts.append('CLAHE')
    if use_edge:  parts.append('Edge')
    if use_align: parts.append('Align')
    name = '+'.join(parts)

    return {
        'name': name,
        'backbone': backbone,
        'acc': acc,
        'f1_per': f1_per,
        'emotions': emotions,
        'ckpt': ckpt_path,
    }


def make_comparison_table(results: list, emotions: list):
    header = f"{'모델':<40} {'전체Acc':>8} " + \
             ' '.join(f"{e:>6}" for e in emotions)
    sep = '-' * len(header)
    lines = [header, sep]
    for r in results:
        f1s = ' '.join(f"{v:>6.3f}" for v in r['f1_per'])
        lines.append(f"{r['name']:<40} {r['acc']:>8.4f} {f1s}")
    return '\n'.join(lines)


def plot_comparison(results: list, emotions: list, save_path: str):
    names = [r['name'] for r in results]
    accs  = [r['acc'] for r in results]
    n_emo = len(emotions)
    x = np.arange(n_emo)
    w = 0.8 / max(len(results), 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 전체 정확도 막대 차트
    axes[0].bar(names, [a * 100 for a in accs], color='steelblue', edgecolor='white')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('전체 Accuracy 비교')
    axes[0].set_ylim(0, 100)
    for i, a in enumerate(accs):
        axes[0].text(i, a * 100 + 0.5, f'{a*100:.1f}%', ha='center', fontsize=9)
    axes[0].tick_params(axis='x', rotation=15)

    # 클래스별 F1 비교
    for i, r in enumerate(results):
        axes[1].bar(x + i * w, r['f1_per'], width=w, label=r['name'])
    axes[1].set_xticks(x + w * (len(results) - 1) / 2)
    axes[1].set_xticklabels(emotions)
    axes[1].set_ylabel('F1-score')
    axes[1].set_title('클래스별 F1 비교')
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f'비교 차트 저장: {save_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',  default='New_sample_3')
    p.add_argument('--output_dir', default='output')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_paths = sorted(glob.glob(os.path.join(args.output_dir, '*', 'best_model.pth')))
    if not ckpt_paths:
        print(f'체크포인트 없음: {args.output_dir}/*/best_model.pth')
        return

    print(f'발견된 체크포인트 {len(ckpt_paths)}개:')
    for p_ in ckpt_paths:
        print(f'  {p_}')

    results = []
    for cp in ckpt_paths:
        print(f'\n평가: {cp}')
        results.append(eval_model(cp, args.data_root, device))

    emotions = results[0]['emotions']
    table = make_comparison_table(results, emotions)
    print('\n' + '=' * 70)
    print(table)
    print('=' * 70)

    table_path = os.path.join(args.output_dir, 'comparison_table.txt')
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(table)
    print(f'\n결과표 저장: {table_path}')

    plot_comparison(results, emotions,
                    save_path=os.path.join(args.output_dir, 'comparison.png'))


if __name__ == '__main__':
    main()
