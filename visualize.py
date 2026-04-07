"""
시각화 스크립트

Usage:
  python visualize.py --checkpoint output/densenet121/best_model.pth --data_root New_sample_3

출력 (output/viz/ 폴더):
  gradcam_*.png       - Grad-CAM 히트맵
  edge_samples.png    - 엣지맵 비교 (Canny / Sobel)
  class_gradcam.png   - 클래스별 평균 Grad-CAM
  tsne.png            - t-SNE 피처 분포
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EmotionDataset, SAMPLE_EMOTIONS
from model import EmotionClassifier


# ── Grad-CAM ─────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model: EmotionClassifier):
        self.model = model
        self.backbone = model.backbone_name
        self._feat = None
        self._grad = None
        self._hook_handles = []
        self._register_hooks()

    def _target_layer(self):
        net = self.model.net
        if self.backbone == 'efficientnet_b0':
            return net.features[-1]
        elif self.backbone.startswith('densenet'):
            return net.features.denseblock4
        elif self.backbone.startswith('resnet'):
            return net.layer4
        raise ValueError(f'Grad-CAM target unknown for {self.backbone}')

    def _register_hooks(self):
        layer = self._target_layer()

        def fwd(_, __, output):
            self._feat = output.detach()

        def bwd(_, __, grad_output):
            self._grad = grad_output[0].detach()

        self._hook_handles.append(layer.register_forward_hook(fwd))
        self._hook_handles.append(layer.register_full_backward_hook(bwd))

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()

    def __call__(self, img_tensor: torch.Tensor, class_idx: int = None):
        """
        Args:
            img_tensor: (1, C, H, W) on model's device
            class_idx:  None → argmax (예측 클래스)
        Returns:
            cam: (H, W) numpy, 0~1
            pred_idx: int
        """
        self.model.eval()
        img_tensor = img_tensor.requires_grad_(False)

        logits = self.model(img_tensor)
        pred_idx = int(logits.argmax(1).item()) if class_idx is None else class_idx

        self.model.zero_grad()
        logits[0, pred_idx].backward()

        weights = self._grad.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)
        cam = (weights * self._feat).sum(dim=1).squeeze()       # (H', W')
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy(), pred_idx


def overlay_cam(img_rgb: np.ndarray, cam: np.ndarray, alpha=0.5) -> np.ndarray:
    """cam (H', W') → 히트맵 오버레이된 이미지."""
    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (img_rgb * (1 - alpha) + heatmap * alpha).clip(0, 255).astype(np.uint8)


# ── 엣지 시각화 ──────────────────────────────────────────────────────────────

def visualize_edges(dataset: EmotionDataset, n=6, save_path='output/viz/edge_samples.png'):
    fig, axes = plt.subplots(n, 4, figsize=(14, 3 * n))
    cols = ['원본', 'Canny', 'Sobel-X', 'Sobel-Y']
    for ax, c in zip(axes[0], cols):
        ax.set_title(c, fontsize=10)

    shown = 0
    for item in dataset.samples:
        if shown >= n:
            break
        with open(item['image_path'], 'rb') as f:
            buf = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        canny = cv2.Canny(gray, 50, 150)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sx = np.uint8(np.abs(sx) / np.abs(sx).max() * 255)
        sy = np.uint8(np.abs(sy) / np.abs(sy).max() * 255)

        for ax, data in zip(axes[shown], [img, canny, sx, sy]):
            ax.imshow(data, cmap='gray' if data.ndim == 2 else None)
            ax.axis('off')
        shown += 1

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f'엣지맵 저장: {save_path}')


# ── t-SNE ────────────────────────────────────────────────────────────────────

def extract_features(model: EmotionClassifier, loader: DataLoader, device):
    """classifier 직전 레이어 피처 추출."""
    model.eval()
    feats, labels_all = [], []
    net = model.net

    def _hook(_, __, output):
        feats.append(output.detach().cpu())

    # backbone별 피처 레이어 hook
    if model.backbone_name == 'efficientnet_b0':
        handle = net.avgpool.register_forward_hook(_hook)
    elif model.backbone_name.startswith('densenet'):
        handle = net.features.register_forward_hook(
            lambda m, i, o: feats.append(F.adaptive_avg_pool2d(o, 1).detach().cpu())
        )
    else:  # resnet
        handle = net.avgpool.register_forward_hook(_hook)

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc='피처 추출'):
            imgs = imgs.to(device)
            model(imgs)
            labels_all.extend(lbls.numpy())

    handle.remove()
    feat_arr = torch.cat(feats, dim=0).squeeze().numpy()  # (N, D)
    return feat_arr, np.array(labels_all)


def visualize_tsne(feats, labels, class_names, save_path='output/viz/tsne.png'):
    print('t-SNE 계산 중...')
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb = tsne.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
    for i, name in enumerate(class_names):
        mask = labels == i
        ax.scatter(emb[mask, 0], emb[mask, 1], c=colors[i % len(colors)],
                   label=name, alpha=0.6, s=20)
    ax.legend(fontsize=9)
    ax.set_title('t-SNE Feature Distribution')
    ax.set_xlabel('dim-1'); ax.set_ylabel('dim-2')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f't-SNE 저장: {save_path}')


# ── 클래스별 Grad-CAM ─────────────────────────────────────────────────────────

def visualize_class_gradcam(gradcam, dataset, device, emotions,
                             save_path='output/viz/class_gradcam.png'):
    n_cls = len(emotions)
    class_cams = {i: [] for i in range(n_cls)}
    class_imgs = {i: None for i in range(n_cls)}

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for item in tqdm(dataset.samples[:200], desc='클래스 CAM'):
        lbl = item['label']
        with open(item['image_path'], 'rb') as f:
            buf = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        tensor = torch.from_numpy(
            ((img / 255.0 - mean) / std).transpose(2, 0, 1).astype(np.float32)
        ).unsqueeze(0).to(device)

        cam, _ = gradcam(tensor, class_idx=lbl)
        class_cams[lbl].append(cv2.resize(cam, (224, 224)))
        if class_imgs[lbl] is None:
            class_imgs[lbl] = img

    fig, axes = plt.subplots(2, n_cls, figsize=(4 * n_cls, 8))
    for i, emo in enumerate(emotions):
        axes[0, i].set_title(emo, fontsize=11)
        if class_imgs[i] is not None:
            axes[0, i].imshow(class_imgs[i])
            if class_cams[i]:
                avg_cam = np.mean(class_cams[i], axis=0)
                axes[1, i].imshow(overlay_cam(class_imgs[i], avg_cam))
        axes[0, i].axis('off')
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('원본', fontsize=10)
    axes[1, 0].set_ylabel('평균 Grad-CAM', fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f'클래스 Grad-CAM 저장: {save_path}')


# ── 개별 샘플 Grad-CAM ────────────────────────────────────────────────────────

def visualize_sample_gradcam(gradcam, dataset, device, emotions, n=8,
                              save_path='output/viz/gradcam_samples.png'):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    results = []

    for item in dataset.samples:
        if len(results) >= n:
            break
        with open(item['image_path'], 'rb') as f:
            buf = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        tensor = torch.from_numpy(
            ((img / 255.0 - mean) / std).transpose(2, 0, 1).astype(np.float32)
        ).unsqueeze(0).to(device)

        cam, pred = gradcam(tensor)
        results.append((img, cam, item['label'], pred))

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for i, (img, cam, gt, pred) in enumerate(results):
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'GT:{emotions[gt]}', fontsize=8)
        axes[0, i].axis('off')
        axes[1, i].imshow(overlay_cam(img, cam))
        axes[1, i].set_title(f'Pred:{emotions[pred]}', fontsize=8,
                              color='green' if gt == pred else 'red')
        axes[1, i].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f'샘플 Grad-CAM 저장: {save_path}')


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_root',  default='New_sample_3')
    p.add_argument('--output_dir', default='output/viz')
    p.add_argument('--batch_size', type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location=device)
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

    val_ds = EmotionDataset(
        args.data_root, emotions=emotions, split='val', augment=False,
        use_clahe=use_clahe, use_edge=use_edge, use_align=use_align,
    )
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=0)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 엣지맵 시각화 (원본 데이터 사용)
    raw_ds = EmotionDataset(args.data_root, emotions=emotions, split='val', augment=False)
    visualize_edges(raw_ds, n=6,
                    save_path=os.path.join(args.output_dir, 'edge_samples.png'))

    # 2. 샘플 Grad-CAM
    gc = GradCAM(model)
    visualize_sample_gradcam(gc, raw_ds, device, emotions, n=8,
                              save_path=os.path.join(args.output_dir, 'gradcam_samples.png'))

    # 3. 클래스별 평균 Grad-CAM
    visualize_class_gradcam(gc, raw_ds, device, emotions,
                             save_path=os.path.join(args.output_dir, 'class_gradcam.png'))
    gc.remove_hooks()

    # 4. t-SNE
    feats, labels = extract_features(model, val_loader, device)
    visualize_tsne(feats, labels, emotions,
                   save_path=os.path.join(args.output_dir, 'tsne.png'))

    print(f'\n모든 시각화 완료 → {args.output_dir}/')


if __name__ == '__main__':
    main()
