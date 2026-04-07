"""
AI Hub 감정인식 데이터셋 로더 (데이터셋 82)

전처리 옵션:
  use_clahe  : CLAHE 히스토그램 평활화 (조도 불균형 보정)
  use_edge   : Canny 엣지를 4번째 채널로 추가 (RGB → RGBE)
  use_align  : mediapipe로 눈 위치 기반 얼굴 정렬
"""
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

EMOTIONS = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
SAMPLE_EMOTIONS = ['기쁨', '당황', '분노', '상처']

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


def _consensus_box(item: dict):
    keys = ['annot_A', 'annot_B', 'annot_C']
    boxes = [item[k]['boxes'] for k in keys if k in item and item[k]]
    if not boxes:
        return None
    return (
        np.mean([b['minX'] for b in boxes]),
        np.mean([b['minY'] for b in boxes]),
        np.mean([b['maxX'] for b in boxes]),
        np.mean([b['maxY'] for b in boxes]),
    )


def _load_json(path: str) -> list:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """CLAHE를 LAB L채널에 적용해 조도 불균형 보정."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def extract_edge(img_rgb: np.ndarray) -> np.ndarray:
    """Canny 엣지맵 반환 (H, W), uint8 0~255."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 50, 150)


def align_face(img_rgb: np.ndarray) -> np.ndarray:
    """
    mediapipe FaceMesh로 눈 랜드마크를 검출해 얼굴 정렬.
    검출 실패 시 원본 반환.
    """
    try:
        import mediapipe as mp
        mp_face = mp.solutions.face_mesh
        with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1,
                               refine_landmarks=True) as fm:
            res = fm.process(img_rgb)
            if not res.multi_face_landmarks:
                return img_rgb
            lm = res.multi_face_landmarks[0].landmark
            h, w = img_rgb.shape[:2]
            # 왼눈 중심(133), 오른눈 중심(362) 인덱스
            le = np.array([lm[133].x * w, lm[133].y * h])
            re = np.array([lm[362].x * w, lm[362].y * h])
            angle = np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0]))
            cx, cy = w / 2, h / 2
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            return cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_LINEAR)
    except Exception:
        return img_rgb


class EmotionDataset(Dataset):
    """
    한국인 감정인식 이미지 데이터셋.

    Args:
        data_root  : New_sample_3 폴더 경로
        emotions   : 사용할 감정 목록
        split      : 'train' | 'val' | 'all'
        val_ratio  : validation 비율
        image_size : 크롭 후 리사이즈 크기
        augment    : 학습 데이터 증강 여부
        use_clahe  : CLAHE 히스토그램 평활화
        use_edge   : 엣지 채널 추가 (출력 4ch)
        use_align  : mediapipe 얼굴 정렬
        seed       : random seed
    """

    def __init__(
        self,
        data_root: str,
        emotions: list = None,
        split: str = 'train',
        val_ratio: float = 0.2,
        image_size: int = 224,
        augment: bool = True,
        use_clahe: bool = False,
        use_edge: bool = False,
        use_align: bool = False,
        seed: int = 42,
    ):
        self.data_root = data_root
        self.emotions = emotions or SAMPLE_EMOTIONS
        self.split = split
        self.image_size = image_size
        self.use_clahe = use_clahe
        self.use_edge = use_edge
        self.use_align = use_align

        self.samples = self._build_samples(val_ratio, seed)

        base_transforms = [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
        ]
        aug_transforms = [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.Resize((image_size, image_size)),
        ]
        tail = [transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD)]

        self.rgb_transform = transforms.Compose(
            (aug_transforms if split == 'train' and augment else base_transforms) + tail
        )

    def _build_samples(self, val_ratio, seed):
        all_samples = []
        for emotion in self.emotions:
            img_dir   = os.path.join(self.data_root, '원천데이터',   f'EMOIMG_{emotion}_SAMPLE')
            label_dir = os.path.join(self.data_root, '라벨링데이터', f'EMOIMG_{emotion}_SAMPLE')
            if not os.path.isdir(img_dir) or not os.path.isdir(label_dir):
                continue
            json_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
            if not json_files:
                continue
            records = _load_json(os.path.join(label_dir, json_files[0]))
            existing = set(os.listdir(img_dir))
            for rec in records:
                fname = rec.get('filename', '')
                if fname not in existing:
                    continue
                box = _consensus_box(rec)
                if box is None:
                    continue
                all_samples.append({
                    'image_path': os.path.join(img_dir, fname),
                    'label': self.emotions.index(emotion),
                    'emotion': emotion,
                    'box': box,
                    'gender': rec.get('gender', ''),
                    'age': rec.get('age', -1),
                })

        rng = np.random.RandomState(seed)
        idx = np.arange(len(all_samples))
        rng.shuffle(idx)
        cut = int(len(idx) * (1 - val_ratio))
        if self.split == 'train':
            idx = idx[:cut]
        elif self.split == 'val':
            idx = idx[cut:]
        return [all_samples[i] for i in idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 한글 경로 대응
        with open(item['image_path'], 'rb') as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"이미지 읽기 실패: {item['image_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 얼굴 크롭 (10% 패딩)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = item['box']
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - bw * 0.1))
        y1 = max(0, int(y1 - bh * 0.1))
        x2 = min(w, int(x2 + bw * 0.1))
        y2 = min(h, int(y2 + bh * 0.1))
        face = img[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else img

        # 얼굴 정렬
        if self.use_align:
            face = align_face(face)

        # CLAHE
        if self.use_clahe:
            face = apply_clahe(face)

        # RGB 텐서 (3, H, W)
        rgb_tensor = self.rgb_transform(face)  # (3, H, W)

        # 엣지 채널 추가 → (4, H, W)
        if self.use_edge:
            face_resized = cv2.resize(face, (self.image_size, self.image_size))
            edge = extract_edge(face_resized).astype(np.float32) / 255.0
            edge_tensor = torch.from_numpy(edge).unsqueeze(0)  # (1, H, W)
            tensor = torch.cat([rgb_tensor, edge_tensor], dim=0)
        else:
            tensor = rgb_tensor

        return tensor, item['label']

    def class_counts(self) -> dict:
        from collections import Counter
        return dict(Counter(s['emotion'] for s in self.samples))
