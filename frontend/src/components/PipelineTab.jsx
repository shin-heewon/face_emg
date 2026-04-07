import { useState } from 'react'
import { api } from '../api'

// ── 파이프라인 단계 정의 ──────────────────────────────────────────────────────
const STEPS = [
  {
    icon:  '📂',
    bg:    '#eff6ff',
    title: '1. 이미지 입력',
    desc:  'AI Hub 한국인 감정인식 데이터셋 (Dataset 82). 7개 감정 클래스, ~488K 이미지. 어노테이터 3인의 합의 bounding box 사용.',
  },
  {
    icon:  '✂️',
    bg:    '#fef9c3',
    title: '2. 얼굴 크롭',
    desc:  '3인 어노테이터 bounding box의 평균 좌표로 얼굴 영역 추출. 10% 여백 추가 패딩 적용.',
  },
  {
    icon:  '🌅',
    bg:    '#f0fdf4',
    title: '3. CLAHE (선택)',
    desc:  'LAB 색공간 L채널에 CLAHE 적용. 조도 불균형 환경의 명암 대비 강화. clipLimit=2.0, tileGrid=8×8.',
  },
  {
    icon:  '🔲',
    bg:    '#fdf4ff',
    title: '4. 엣지 채널 (선택)',
    desc:  'Canny 엣지맵(50, 150)을 4번째 채널로 추가 (RGB → RGBE). 모델 첫 Conv를 3→4ch로 확장.',
  },
  {
    icon:  '🔀',
    bg:    '#fff7ed',
    title: '5. 증강 (학습 시)',
    desc:  '랜덤 수평 뒤집기, ColorJitter (brightness±0.3), 랜덤 회전 ±15°. 검증 시에는 비적용.',
  },
  {
    icon:  '🧠',
    bg:    '#fef2f2',
    title: '6. 모델 추론',
    desc:  'ImageNet 사전학습 DenseNet121 / EfficientNet-B0. FC 레이어 교체 후 fine-tuning. AdamW + CosineAnnealingLR.',
  },
  {
    icon:  '📊',
    bg:    '#f0f9ff',
    title: '7. 결과 출력',
    desc:  '4개 감정 클래스 (기쁨/당황/분노/상처) Softmax 확률. 최고 성능: DenseNet121 87.6% Acc.',
  },
]

// ── 시각화 이미지 목록 ────────────────────────────────────────────────────────
const VIZ_ITEMS = [
  {
    key:   'edge_samples',
    label: '엣지맵',
    emoji: '🔲',
    desc:  '원본 · Canny · Sobel-X · Sobel-Y 비교. 얼굴 윤곽선 추출 파이프라인 확인.',
  },
  {
    key:   'gradcam_samples',
    label: 'Grad-CAM',
    emoji: '🔥',
    desc:  '예측 클래스에 대한 역전파 기반 활성화 히트맵. 모델이 어느 부위를 보는지 시각화.',
  },
  {
    key:   'class_gradcam',
    label: '클래스별 CAM',
    emoji: '🎯',
    desc:  '각 감정 클래스별 평균 Grad-CAM. 감정별 특징 영역 차이 비교.',
  },
  {
    key:   'tsne',
    label: 't-SNE',
    emoji: '🌐',
    desc:  'Classifier 직전 피처를 2D 투영. 클래스별 분리도 및 클러스터링 확인.',
  },
  {
    key:   'comparison',
    label: '모델 비교',
    emoji: '📈',
    desc:  '4개 모델의 전체 Accuracy 및 클래스별 F1 비교 차트.',
  },
]

// ── 단계별 파이프라인 ─────────────────────────────────────────────────────────
function StepList() {
  return (
    <div>
      {STEPS.map((s, i) => (
        <div key={i} className="pipeline-step">
          <div className="step-icon" style={{ background: s.bg }}>{s.icon}</div>
          <div className="step-info">
            <h4>{s.title}</h4>
            <p>{s.desc}</p>
          </div>
        </div>
      ))}
    </div>
  )
}

// ── 시각화 이미지 뷰어 ────────────────────────────────────────────────────────
function VizViewer() {
  const [selected, setSelected] = useState('edge_samples')
  const [imgError, setImgError] = useState(false)

  const current = VIZ_ITEMS.find(v => v.key === selected)
  const url = api.pipelineImageUrl(selected)

  const handleSelect = (key) => {
    setSelected(key)
    setImgError(false)
  }

  return (
    <div>
      <div className="viz-selector">
        {VIZ_ITEMS.map(v => (
          <button
            key={v.key}
            className={`viz-btn${selected === v.key ? ' active' : ''}`}
            onClick={() => handleSelect(v.key)}
          >
            {v.emoji} {v.label}
          </button>
        ))}
      </div>

      {current && (
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
          {current.desc}
        </p>
      )}

      <div className="viz-image-wrap">
        {imgError ? (
          <div style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)' }}>
            <div style={{ fontSize: 32, marginBottom: 8 }}>🖼️</div>
            <p style={{ fontSize: 13 }}>이미지가 없습니다.</p>
            <p style={{ fontSize: 11, marginTop: 4 }}>학습 후 visualize.py 실행 시 생성됩니다.</p>
          </div>
        ) : (
          <img
            key={url}
            src={url}
            alt={current?.label}
            onError={() => setImgError(true)}
          />
        )}
      </div>
    </div>
  )
}

// ── 메인 파이프라인 탭 ────────────────────────────────────────────────────────
export default function PipelineTab() {
  const [view, setView] = useState('steps')  // 'steps' | 'viz'

  return (
    <div>
      <div className="section">
        <div className="chip-group">
          <button className={`chip${view === 'steps' ? ' active' : ''}`} onClick={() => setView('steps')}>
            🔄 처리 단계
          </button>
          <button className={`chip${view === 'viz' ? ' active' : ''}`} onClick={() => setView('viz')}>
            🖼️ 시각화 결과
          </button>
        </div>
      </div>

      <div className="section" style={{ paddingTop: 0 }}>
        {view === 'steps' && <StepList />}
        {view === 'viz'   && <VizViewer />}
      </div>
    </div>
  )
}
