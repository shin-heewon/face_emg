import { useRef, useState, useCallback } from 'react'
import { api } from '../api'

const EMOTIONS = ['기쁨', '당황', '분노', '상처']
const EMOTION_EMOJI = { 기쁨: '😄', 당황: '😳', 분노: '😡', 상처: '😢' }
const EMOTION_COLOR = { 기쁨: '#f59e0b', 당황: '#f97316', 분노: '#ef4444', 상처: '#6366f1' }

const MODELS = [
  { id: 'densenet121',             label: 'DenseNet121',             color: '#4F86C6' },
  { id: 'densenet121_clahe_edge',  label: 'DenseNet121+CLAHE+Edge', color: '#57B894' },
  { id: 'efficientnet_b0',         label: 'EfficientNet-B0',         color: '#F4845F' },
  { id: 'efficientnet_b0_clahe_edge', label: 'EfficientNet+CLAHE+Edge', color: '#9b59b6' },
]

// ── 신뢰도 바 ─────────────────────────────────────────────────────────────────
function ConfidenceBar({ emotion, score, highlight }) {
  const color = EMOTION_COLOR[emotion] || '#6366f1'
  return (
    <div className="conf-bar-row">
      <span className="conf-bar-label">{EMOTION_EMOJI[emotion]} {emotion}</span>
      <div className="conf-bar-track">
        <div
          className="conf-bar-fill"
          style={{ width: `${(score * 100).toFixed(1)}%`, background: color, opacity: highlight ? 1 : 0.45 }}
        />
      </div>
      <span className="conf-bar-pct">{(score * 100).toFixed(1)}%</span>
    </div>
  )
}

// ── 단일 모델 결과 ─────────────────────────────────────────────────────────────
function SingleResult({ result }) {
  const { emotion, emoji, confidence, scores } = result
  const color = EMOTION_COLOR[emotion] || '#6366f1'
  return (
    <div className="slide-up">
      <div style={{ textAlign: 'center', padding: '20px 0 16px' }}>
        <div style={{ fontSize: 56, lineHeight: 1.1, marginBottom: 8 }}>{emoji}</div>
        <div style={{ fontSize: 26, fontWeight: 800, color }}>{emotion}</div>
        <div style={{ fontSize: 14, color: '#71717a', marginTop: 4 }}>
          신뢰도 {(confidence * 100).toFixed(1)}%
        </div>
      </div>
      <div style={{ padding: '0 16px 16px' }}>
        {EMOTIONS.map(e => (
          <ConfidenceBar key={e} emotion={e} score={scores[e] ?? 0} highlight={e === emotion} />
        ))}
      </div>
    </div>
  )
}

// ── 비교 모드 결과 ─────────────────────────────────────────────────────────────
function CompareResult({ results }) {
  return (
    <div className="slide-up" style={{ padding: '12px 16px 16px' }}>
      <p style={{ fontSize: 12, color: '#71717a', marginBottom: 12 }}>
        {results.length}개 모델 동시 분석
      </p>
      {results.map(r => (
        <div key={r.model_id} className="compare-card">
          <div className="compare-card-header">
            <div className="model-dot" style={{ background: r.color }} />
            <span className="compare-card-title">{r.model_label}</span>
            <span style={{ fontSize: 12, color: '#71717a' }}>{r.infer_ms}ms</span>
          </div>
          <div className="compare-card-top-emotion">
            <span>{r.emoji}</span>
            <span style={{ color: EMOTION_COLOR[r.emotion] }}>{r.emotion}</span>
            <span style={{ fontSize: 14, color: '#71717a', fontWeight: 500 }}>
              {(r.confidence * 100).toFixed(1)}%
            </span>
          </div>
          {EMOTIONS.map(e => (
            <ConfidenceBar key={e} emotion={e} score={r.scores[e] ?? 0} highlight={e === r.emotion} />
          ))}
        </div>
      ))}
    </div>
  )
}

// ── 메인 분석 탭 ───────────────────────────────────────────────────────────────
export default function AnalyzeTab() {
  const [mode, setMode]           = useState('upload')    // 'upload' | 'camera'
  const [compareMode, setCompare] = useState(false)
  const [selectedModel, setModel] = useState('densenet121')
  const [preview, setPreview]     = useState(null)
  const [file, setFile]           = useState(null)
  const [loading, setLoading]     = useState(false)
  const [result, setResult]       = useState(null)
  const [error, setError]         = useState(null)
  const [faceB64, setFaceB64]     = useState(null)
  const [faceDetected, setFaceDetected] = useState(null)

  // 웹캠
  const videoRef    = useRef(null)
  const streamRef   = useRef(null)
  const [camActive, setCamActive] = useState(false)

  // 파일 선택
  const onFileChange = (e) => {
    const f = e.target.files[0]
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError(null)
    setFaceB64(null)
  }

  // 카메라 시작
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
      }
      setCamActive(true)
      setResult(null)
      setError(null)
    } catch (e) {
      setError('카메라 접근 권한이 필요합니다.')
    }
  }, [])

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    setCamActive(false)
  }, [])

  // 카메라 캡처
  const captureCamera = useCallback(() => {
    const video = videoRef.current
    if (!video) return
    const canvas = document.createElement('canvas')
    canvas.width  = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)
    canvas.toBlob(blob => {
      setFile(blob)
      setPreview(canvas.toDataURL('image/jpeg'))
      setResult(null)
      setFaceB64(null)
      stopCamera()
    }, 'image/jpeg', 0.92)
  }, [stopCamera])

  // 탭 전환
  const switchMode = (m) => {
    setMode(m)
    setResult(null)
    setError(null)
    setPreview(null)
    setFile(null)
    setFaceB64(null)
    if (m !== 'camera' && camActive) stopCamera()
  }

  // 분석 실행
  const analyze = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    setFaceB64(null)
    try {
      let res
      if (compareMode) {
        res = await api.analyzeCompare(file)
        setResult({ type: 'compare', data: res.data.results })
        setFaceB64(res.data.face_b64)
        setFaceDetected(res.data.face_detected)
      } else {
        res = await api.analyze(file, selectedModel)
        setResult({ type: 'single', data: res.data })
        setFaceB64(res.data.face_b64)
        setFaceDetected(res.data.face_detected)
      }
    } catch (e) {
      if (e?.code === 'ERR_NETWORK' || e?.message === 'Network Error') {
        setError('서버에 연결할 수 없습니다. 백엔드 서버(포트 8001)가 실행 중인지 확인하세요.')
      } else if (e?.response?.status === 503) {
        setError('모델이 아직 로딩 중입니다. 잠시 후 다시 시도하세요.')
      } else {
        const detail = e?.response?.data?.detail
        setError(detail ? `오류: ${detail}` : `서버 오류 (${e?.response?.status ?? 'unknown'}) — 브라우저 콘솔 F12를 확인하세요.`)
      }
      console.error('[analyze error]', e)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      {/* 입력 모드 전환 */}
      <div className="section">
        <div className="chip-group">
          <button className={`chip${mode === 'upload' ? ' active' : ''}`} onClick={() => switchMode('upload')}>
            📁 파일 업로드
          </button>
          <button className={`chip${mode === 'camera' ? ' active' : ''}`} onClick={() => switchMode('camera')}>
            📷 카메라
          </button>
        </div>
      </div>

      {/* 이미지/카메라 미리보기 */}
      <div className="section" style={{ paddingTop: 0 }}>
        <div className="preview-wrap">
          {mode === 'camera' ? (
            camActive ? (
              <>
                <video ref={videoRef} autoPlay playsInline muted style={{ transform: 'scaleX(-1)' }} />
                <div className="preview-overlay" />
              </>
            ) : preview ? (
              <img src={preview} alt="캡처 이미지" />
            ) : (
              <span style={{ fontSize: 40 }}>📷</span>
            )
          ) : preview ? (
            <img src={preview} alt="미리보기" />
          ) : (
            <span style={{ fontSize: 40 }}>🖼️</span>
          )}
        </div>
      </div>

      {/* 카메라 컨트롤 */}
      {mode === 'camera' && (
        <div className="section" style={{ paddingTop: 0, display: 'flex', gap: 8 }}>
          {!camActive ? (
            <button className="btn btn-outline btn-full" onClick={startCamera}>카메라 시작</button>
          ) : (
            <>
              <button className="btn btn-primary" style={{ flex: 2 }} onClick={captureCamera}>📸 촬영</button>
              <button className="btn btn-outline" style={{ flex: 1 }} onClick={stopCamera}>취소</button>
            </>
          )}
        </div>
      )}

      {/* 파일 선택 */}
      {mode === 'upload' && (
        <div className="section" style={{ paddingTop: 0 }}>
          <label className="btn btn-outline btn-full" style={{ cursor: 'pointer' }}>
            {preview ? '🔄 다른 이미지 선택' : '📂 이미지 선택'}
            <input type="file" accept="image/*" style={{ display: 'none' }} onChange={onFileChange} />
          </label>
        </div>
      )}

      {/* 분석 옵션 */}
      <div className="section" style={{ paddingTop: 0 }}>
        <div className="chip-group" style={{ marginBottom: 10 }}>
          <button className={`chip${!compareMode ? ' active' : ''}`} onClick={() => setCompare(false)}>
            단일 모델
          </button>
          <button className={`chip${compareMode ? ' active' : ''}`} onClick={() => setCompare(true)}>
            전체 비교
          </button>
        </div>

        {!compareMode && (
          <select
            value={selectedModel}
            onChange={e => setModel(e.target.value)}
            style={{
              width: '100%', padding: '10px 12px', borderRadius: 10,
              border: '1.5px solid var(--border)', fontSize: 14,
              background: 'var(--surface)', color: 'var(--text)',
              marginBottom: 10, appearance: 'none',
            }}
          >
            {MODELS.map(m => (
              <option key={m.id} value={m.id}>{m.label}</option>
            ))}
          </select>
        )}

        <button
          className="btn btn-primary btn-full"
          disabled={!file || loading}
          onClick={analyze}
        >
          {loading ? <><div className="spinner" /> 분석 중...</> : '🔍 감정 분석'}
        </button>
      </div>

      {/* 에러 */}
      {error && (
        <div className="section" style={{ paddingTop: 0 }}>
          <div className="notice">{error}</div>
        </div>
      )}

      {/* 얼굴 검출 정보 */}
      {faceDetected !== null && (
        <div className="section" style={{ paddingTop: 0 }}>
          <div className={`notice${faceDetected ? ' info' : ''}`}>
            {faceDetected
              ? '✅ 얼굴이 검출되었습니다'
              : '⚠️ 얼굴 미검출 — 중앙 크롭으로 분석했습니다'}
          </div>
        </div>
      )}

      {/* 얼굴 크롭 미리보기 */}
      {faceB64 && (
        <div className="section" style={{ paddingTop: 0 }}>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 6 }}>분석에 사용된 얼굴 영역</p>
          <img
            src={`data:image/jpeg;base64,${faceB64}`}
            alt="분석 얼굴"
            style={{ width: 100, height: 100, objectFit: 'cover', borderRadius: 12, border: '1px solid var(--border)' }}
          />
        </div>
      )}

      {/* 결과 */}
      {result && (
        result.type === 'single'
          ? <SingleResult result={result.data} />
          : <CompareResult results={result.data} />
      )}
    </div>
  )
}
