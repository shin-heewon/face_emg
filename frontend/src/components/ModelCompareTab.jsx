import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, Cell,
} from 'recharts'
import { api } from '../api'

const EMOTIONS = ['기쁨', '당황', '분노', '상처']
const EMOTION_EMOJI = { 기쁨: '😄', 당황: '😳', 분노: '😡', 상처: '😢' }

// 전체 정확도 기준 색상 (70~100%)
function accColor(acc) {
  if (acc >= 0.87) return '#22c55e'
  if (acc >= 0.83) return '#f59e0b'
  return '#ef4444'
}

// ── 커스텀 tooltip ─────────────────────────────────────────────────────────────
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#fff', border: '1px solid #e4e4e7',
      borderRadius: 8, padding: '8px 12px', fontSize: 12,
    }}>
      <p style={{ fontWeight: 600, marginBottom: 4 }}>{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.fill }}>
          {p.name}: {(p.value * 100).toFixed(1)}%
        </p>
      ))}
    </div>
  )
}

// ── 모델명 줄임 ────────────────────────────────────────────────────────────────
function shortLabel(label) {
  return label
    .replace('EfficientNet-B0', 'Eff-B0')
    .replace('DenseNet121', 'Dense121')
    .replace(' + CLAHE + Edge', '+CE')
}

export default function ModelCompareTab() {
  const [models, setModels]   = useState([])
  const [view, setView]       = useState('table') // 'table' | 'chart'
  const [chartBy, setChartBy] = useState('acc')   // 'acc' | 'f1'
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(null)

  useEffect(() => {
    api.models()
      .then(r => setModels(r.data.models))
      .catch(e => setError(
        e?.code === 'ERR_NETWORK'
          ? '서버에 연결할 수 없습니다. 백엔드(포트 8001)가 실행 중인지 확인하세요.'
          : `오류: ${e?.response?.data?.detail ?? e?.message}`
      ))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', padding: 48 }}>
        <div className="spinner" style={{ borderTopColor: 'var(--primary)', borderColor: '#e4e4e7' }} />
      </div>
    )
  }

  if (error) {
    return <div className="section"><div className="notice">{error}</div></div>
  }

  // recharts 데이터 구성
  const accData = models.map(m => ({
    name:    shortLabel(m.label),
    전체Acc: m.val_acc,
    color:   m.color,
    loaded:  m.loaded,
  }))

  const f1Data = EMOTIONS.map(emo => {
    const row = { name: `${EMOTION_EMOJI[emo]} ${emo}` }
    models.forEach(m => {
      row[shortLabel(m.label)] = m.f1_per[emo] ?? 0
    })
    return row
  })

  return (
    <div>
      {/* 뷰 전환 */}
      <div className="section">
        <div className="chip-group">
          <button className={`chip${view === 'table' ? ' active' : ''}`} onClick={() => setView('table')}>
            📋 표
          </button>
          <button className={`chip${view === 'chart' ? ' active' : ''}`} onClick={() => setView('chart')}>
            📊 차트
          </button>
        </div>
      </div>

      {/* ── 표 뷰 ───────────────────────────────────────────────────── */}
      {view === 'table' && (
        <div className="section" style={{ paddingTop: 0, overflowX: 'auto' }}>
          <table className="metric-table">
            <thead>
              <tr>
                <th>모델</th>
                <th>전체Acc</th>
                {EMOTIONS.map(e => <th key={e}>{EMOTION_EMOJI[e]}{e}</th>)}
                <th>상태</th>
              </tr>
            </thead>
            <tbody>
              {models.map(m => (
                <tr key={m.id}>
                  <td>
                    <div className="model-name-cell">
                      <div className="model-dot" style={{ background: m.color }} />
                      <span style={{ fontSize: 12, lineHeight: 1.3 }}>{m.label}</span>
                    </div>
                  </td>
                  <td>
                    <span
                      className="acc-badge"
                      style={{ background: accColor(m.val_acc) }}
                    >
                      {(m.val_acc * 100).toFixed(1)}%
                    </span>
                  </td>
                  {EMOTIONS.map(e => (
                    <td key={e} className="f1-cell">
                      {m.f1_per[e]?.toFixed(3) ?? '-'}
                    </td>
                  ))}
                  <td style={{ fontSize: 12 }}>
                    {m.loaded
                      ? <span style={{ color: '#22c55e' }}>✓ 로드</span>
                      : <span style={{ color: '#ef4444' }}>✗ 없음</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* 범례 설명 */}
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 10, lineHeight: 1.5 }}>
            * F1 수치는 val set(420장) 기준. 녹색 ≥ 87%, 노란색 ≥ 83%, 빨간색 미만.
          </p>
        </div>
      )}

      {/* ── 차트 뷰 ─────────────────────────────────────────────────── */}
      {view === 'chart' && (
        <div className="section" style={{ paddingTop: 0 }}>
          <div className="chip-group" style={{ marginBottom: 16 }}>
            <button className={`chip${chartBy === 'acc' ? ' active' : ''}`} onClick={() => setChartBy('acc')}>
              전체 Accuracy
            </button>
            <button className={`chip${chartBy === 'f1' ? ' active' : ''}`} onClick={() => setChartBy('f1')}>
              클래스별 F1
            </button>
          </div>

          {chartBy === 'acc' && (
            <>
              <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12 }}>모델별 Validation Accuracy</p>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={accData} margin={{ top: 4, right: 8, left: -20, bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 10 }}
                    angle={-25}
                    textAnchor="end"
                    interval={0}
                  />
                  <YAxis
                    domain={[0.75, 1.0]}
                    tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                    tick={{ fontSize: 10 }}
                  />
                  <Tooltip formatter={v => `${(v * 100).toFixed(1)}%`} />
                  <Bar dataKey="전체Acc" radius={[4, 4, 0, 0]}>
                    {accData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </>
          )}

          {chartBy === 'f1' && (
            <>
              <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12 }}>감정 클래스별 F1-score</p>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={f1Data} margin={{ top: 4, right: 8, left: -20, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0.6, 1.0]} tickFormatter={v => v.toFixed(1)} tick={{ fontSize: 10 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  {models.map(m => (
                    <Bar
                      key={m.id}
                      dataKey={shortLabel(m.label)}
                      fill={m.color}
                      radius={[3, 3, 0, 0]}
                    />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </>
          )}
        </div>
      )}

      {/* 모델 카드 */}
      <div className="section" style={{ paddingTop: 0 }}>
        <p style={{ fontSize: 13, fontWeight: 600, marginBottom: 10 }}>모델 정보</p>
        {models.map(m => (
          <div key={m.id} style={{
            display: 'flex', alignItems: 'center', gap: 10,
            padding: '10px 12px', borderRadius: 10,
            border: '1px solid var(--border)', marginBottom: 8,
          }}>
            <div className="model-dot" style={{ background: m.color, width: 12, height: 12 }} />
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 600 }}>{m.label}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{m.description}</div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: 16, fontWeight: 700, color: accColor(m.val_acc) }}>
                {(m.val_acc * 100).toFixed(1)}%
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                {m.loaded ? '✓ 로드됨' : '✗ 미로드'}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
