import { useState } from 'react'
import AnalyzeTab    from './components/AnalyzeTab'
import ModelCompareTab from './components/ModelCompareTab'
import PipelineTab   from './components/PipelineTab'

const TABS = [
  {
    id: 'analyze',
    label: '분석',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
      </svg>
    ),
  },
  {
    id: 'models',
    label: '모델 성능',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>
      </svg>
    ),
  },
  {
    id: 'pipeline',
    label: '파이프라인',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="2" y="7" width="6" height="10" rx="1"/>
        <rect x="9" y="4" width="6" height="16" rx="1"/>
        <rect x="16" y="9" width="6" height="8" rx="1"/>
      </svg>
    ),
  },
]

const TAB_LABELS = {
  analyze:  '감정인식 AI',
  models:   '모델 성능 비교',
  pipeline: '파이프라인 시각화',
}

export default function App() {
  const [tab, setTab] = useState('analyze')

  return (
    <div className="app">
      <header className="app-header">
        <h1>{TAB_LABELS[tab]}</h1>
      </header>

      <main className="app-content">
        {tab === 'analyze'   && <AnalyzeTab />}
        {tab === 'models'    && <ModelCompareTab />}
        {tab === 'pipeline'  && <PipelineTab />}
      </main>

      <nav className="tab-bar">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`tab-btn${tab === t.id ? ' active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            {t.icon}
            {t.label}
          </button>
        ))}
      </nav>
    </div>
  )
}
