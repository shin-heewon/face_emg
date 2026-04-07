import axios from 'axios'

// 환경변수 VITE_API_BASE가 설정되면 외부 백엔드를 사용 (Vercel 배포 시)
// 미설정 시 Vite 프록시를 통해 로컬 서버로 라우팅
const BASE = (import.meta.env.VITE_API_BASE ?? '') + '/api'

export const api = {
  health: () => axios.get(`${BASE}/health`),

  models: () => axios.get(`${BASE}/models`),

  analyze: (file, modelId = 'densenet121') => {
    const form = new FormData()
    form.append('file', file)
    form.append('model_id', modelId)
    return axios.post(`${BASE}/analyze`, form)
  },

  analyzeCompare: (file) => {
    const form = new FormData()
    form.append('file', file)
    return axios.post(`${BASE}/analyze/compare`, form)
  },

  analyzeBase64: (imageB64, modelId = 'densenet121', compare = false) =>
    axios.post(`${BASE}/analyze/base64`, { image_b64: imageB64, model_id: modelId, compare }),

  pipelineImageUrl: (name) => `${BASE}/pipeline/${name}`,
}
