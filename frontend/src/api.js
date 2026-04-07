import axios from 'axios'

const BASE = '/api'

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
