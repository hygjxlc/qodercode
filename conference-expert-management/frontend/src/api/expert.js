import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 10000
})

// 专家管理 API
export const expertApi = {
  // 获取所有专家（分页）
  getExperts(page = 0, size = 10) {
    return api.get(`/experts?page=${page}&size=${size}`)
  },

  // 获取单个专家
  getExpertById(id) {
    return api.get(`/experts/${id}`)
  },

  // 创建专家
  createExpert(data) {
    return api.post('/experts', data)
  },

  // 更新专家
  updateExpert(id, data) {
    return api.put(`/experts/${id}`, data)
  },

  // 删除专家
  deleteExpert(id) {
    return api.delete(`/experts/${id}`)
  },

  // 搜索专家
  searchExperts(keyword) {
    return api.get(`/experts/search?keyword=${encodeURIComponent(keyword)}`)
  },

  // 筛选专家
  filterExperts(params) {
    const query = new URLSearchParams(params).toString()
    return api.get(`/experts/filter?${query}`)
  }
}

// 联系记录 API
export const contactApi = {
  // 获取专家的联系记录
  getContactRecords(expertId) {
    return api.get(`/contacts/expert/${expertId}`)
  },

  // 添加联系记录
  addContactRecord(data) {
    return api.post('/contacts', data)
  }
}

// 会务记录 API
export const conferenceApi = {
  // 获取会务记录
  getConferenceRecords(expertId) {
    return api.get(`/conferences/expert/${expertId}`)
  },

  // 添加会务记录
  addConferenceRecord(data) {
    return api.post('/conferences', data)
  },

  // 添加反馈
  addFeedback(id, feedback) {
    return api.post(`/conferences/${id}/feedback`, { feedback })
  }
}

// Excel 导入导出 API
export const excelApi = {
  // 导出专家
  exportExperts() {
    return api.get('/excel/export', { responseType: 'blob' })
  },

  // 下载模板
  downloadTemplate() {
    return api.get('/excel/template', { responseType: 'blob' })
  },

  // 导入专家
  importExperts(file) {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/excel/import', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
  }
}

export default api
