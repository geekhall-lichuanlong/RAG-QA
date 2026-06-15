/**
 * API 请求封装
 * 所有与后端的通信都通过此模块
 */
const app = getApp()

/**
 * 通用请求方法
 */
function request(path, method = 'GET', data = null) {
  return new Promise((resolve, reject) => {
    wx.request({
      url: app.globalData.apiBaseUrl + path,
      method: method,
      data: data,
      header: {
        'Content-Type': 'application/json',
        'ngrok-skip-browser-warning': 'true'  // ngrok 免费版需要，不然会返回警告页
      },
      success(res) {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          resolve(res.data)
        } else {
          wx.showToast({
            title: res.data?.detail || '请求失败',
            icon: 'none'
          })
          reject(res)
        }
      },
      fail(err) {
        wx.showToast({
          title: '网络连接失败，请检查后端服务',
          icon: 'none'
        })
        reject(err)
      }
    })
  })
}

/**
 * RAG 对话
 */
function chat(question) {
  return request('/api/chat', 'POST', { question })
}

/**
 * 获取题目
 */
function getQuestion(index) {
  return request(`/api/quiz/question?index=${index}`)
}

/**
 * 提交答案并获取解析
 */
function submitAnswer(questionId, question, options, userChoice) {
  return request('/api/quiz/explain', 'POST', {
    question_id: questionId,
    question,
    options,
    user_choice: userChoice
  })
}

/**
 * 获取题库总数
 */
function getQuizTotal() {
  return request('/api/quiz/total')
}

/**
 * 健康检查
 */
function healthCheck() {
  return request('/api/health')
}

module.exports = {
  chat,
  getQuestion,
  submitAnswer,
  getQuizTotal,
  healthCheck
}
