const api = require('../../utils/api')

Page({
  data: {
    messages: [],
    inputText: '',
    loading: false
  },

  onLoad() {
    // 页面加载时发送欢迎消息
    this.setData({
      messages: [{
        role: 'assistant',
        content: '你好！我是医学知识助手。请输入你的问题，我会基于知识库为你提供专业的回答。',
        cot: '',
        cotExpanded: false
      }]
    })
  },

  onInput(e) {
    this.setData({ inputText: e.detail.value })
  },

  async onSend() {
    const question = this.data.inputText.trim()
    if (!question || this.data.loading) return

    // 添加用户消息
    const messages = [...this.data.messages, {
      role: 'user',
      content: question
    }]

    this.setData({
      messages,
      inputText: '',
      loading: true
    })

    try {
      const res = await api.chat(question)
      const assistantMsg = {
        role: 'assistant',
        content: res.answer,
        cot: res.cot,
        cotExpanded: false
      }

      this.setData({
        messages: [...this.data.messages, assistantMsg],
        loading: false
      })
    } catch (err) {
      this.setData({
        messages: [...this.data.messages, {
          role: 'assistant',
          content: '抱歉，请求出错了，请稍后重试。',
          cot: '',
          cotExpanded: false
        }],
        loading: false
      })
    }
  },

  // 展开/折叠 CoT
  toggleCot(e) {
    const index = e.currentTarget.dataset.index
    const key = `messages[${index}].cotExpanded`
    this.setData({ [key]: !this.data.messages[index].cotExpanded })
  }
})
