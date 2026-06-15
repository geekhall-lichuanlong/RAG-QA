const api = require('../../utils/api')

Page({
  data: {
    currentIndex: 0,
    total: 0,
    progress: 0,

    questionId: '',
    questionText: '',
    options: {},
    optionKeys: [],
    correctAnswer: '',

    selectedOption: '',
    submitted: false,
    isCorrect: false,

    explanation: '',
    explainLoading: false,
    questionLoading: true
  },

  onLoad() {
    this.loadQuestion(0)
  },

  async loadQuestion(index) {
    this.setData({
      questionLoading: true,
      submitted: false,
      selectedOption: '',
      explanation: '',
      explainLoading: false
    })

    try {
      if (index === 0 && this.data.total === 0) {
        const totalRes = await api.getQuizTotal()
        this.setData({ total: totalRes.total })
      }

      const q = await api.getQuestion(index)
      this.setData({
        currentIndex: index,
        progress: Math.round(((index + 1) / this.data.total) * 100),
        questionId: q.id,
        questionText: q.question,
        options: q.options,
        optionKeys: Object.keys(q.options),
        total: q.total,
        questionLoading: false
      })
    } catch (err) {
      wx.showToast({ title: '加载题目失败', icon: 'none' })
      this.setData({ questionLoading: false })
    }
  },

  onOptionChange(e) {
    if (this.data.submitted) return
    this.setData({ selectedOption: e.detail.value })
  },

  async onSubmit() {
    if (!this.data.selectedOption) return

    wx.showLoading({ title: '正在提交并生成解析...' })

    try {
      const res = await api.submitAnswer(
        this.data.questionId,
        this.data.questionText,
        this.data.options,
        this.data.selectedOption
      )

      this.setData({
        submitted: true,
        correctAnswer: res.correct_answer,
        isCorrect: res.is_correct,
        // 把解析也存下来，避免 onExplain 重复请求
        explanation: res.explanation || ''
      })
    } catch (err) {
      wx.showToast({ title: '提交失败', icon: 'none' })
    } finally {
      wx.hideLoading()
    }
  },

  onExplain() {
    // 如果提交时已经拿到了解析，直接显示（无需再次请求后端）
    if (this.data.explanation) {
      return
    }

    // 否则重新请求（兜底逻辑）
    this.setData({ explainLoading: true })

    api.submitAnswer(
      this.data.questionId,
      this.data.questionText,
      this.data.options,
      this.data.selectedOption
    ).then(res => {
      this.setData({
        explanation: res.explanation || '暂无解析',
        explainLoading: false
      })
    }).catch(() => {
      this.setData({
        explanation: '解析生成失败，请稍后重试',
        explainLoading: false
      })
    })
  },

  onPrev() {
    if (this.data.currentIndex > 0) {
      this.loadQuestion(this.data.currentIndex - 1)
    }
  },

  onNext() {
    if (this.data.currentIndex < this.data.total - 1) {
      this.loadQuestion(this.data.currentIndex + 1)
    }
  }
})
