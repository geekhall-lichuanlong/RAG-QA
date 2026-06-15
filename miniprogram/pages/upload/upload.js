const app = getApp()

Page({
  data: {
    activeTab: 'file',

    // 文件上传
    selectedFile: null,
    uploading: false,

    // 文本录入
    textTitle: '',
    textContent: '',

    // 结果
    uploadResult: null
  },

  switchTab(e) {
    const tab = e.currentTarget.dataset.tab
    this.setData({ activeTab: tab, uploadResult: null })
  },

  // ---- 文件上传 ----

  onChooseFile() {
    wx.chooseMessageFile({
      count: 1,
      type: 'file',
      extension: ['txt'],
      success: (res) => {
        const file = res.tempFiles[0]
        this.setData({
          selectedFile: {
            name: file.name,
            size: (file.size / 1024).toFixed(1),
            path: file.path
          },
          uploadResult: null
        })
      },
      fail: (err) => {
        if (err.errMsg.indexOf('cancel') === -1) {
          wx.showToast({ title: '选择文件失败', icon: 'none' })
        }
      }
    })
  },

  onUploadFile() {
    if (!this.data.selectedFile) return

    this.setData({ uploading: true })

    wx.uploadFile({
      url: app.globalData.apiBaseUrl + '/api/upload/file',
      filePath: this.data.selectedFile.path,
      name: 'file',
      success: (res) => {
        try {
          const data = JSON.parse(res.data)
          this.setData({ uploadResult: data, selectedFile: null })
        } catch (e) {
          this.setData({
            uploadResult: { status: 'fail', error: '响应解析失败' }
          })
        }
      },
      fail: () => {
        this.setData({
          uploadResult: { status: 'fail', error: '上传失败，请检查后端服务' }
        })
      },
      complete: () => {
        this.setData({ uploading: false })
      }
    })
  },

  // ---- 文本录入 ----

  onTextTitle(e) {
    this.setData({ textTitle: e.detail.value })
  },

  onTextContent(e) {
    this.setData({ textContent: e.detail.value })
  },

  async onUploadText() {
    const content = this.data.textContent.trim()
    if (!content) return

    this.setData({ uploading: true })

    try {
      const res = await new Promise((resolve, reject) => {
        wx.request({
          url: app.globalData.apiBaseUrl + '/api/upload/text',
          method: 'POST',
          data: {
            text: content,
            filename: this.data.textTitle || null
          },
          header: { 'Content-Type': 'application/json' },
          success: (r) => resolve(r.data),
          fail: reject
        })
      })

      this.setData({
        uploadResult: res,
        textContent: '',
        textTitle: ''
      })
    } catch (err) {
      this.setData({
        uploadResult: { status: 'fail', error: '提交失败，请检查后端服务' }
      })
    } finally {
      this.setData({ uploading: false })
    }
  }
})
