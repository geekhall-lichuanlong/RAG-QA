App({
  globalData: {
    // 后端 API 地址（ngrok 内网穿透，公网可访问）
    // 隧道关了再开时 URL 会变，需要重新修改这里
    apiBaseUrl: 'https://motive-chain-creamer.ngrok-free.dev'
  },

  onLaunch() {
    console.log('[App] 医学知识助手启动')
    // 可在此处检查登录态
  }
})
