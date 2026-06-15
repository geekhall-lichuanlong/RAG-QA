# -*- coding: utf-8 -*-
r"""
一键启动 ngrok 内网穿透
1. 先去 https://dashboard.ngrok.com/signup 注册
2. 复制你的 authtoken
3. 替换下面 AUTH_TOKEN
4. 运行: python start_tunnel.py
"""
from pyngrok import ngrok, conf

# ====== 把你的 authtoken 贴在这里 ======
AUTH_TOKEN = "3EZ1egqDLh5FXmsGBDHPHqrWL82_4oSsuDypHrf4BinvDi1F9"
# =====================================

conf.get_default().auth_token = AUTH_TOKEN

print("正在启动隧道...")
tunnel = ngrok.connect(8000, "http")
print(f"\n✅ 公网地址: {tunnel.public_url}")
print(f"   健康检查: {tunnel.public_url}/api/health")
print("\n把上面地址更新到 miniprogram/app.js 的 apiBaseUrl")
print("按 Ctrl+C 停止\n")

try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n隧道已关闭")
    ngrok.kill()
