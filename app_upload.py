"""
知识库上传服务 - 支持 TXT / PDF / JSON / JSONL / Markdown
"""

import streamlit as st
import time
import sys
import os

from knowledge_base import KnowledgeBaseService, parse_file, SUPPORTED_EXTENSIONS, SUPPORTED_DESCRIPTION

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="知识库更新服务",
    page_icon="📚",
    layout="wide",
)

st.title("📚 知识库更新服务")
st.markdown(
    f"上传 **TXT · PDF · JSON · JSONL · Markdown** 文件，自动解析并载入向量知识库。"
)

st.divider()

# ---------- 初始化 ----------
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

# ---------- 文件上传区 ----------
uploaded_files = st.file_uploader(
    "选择文件（支持多选）",
    type=["txt", "pdf", "json", "jsonl", "md"],
    accept_multiple_files=True,
)

# 上传结果缓存
if "upload_results" not in st.session_state:
    st.session_state["upload_results"] = []

# ---------- 处理上传 ----------
col1, col2 = st.columns([1, 5])
with col1:
    upload_clicked = st.button("🚀 上传到知识库", type="primary", use_container_width=True)
with col2:
    if st.session_state["upload_results"]:
        st.button("🧹 清除结果", key="clear_results")

if upload_clicked and uploaded_files:
    st.session_state["upload_results"] = []
    progress_bar = st.progress(0, text="准备上传...")
    total = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        file_name = uploaded_file.name
        file_ext = os.path.splitext(file_name)[1].lower()
        file_size = uploaded_file.size / 1024  # KB
        content_bytes = uploaded_file.getvalue()

        status_icon = "⏳"
        status_msg = "处理中..."
        detail = ""

        try:
            # 校验扩展名
            if file_ext not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"不支持的文件格式「{file_ext}」")

            # 调用知识库服务上传
            result = st.session_state["service"].upload_by_file(content_bytes, file_name)
            status_icon = "✅"
            status_msg = result

            # 显示解析后文本长度信息
            if "Success" in result:
                try:
                    text = parse_file(content_bytes, file_name)
                    detail = f"（解析文本长度：{len(text)} 字符）"
                except Exception:
                    pass

        except ImportError as e:
            # 缺少依赖库（如 pypdf）
            status_icon = "⚠️"
            status_msg = str(e)
        except Exception as e:
            status_icon = "❌"
            status_msg = f"上传失败: {str(e)}"

        st.session_state["upload_results"].append({
            "icon": status_icon,
            "name": file_name,
            "ext": file_ext,
            "size": file_size,
            "msg": status_msg,
            "detail": detail,
        })

        progress_bar.progress((idx + 1) / total, text=f"处理 {idx + 1}/{total}：{file_name}")

    progress_bar.empty()

# ---------- 显示上传结果 ----------
if st.session_state["upload_results"]:
    st.divider()
    st.subheader("📋 上传结果")

    for r in st.session_state["upload_results"]:
        label = r["ext"].upper().lstrip(".") if r["ext"] else "?"
        size_str = f"{r['size']:.1f} KB" if r['size'] < 1024 else f"{r['size'] / 1024:.2f} MB"

        st.markdown(
            f"{r['icon']} **{r['name']}**  （{label} · {size_str}）  \n"
            f"　　{r['msg']} {r['detail']}"
        )

    # 清除按钮
    if st.button("清除本次上传结果"):
        st.session_state["upload_results"] = []
        st.rerun()

# ---------- 数据处理工具 ----------
st.divider()
with st.expander("📁 数据处理工具"):
    st.write("上传完成后，点击下方按钮对数据进行格式化处理。")
    if st.button("开始处理数据", type="primary"):
        with st.spinner("正在处理数据，请稍候..."):
            import subprocess
            result = subprocess.run(
                [sys.executable, "12.0数据处理.py"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                st.success("数据处理完成！")
                if result.stdout.strip():
                    st.code(result.stdout[:2000])
            else:
                st.error(f"数据处理失败！错误信息：\n{result.stderr[:2000]}")
