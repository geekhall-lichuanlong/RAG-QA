import streamlit as st
from enhanced_rag import EnhancedRAG

st.set_page_config(page_title="智能客服 - 增强RAG", page_icon="🧠")
st.title("智能客服")
st.divider()

if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]

if "rag_engine" not in st.session_state:
    with st.spinner("正在加载知识库引擎..."):
        st.session_state["rag_engine"] = EnhancedRAG()

for message in st.session_state["message"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("请输入你的问题...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("正在检索并生成回答..."):
            try:
                result = st.session_state["rag_engine"].answer_question(prompt)
                answer = result.get("answer", "未获取到答案")
                cot = result.get("cot", "未生成推理过程")
                # 可选显示来源
                # sources = result.get("sources", [])
                with st.expander("🔍 推理过程（点击展开）"):
                    st.markdown(cot)
                st.markdown(f"**最终答案：** {answer}")
                full_response = f"推理过程：\n{cot}\n\n最终答案：{answer}"
                st.session_state["message"].append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"错误：{str(e)}")
                st.session_state["message"].append({"role": "assistant", "content": f"错误：{str(e)}"})