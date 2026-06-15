import streamlit as st
import json
from enhanced_rag import EnhancedRAG

# 页面配置
st.set_page_config(page_title="医学在线答题库", page_icon="🩺", layout="centered")

# 自定义CSS样式
st.markdown("""
<style>
    /* 全局背景 - 医学蓝渐变 */
    .stApp {
        background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 50%, #f7fafc 100%);
    }
    /* 英雄区 Hero Section */
    .hero-section {
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 25px;
    }
    .hero-icon {
        font-size: 60px;
        background: linear-gradient(135deg, #2563eb, #0ea5e9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-title {
        margin: 0;
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(135deg, #1e293b, #2d3748);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        margin: 0;
        color: #64748b;
        font-size: 16px;
    }
    /* 题目卡片 */
    .question-card {
        background: white;
        border-radius: 24px;
        padding: 2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    /* 自定义按钮 */
    .stButton > button {
        height: 52px !important;
        background: linear-gradient(135deg, #2563eb, #0ea5e9) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:active {
        transform: scale(0.97);
    }
    /* 选项卡片样式 */
    div[role="radiogroup"] label {
        background: white;
        border-radius: 16px;
        padding: 14px;
        margin: 8px 0;
        border: 1px solid #e2e8f0;
        transition: 0.25s;
    }
    div[role="radiogroup"] label:hover {
        border-color: #3b82f6;
        background: #eff6ff;
    }
    /* 成功/错误提示框圆角 */
    .stAlert {
        border-radius: 16px !important;
        border-left: 5px solid !important;
    }
    /* 侧边栏美化 */
    .css-1d391kg {
        background: rgba(255,255,255,0.8) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid #e2e8f0;
    }
    /* 中部导航文本 */
    .nav-text {
        text-align: center;
        font-size: 18px;
        color: #64748b;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# 标题区重构
st.markdown("""
<div class="hero-section">
    <div class="hero-icon">🩺</div>
    <div>
        <h1 class="hero-title">医学在线答题库</h1>
        <p class="hero-subtitle">
            AI驱动医学知识训练平台 · 智能解析 · 强化学习
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# 加载题库
@st.cache_data
def load_questions():
    try:
        with open(r"C:\Users\SUS\KnowledgeBase-RAG-LLM-System\MIRAGE\mmlu.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        questions = []
        for dataset_name, samples in data.items():
            for qid, sample in samples.items():
                questions.append({
                    "id": qid,
                    "question": sample["question"],
                    "options": sample["options"],
                    "answer": sample["answer"]
                })
        return questions
    except FileNotFoundError:
        st.error("题库文件未找到，请检查路径")
        return []

questions = load_questions()
total = len(questions)

# 初始化 session state
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "selected_option" not in st.session_state:
    st.session_state.selected_option = None
if "answer_submitted" not in st.session_state:
    st.session_state.answer_submitted = False
if "show_explanation" not in st.session_state:
    st.session_state.show_explanation = False
if "explanations" not in st.session_state:
    st.session_state.explanations = {}
if "rag_engine" not in st.session_state:
    with st.spinner("正在加载知识库引擎..."):
        st.session_state.rag_engine = EnhancedRAG()

# 当前题目
q = questions[st.session_state.current_idx]
question_text = q["question"]
options = q["options"]
correct_answer = q["answer"]

# 进度显示（紧凑型卡片 + 内嵌进度条，无白框）
progress = (st.session_state.current_idx + 1) / total
st.markdown(f"""
<div style="background: white; border-radius: 16px; padding: 0.75rem 1.2rem; margin-bottom: 1rem; border: 1px solid #e2e8f0;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
        <span>📊 学习进度</span>
        <span>{int(progress*100)}%</span>
    </div>
    <div style="background: #e2e8f0; border-radius: 20px; height: 8px; overflow: hidden;">
        <div style="width: {progress*100}%; background: linear-gradient(90deg, #2563eb, #0ea5e9); height: 100%; border-radius: 20px;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# 题目卡片
st.markdown(
    f"""
    <div class="question-card">
        <div class="question-number">
            📖 题目 {st.session_state.current_idx+1} / {total}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader(question_text)

# 选项
choice = st.radio(
    "选择答案",
    list(options.keys()),
    format_func=lambda x: f"{x}. {options[x]}",
    key="quiz_radio",
    disabled=st.session_state.answer_submitted,
    label_visibility="collapsed"
)

# 提交按钮
if not st.session_state.answer_submitted:
    if st.button("🚀 提交答案", use_container_width=True):
        st.session_state.selected_option = choice
        st.session_state.answer_submitted = True
        st.session_state.show_explanation = False
        st.rerun()

# 结果反馈
if st.session_state.answer_submitted:
    is_correct = (st.session_state.selected_option == correct_answer)
    if is_correct:
        st.success(f"🎉 回答正确！\n\n答案：{correct_answer}. {options[correct_answer]}")
        st.balloons()   # 庆祝动画
    else:
        st.error(f"❌ 回答错误\n\n正确答案：{correct_answer}. {options[correct_answer]}")
    
    # 解析按钮
    if st.button("🧠 AI智能解析", use_container_width=True):
        st.session_state.show_explanation = True
        if q["id"] not in st.session_state.explanations:
            with st.spinner("正在生成智能解析，请稍候..."):
                full_prompt = f"{question_text}\n选项：\n" + "\n".join([f"{k}. {v}" for k,v in options.items()])
                result = st.session_state.rag_engine.answer_question(full_prompt)
                explanation = result.get("cot", "无法生成解析")
                st.session_state.explanations[q["id"]] = explanation
        st.rerun()
    
    # 显示解析（知识卡片形式）
    if st.session_state.show_explanation:
        explanation = st.session_state.explanations.get(q["id"], "暂无解析")
        with st.expander("🧠 AI智能解析", expanded=True):
            st.markdown(explanation)

# 底部导航
st.divider()
col_prev, col_mid, col_next = st.columns([1, 3, 1])
with col_prev:
    if st.button("◀ 上一题", use_container_width=True) and st.session_state.current_idx > 0:
        st.session_state.current_idx -= 1
        st.session_state.answer_submitted = False
        st.session_state.show_explanation = False
        st.session_state.selected_option = None
        st.rerun()
with col_mid:
    st.markdown(f'<div class="nav-text">第 {st.session_state.current_idx+1} 题 / 共 {total} 题</div>', unsafe_allow_html=True)
with col_next:
    if st.button("下一题 ▶", use_container_width=True) and st.session_state.current_idx < total - 1:
        st.session_state.current_idx += 1
        st.session_state.answer_submitted = False
        st.session_state.show_explanation = False
        st.session_state.selected_option = None
        st.rerun()

# 侧边栏升级
st.sidebar.markdown("## 📚 学习中心")
st.sidebar.metric("完成进度", f"{int(progress*100)}%")
st.sidebar.progress(progress)
st.sidebar.metric("已解析题目", len([k for k in st.session_state.explanations if st.session_state.explanations[k]]))
st.sidebar.markdown("---")
st.sidebar.info(
"""
💡 **学习建议**

• 先独立思考再查看解析

• 关注错误题目对应知识点

• 利用AI解析建立知识体系
"""
)