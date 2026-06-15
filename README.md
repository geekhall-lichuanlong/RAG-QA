# 🩺 KnowledgeBase-RAG-LLM-System

<div align="center">

**医学知识库 RAG 检索增强生成系统** · 智能问答 · 题库解析 · 知识管理

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![WeChat](https://img.shields.io/badge/微信小程序-✅-07C160.svg)](./miniprogram/)

</div>

---

## 📖 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [运行方式](#运行方式)
- [API 接口文档](#api-接口文档)
- [技术栈详解](#技术栈详解)
- [数据流程](#数据流程)
- [常见问题](#常见问题)
- [后续规划](#后续规划)
- [致谢](#致谢)

---

## 项目概述

本项目是一个面向**医学领域**的 **RAG（检索增强生成）** 智能问答系统，核心能力包括：

- 🔍 **双通道向量检索**：FAISS + MedCPT 嵌入 & ChromaDB + DashScope 嵌入，互补检索
- 🧠 **Chain-of-Thought 推理**：基于 DeepSeek API 的多步推理链，先推理后作答
- 📚 **医学题库智能解析**：支持 MMLU / MedQA / MedMCQA 题库的 AI 批改与解析
- 🔄 **多样化查询扩展**：自动生成候选答案、相关内容、标题等多视角查询，提升召回率
- 📱 **双端覆盖**：Streamlit Web 端（PC）+ 微信小程序（移动端），一套引擎两种体验

### 适用场景

| 场景 | 说明 |
|------|------|
| 医学知识问答 | 输入医学问题，获得带推理过程的专业回答 |
| 题库练习与解析 | 选择题自测，AI 给出推理过程与正确答案 |
| 知识库管理 | 上传医学文献/教材，构建可检索的向量知识库 |
| 研究与评估 | MMLU 基准测试，模型推理能力评估 |

---

## 核心特性

### 1. 增强 RAG 引擎 (`enhanced_rag.py`)

```
用户问题
   │
   ├─→ 生成候选答案 ──→ FAISS 检索 ──┐
   ├─→ 生成候选内容 ──→ FAISS 检索 ──┤
   ├─→ 生成候选标题 ──→ FAISS 检索 ──┤  三路并行检索
   │                                  │
   │              ┌───────────────────┘
   │              ▼
   │         事实提取与问题精炼
   │              │
   │              ▼
   │         CoT 推理链生成
   │              │
   │              ▼
   │         最终答案 / 选项字母
   ▼
 返回 { answer, cot, sources }
```

- **多路并行检索**：三种查询扩展策略同时检索，提升召回覆盖率
- **迭代精炼**：支持多轮检索-提取-精炼循环（可配置迭代次数），提高检索文档相关性
- **选择题智能识别**：自动识别 A/B/C/D 选项格式，输出规范化答案
- **Fallback 机制**：当检索结果不相关时，自动回退到模型自身知识

### 2. 医学题库系统

- 支持 **MMLU**（大规模多任务语言理解）医学子集
- 支持 **MedQA** 和 **MedMCQA** 等多种题库
- AI 智能解析每一道题，给出推理过程
- 题库中文翻译工具（`translate_quiz.py`），批量翻译英文题目
- 内置 MMLU 评估脚本（`12.3evaluate.py`），计算答题准确率

### 3. 知识库管理

- 支持文本文件和原始文本上传
- 文本自动分块（RecursiveCharacterTextSplitter）
- 向量化存入 ChromaDB，支持持久化
- MD5 去重，避免重复入库

### 4. 双模式前端

| 特性 | Streamlit 原版 | 微信小程序版 |
|------|---------------|-------------|
| 访问方式 | PC 浏览器 | 手机微信 |
| 问答聊天 | ✅ | ✅ |
| 题库答题 | ✅ | ✅ |
| 知识上传 | ✅ | ✅ |
| UI 风格 | 简洁专业 | 原生移动端 |
| 部署 | 本地运行 | 需要后端 + 公网 |

---

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                      前端展示层                            │
│  ┌─────────────────────┐   ┌─────────────────────────┐   │
│  │  Streamlit Web UI   │   │   微信小程序 (WXML/JS)    │   │
│  │  app_chat.py        │   │   miniprogram/pages/     │   │
│  │  app_quiz.py        │   │   ├─ chat  (问答)        │   │
│  │  app_upload.py      │   │   ├─ quiz  (答题)        │   │
│  └────────┬────────────┘   │   └─ upload (上传)       │   │
│           │                └────────────┬────────────┘   │
│           │ 直接函数调用       HTTPS REST API              │
│           │                ┌────────────┴────────────┐   │
│           │                │   FastAPI 后端           │   │
│           │                │   backend_api.py        │   │
│           │                └────────────┬────────────┘   │
│           ▼                      ▼                      │
├──────────────────────────────────────────────────────────┤
│                      RAG 引擎层                            │
│  ┌──────────────────────────────────────────────────┐    │
│  │              EnhancedRAG (enhanced_rag.py)        │    │
│  │  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │ RetrievalSystem │  │ AnswerGenerator │           │    │
│  │  │ (FAISS+MedCPT) │  │ (DeepSeek API) │           │    │
│  │  └───────┬──────┘  └───────┬──────┘              │    │
│  │          │                 │                      │    │
│  │  ┌───────┴─────────────────┴──────┐               │    │
│  │  │  query_to_cot → cot_to_answer │               │    │
│  │  │  (CoT 推理链)                  │               │    │
│  │  └────────────────────────────────┘               │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │         RagService (rag.py) - ChromaDB 通道       │    │
│  │  LangChain + ChromaDB + DashScope Embedding      │    │
│  └──────────────────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────┤
│                     向量存储层                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  FAISS 索引   │  │  ChromaDB    │  │ 本地模型      │   │
│  │  corpus/     │  │  chroma_db/  │  │ models/      │   │
│  │  (MedCPT)    │  │  (DashScope) │  │ (Encoder)    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### 数据流概览

```text
用户提问 → 查询扩展（3路）→ FAISS/ChromaDB 检索
       → 事实提取 → 问题精炼 → CoT 推理 → 答案生成 → 返回结果
```

---

## 项目结构

```text
KnowledgeBase-RAG-LLM-System/
│
├── 📄 backend_api.py              # FastAPI 后端入口（小程序 API 服务）
├── 🧠 enhanced_rag.py              # 增强 RAG 引擎（核心：检索 + CoT 推理 + 答案生成）
├── 📊 多样化及事实知识提取deepseek.py # 医学知识检索系统（FAISS + MedCPT 嵌入）
├── 📝 template.py                  # Prompt 模板库（Jinja2/Liquid 模板引擎）
├── 🔗 query_to_cot.py              # Query → Chain-of-Thought 推理链
├── ✅ cot_to_answer.py             # CoT → 最终答案
├── 📦 rag.py                       # LangChain RAG 链（ChromaDB + 通义千问 通道）
├── 🗄️ vector_stores.py            # ChromaDB 向量库封装
├── 💾 file_history_store.py        # 对话历史文件存储
├── file_history_store.py          # 对话历史管理（RunnableWithMessageHistory）
├── ⚙️ config_data.py               # 全局配置（模型、路径、chunk 参数等）
│
├── 🌐 Streamlit 前端（原版 PC 端）──
├── app_chat.py                     # 智能客服问答页面
├── app_quiz.py                     # 医学题库答题页面
├── app_upload.py                   # 知识库上传 & 数据处理页面
│
├── 📱 微信小程序前端 ──
├── miniprogram/
│   ├── app.js / app.json / app.wxss     # 全局配置、路由、样式
│   ├── utils/api.js                     # 后端 API 请求封装层
│   └── pages/
│       ├── chat/                # 智能问答页
│       ├── quiz/                # 答题练习页
│       └── upload/              # 知识上传页
│
├── 🗂️ 数据与模型 ──
├── corpus/                      # 知识语料库 & FAISS 索引
│   └── textbooks/               # 教材文本（分块 + 索引文件）
├── chroma_db/                   # ChromaDB 持久化向量数据库
├── models/                      # 本地嵌入模型
│   ├── MedCPT-Query-Encoder/    # 查询编码器
│   └── MedCPT-Article-Encoder/  # 文章编码器
├── MIRAGE/                      # 医学题库数据
│   ├── mmlu.json                # MMLU 医学题库
│   ├── medqa.json               # MedQA 题库
│   └── medmcqa.json             # MedMCQA 题库
│
├── 🛠️ 工具脚本 ──
├── translate_quiz.py            # 题库中英文翻译工具（批量翻译，断点续传）
├── start_tunnel.py              # ngrok 内网穿透一键启动（小程序调试用）
├── 12.0数据处理.py              # 数据预处理脚本
├── 12.3evaluate.py              # MMLU 基准评估脚本
│
├── 📋 环境配置 ──
├── requirements.txt             # Python 依赖清单
├── .env.example                 # 环境变量模板
└── LICENSE                      # 开源协议
```

---

## 快速开始

### 环境要求

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Python | 3.9+ | 推荐使用 Conda 环境 |
| Conda | Miniconda3 / Anaconda | 管理 ML 依赖（torch, faiss 等） |
| 微信开发者工具 | 最新版 | 仅小程序模式需要 |
| DeepSeek API Key | - | LLM 推理与答案生成 |
| DashScope API Key | - | 通义千问嵌入模型（ChromaDB 通道） |

### 1. 克隆项目 & 安装依赖

```bash
# 克隆项目
git clone <repo-url>
cd KnowledgeBase-RAG-LLM-System

# 安装 Python 依赖
pip install -r requirements.txt

# 如果有 Conda 环境（推荐），需要额外安装 ML 依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install faiss-cpu sentence-transformers
```

### 2. 配置 API Key

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key
```

`.env` 文件内容：

```env
# DeepSeek API Key（必填 - LLM 推理引擎）
DEEPSEEK_API_KEY=sk-your-deepseek-key-here

# 阿里云 DashScope API Key（ChromaDB 嵌入模型）
DASHSCOPE_API_KEY=sk-your-dashscope-key-here

# 题库文件路径（可选）
QUIZ_DATA_PATH=./MIRAGE/mmlu.json

# 向量索引目录（可选）
INDEX_DIR=./corpus

# ChromaDB 持久化目录（可选）
CHROMA_PERSIST_DIR=./chroma_db
```

### 3. 准备知识库数据

将医学教材或文献的 `.txt` 文件放入 `corpus/` 目录，然后运行数据处理脚本构建向量索引：

```bash
python 12.0数据处理.py
```

---

## 运行方式

### 方式一：Streamlit 原版（PC 浏览器）🖥️

最简单的本地运行方式，适合开发调试和个人使用：

```bash
# 智能客服问答
streamlit run app_chat.py
# → 浏览器打开 http://localhost:8501

# 医学题库答题
streamlit run app_quiz.py
# → 浏览器打开 http://localhost:8501

# 知识库上传与管理
streamlit run app_upload.py
# → 浏览器打开 http://localhost:8501
```

三个页面可以同时运行在不同的端口上。

### 方式二：微信小程序版（移动端）📱

适合在手机上使用，需要同时启动后端服务和微信开发者工具。

#### 步骤 1：启动 FastAPI 后端

```bash
python backend_api.py
```

看到以下输出表示启动成功：
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

API 文档自动生成在：http://localhost:8000/docs （交互式 Swagger 文档）

#### 步骤 2：（可选）开启内网穿透

如果需要在手机上通过微信访问，需要公网地址：

```bash
# 使用 ngrok 一键穿透（需先配置 start_tunnel.py 中的 AUTH_TOKEN）
python start_tunnel.py
# → 输出公网地址，复制到 miniprogram/app.js 的 apiBaseUrl
```

#### 步骤 3：配置微信开发者工具

1. 下载安装 [微信开发者工具](https://developers.weixin.qq.com/miniprogram/dev/devtools/download.html)
2. 打开工具 → 点击 **"导入项目"**
3. 填写项目信息：
   - 项目名称：`医学知识助手`
   - 目录：选择 `./miniprogram`
   - AppID：使用**测试号**（或注册的小程序 AppID）
4. 导入后 → 右上角 **"详情"** → **"本地设置"** → 勾选 ✅ **"不校验合法域名"**
5. 修改 `miniprogram/app.js` 中的 `apiBaseUrl` 为你的后端地址
6. 点击 **"编译"** 运行

#### 步骤 4：测试接口

```bash
# 健康检查
curl http://localhost:8000/api/health
# → {"status":"ok","service":"Medical RAG API"}

# 对话测试
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"什么是肺炎？"}'
```

---

## API 接口文档

启动后端后，访问 http://localhost:8000/docs 查看完整的交互式 Swagger 文档。

### 接口一览

| 方法 | 路径 | 功能 | 认证 |
|------|------|------|------|
| `GET` | `/api/health` | 健康检查 | 无 |
| `POST` | `/api/chat` | RAG 对话问答 | 无 |
| `GET` | `/api/quiz/question` | 获取指定题目 | 无 |
| `POST` | `/api/quiz/explain` | 提交答案 + AI 解析 | 无 |
| `GET` | `/api/quiz/total` | 题库总数 | 无 |
| `POST` | `/api/upload/text` | 文本上传 | 无 |
| `POST` | `/api/upload/file` | 文件上传 | 无 |

### 接口详情

#### `POST /api/chat` — 智能问答

**请求：**
```json
{
  "question": "一名45岁男性出现胸痛、呼吸困难，最可能的诊断是什么？\nA. 心肌梗死\nB. 肺栓塞\nC. 气胸\nD. 主动脉夹层"
}
```

**响应：**
```json
{
  "answer": "B",
  "cot": "步骤一：分析症状——胸痛+呼吸困难...\n步骤二：逐一排除...\n最终选择B：肺栓塞。",
  "sources": [
    {
      "title": "内科学-第9版-肺栓塞章节",
      "content": "肺栓塞的典型表现为胸痛、呼吸困难..."
    }
  ]
}
```

#### `GET /api/quiz/question?index=0`

**响应：**
```json
{
  "id": "mmlu_001",
  "question": "下列哪项是社区获得性肺炎最常见的病原体？",
  "options": {
    "A": "肺炎链球菌",
    "B": "金黄色葡萄球菌",
    "C": "肺炎支原体",
    "D": "流感嗜血杆菌"
  },
  "total": 500,
  "current_index": 0
}
```

#### `POST /api/quiz/explain`

**请求：**
```json
{
  "question_id": "mmlu_001",
  "question": "下列哪项是社区获得性肺炎最常见的病原体？",
  "options": {
    "A": "肺炎链球菌",
    "B": "金黄色葡萄球菌",
    "C": "肺炎支原体",
    "D": "流感嗜血杆菌"
  },
  "user_choice": "A"
}
```

**响应：**
```json
{
  "correct_answer": "A",
  "is_correct": true,
  "explanation": "社区获得性肺炎最常见的病原体是肺炎链球菌..."
}
```

---

## 技术栈详解

### RAG 检索引擎

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 主检索器 | FAISS + MedCPT | 高效向量相似度搜索，医学领域专用嵌入 |
| 备选检索器 | ChromaDB + text-embedding-v4 | 阿里云 DashScope 嵌入，持久化存储 |
| 查询扩展 | DeepSeek-Chat 生成 | 自动生成候选答案/内容/标题三种查询视角 |
| 事实提取 | DeepSeek-Chat 提取 | 从检索结果中提取关键事实，精炼下一步查询 |

### 推理框架

```
用户问题
    │
    ▼
┌─────────────────────────────┐
│  第一阶段：Query → CoT      │  query_to_cot.py
│  输入：问题 + 检索到的资料    │  生成逐步推理过程
│  输出：Chain of Thought     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  第二阶段：CoT → Answer     │  cot_to_answer.py
│  输入：问题 + CoT 推理过程   │  选择题 → 选项字母
│  输出：最终答案              │  开放题 → 完整回答
└─────────────────────────────┘
```

### 嵌入模型

| 模型 | 用途 | 维度 |
|------|------|------|
| MedCPT-Query-Encoder | 查询编码（FAISS 通道） | 768 |
| MedCPT-Article-Encoder | 文章编码（FAISS 通道） | 768 |
| text-embedding-v4 | 文档嵌入（ChromaDB 通道，DashScope API） | 1024 |

### LLM 模型

| 模型 | 用途 | API |
|------|------|-----|
| DeepSeek-Chat | CoT 推理、答案生成、查询扩展、事实提取 | DeepSeek API |


### 开发框架

| 框架 | 用途 | 版本 |
|------|------|------|
| Streamlit | Web 前端 UI | ≥1.35 |
| FastAPI | 小程序 REST API | ≥0.110 |
| Uvicorn | ASGI 服务器 | ≥0.29 |
| LangChain | RAG 链编排 | ≥0.1.0 |
| ChromaDB | 向量数据库 | ≥0.4.24 |

---

## 数据流程

### 问答完整流程

```text
1. 用户输入问题
      │
2. 查询扩展（3 路并行）          ← DeepSeek API
   ├─ generate_possible_answer   "最可能的答案是..."
   ├─ generate_possible_content  "相关内容可能涉及..."
   └─ generate_possible_title    "可能出现在标题为...的章节"
      │
3. 向量检索（3 路并行）          ← FAISS / ChromaDB
   ├─ 问题 + 候选答案 → 检索
   ├─ 问题 + 候选内容 → 检索
   └─ 问题 + 候选标题 → 检索
      │
4. 合并检索结果 & 去重
      │
5. 事实提取                      ← DeepSeek API
   │  extract_factual_info_rag()
   │  从检索结果中提取关键事实
      │
6. 问题精炼（可选迭代）
   │  将提取的事实拼接到原始问题
      │
7. CoT 推理链生成                ← DeepSeek API
   │  query_to_cot()
   │  "请一步步推理..."
      │
8. 最终答案生成                  ← DeepSeek API
   │  cot_to_answer()
   │  选择题 → 选项字母 / 开放题 → 完整回答
      │
9. 返回结果
   → { answer, cot, sources }
```

### 知识入库流程

```text
文件上传 (.txt)
      │
      ▼
文本分块 (RecursiveCharacterTextSplitter)
  chunk_size=1000, chunk_overlap=100
      │
      ▼
MD5 去重检查
      │
      ▼
向量化编码 (MedCPT / text-embedding-v4)
      │
      ▼
向量存储 (FAISS 索引 / ChromaDB)
      │
      ▼
入库完成 → 可被检索
```

---

## 常见问题

### Q1：启动后端报 `ModuleNotFoundError: No module named 'xxx'`

**原因**：依赖未安装或 Python 环境不匹配。

**解决**：
```bash
# 确保安装了所有依赖
pip install -r requirements.txt

# 如果使用 Conda，需要额外安装 ML 依赖
pip install faiss-cpu sentence-transformers torch
```

### Q2：小程序提示"网络连接失败"

检查清单：
1. ✅ 后端是否已启动？访问 `http://localhost:8000/api/health` 确认
2. ✅ 微信开发者工具是否勾选了"不校验合法域名"？
3. ✅ `miniprogram/app.js` 中的 `apiBaseUrl` 是否正确？
4. ✅ 如果使用真机调试，内网穿透是否正常工作？

### Q3：答题页面"解析生成失败"

检查后端终端的报错信息，常见原因：
- DeepSeek API Key 失效或额度用尽
- FAISS 索引文件路径不正确
- 网络代理导致 API 调用超时

### Q4：上传文件后，问答没有检索到内容

- 上传的文件保存在 `corpus/uploads/` 目录
- 需要运行数据处理脚本（`12.0数据处理.py`）重建向量索引
- 确认 `config_data.py` 中的路径配置与你的目录结构一致

### Q5：Streamlit 原版还能用吗？

**完全可以。** Streamlit 前端和 FastAPI 后端完全独立，互不影响。两种方式可同时运行。

### Q6：API Key 硬编码问题

⚠️ **安全提醒**：部分文件中存在硬编码的 API Key（详见下文），部署前务必迁移到环境变量：

- `多样化及事实知识提取deepseek.py` 第 139 行、第 368 行 → 已支持 `os.getenv('DEEPSEEK_API_KEY', 'fallback')`
- `query_to_cot.py` 第 64 行 → 已支持 `os.getenv('DEEPSEEK_API_KEY', 'fallback')`
- `cot_to_answer.py` 第 63 行 → 已支持 `os.getenv('DEEPSEEK_API_KEY', 'fallback')`

> 配置 `.env` 文件后，环境变量将覆盖硬编码的默认值。

---

## 后续规划

- [ ] 支持 PDF / Word / Markdown 文档上传与解析（依赖已预装）
- [ ] 添加用户认证与权限管理
- [ ] 支持多轮对话上下文记忆
- [ ] 向量库增量更新（无需全量重建索引）
- [ ] Docker 一键部署
- [ ] 支持更多 LLM（OpenAI GPT、Claude 等）
- [ ] 题库答错自动收藏与复习功能

---

## 致谢

本项目得益于以下优秀的开源项目和技术：

- [Streamlit](https://streamlit.io/) — 让 Python 快速构建 Web UI
- [LangChain](https://www.langchain.com/) — LLM 应用开发框架
- [ChromaDB](https://www.trychroma.com/) — 开源向量数据库
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook 高效相似度搜索
- [FastAPI](https://fastapi.tiangolo.com/) — 现代 Python Web 框架
- [MedCPT](https://github.com/ncbi/MedCPT) — 医学领域预训练嵌入模型
- [DeepSeek](https://www.deepseek.com/) — LLM API 服务
- [阿里云 DashScope](https://dashscope.aliyun.com/) — 通义千问模型服务
- 微信小程序平台

---

## License

本项目基于 MIT 协议开源，详见 [LICENSE](./LICENSE) 文件。

项目仅用于学习与交流，如需商用请自行补全安全、合规与授权相关内容。

---

<div align="center">
  <sub>Built with ❤️ for medical AI research and education</sub>
</div>
