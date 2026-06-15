# -*- coding: utf-8 -*-
"""
微信小程序后端 API 服务
基于 FastAPI，封装现有 EnhancedRAG 引擎
"""
import sys
import os
import io
import json
import hashlib
import traceback
from pathlib import Path

# 加载 .env 环境变量（必须在其他导入之前）
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# 强制标准输出使用 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 确保当前目录在 sys.path 中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

# ---------- 请求/响应模型 ----------

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    cot: str
    sources: Optional[list] = None

class QuizQuestion(BaseModel):
    id: str
    question: str
    options: dict
    total: int
    current_index: int

class QuizSubmitRequest(BaseModel):
    question_id: str
    question: str
    options: dict
    user_choice: str

class QuizSubmitResponse(BaseModel):
    correct_answer: str
    is_correct: bool
    explanation: Optional[str] = None

# ---------- 初始化 App ----------

app = FastAPI(
    title="Medical RAG API",
    description="医学知识库 RAG 系统 - 微信小程序后端",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 延迟加载 RAG 引擎 ----------

_rag_engine = None
_quiz_data = None
_kb_service = None

def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        from enhanced_rag import EnhancedRAG
        _rag_engine = EnhancedRAG()
    return _rag_engine

def get_kb_service():
    global _kb_service
    if _kb_service is None:
        from knowledge_base import KnowledgeBaseService
        _kb_service = KnowledgeBaseService()
    return _kb_service

def load_quiz_data():
    global _quiz_data
    if _quiz_data is None:
        quiz_path = os.getenv(
            "QUIZ_DATA_PATH",
            os.path.join(os.path.dirname(__file__), "MIRAGE", "mmlu.json")
        )
        if not os.path.exists(quiz_path):
            raise FileNotFoundError(f"题库文件未找到: {quiz_path}")
        with open(quiz_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        questions = []
        for dataset_name, samples in raw.items():
            for qid, sample in samples.items():
                questions.append({
                    "id": qid,
                    "question": sample["question"],
                    "options": sample["options"],
                    "answer": sample["answer"]
                })
        _quiz_data = questions
    return _quiz_data

# ---------- API 路由 ----------

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Medical RAG API"}

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """RAG 对话接口：返回答案 + 推理过程"""
    try:
        engine = get_rag_engine()
        result = engine.answer_question(req.question)
        return ChatResponse(
            answer=result.get("answer", "未获取到答案"),
            cot=result.get("cot", ""),
            sources=result.get("sources", [])[:3]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/quiz/question")
def get_question(index: int = 0):
    """获取指定索引的题目"""
    try:
        data = load_quiz_data()
        if index < 0 or index >= len(data):
            raise HTTPException(status_code=400, detail="题目索引超出范围")
        q = data[index]
        return QuizQuestion(
            id=q["id"],
            question=q["question"],
            options=q["options"],
            total=len(data),
            current_index=index
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/quiz/explain", response_model=QuizSubmitResponse)
def quiz_explain(req: QuizSubmitRequest):
    """提交答案并获取 AI 解析"""
    # 加载题库数据验证答案
    data = load_quiz_data()
    correct = None
    for q in data:
        if q["id"] == req.question_id:
            correct = q["answer"]
            break
    if correct is None:
        raise HTTPException(status_code=404, detail="题目未找到")

    is_correct = (req.user_choice == correct)

    # 生成 AI 解析
    try:
        engine = get_rag_engine()
        full_prompt = (
            f"{req.question}\n"
            f"选项：\n" +
            "\n".join([f"{k}. {v}" for k, v in req.options.items()])
        )
        result = engine.answer_question(full_prompt, explain_only=True)
        explanation = result.get("cot", "")
    except Exception as e:
        traceback.print_exc()
        explanation = f"解析生成失败: {str(e)}"

    return QuizSubmitResponse(
        correct_answer=correct,
        is_correct=is_correct,
        explanation=explanation
    )

@app.get("/api/quiz/total")
def quiz_total():
    """返回题库总数"""
    data = load_quiz_data()
    return {"total": len(data)}


# ---------- 知识上传 ----------

class TextUploadRequest(BaseModel):
    text: str
    filename: Optional[str] = None
    vectorize: bool = True

@app.post("/api/upload/text")
def upload_text(req: TextUploadRequest):
    """上传文本内容到知识库"""
    try:
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="文本内容不能为空")

        # 生成文件名
        filename = req.filename or f"upload_{hashlib.md5(text.encode()).hexdigest()[:8]}.txt"
        # 保存到 corpus 目录
        upload_dir = os.path.join(os.path.dirname(__file__), "corpus", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

        # 可选：向量化到知识库
        kb_result = None
        if req.vectorize:
            kb = get_kb_service()
            kb_result = kb.upload_by_str(text, filename)

        return {
            "status": "ok",
            "filename": filename,
            "path": filepath,
            "size": len(text),
            "vectorize": kb_result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload/file")
async def upload_file(file: UploadFile):
    """上传文件到知识库，支持 TXT / PDF / JSON / JSONL / Markdown"""
    try:
        content = await file.read()
        filename = file.filename or "unknown.txt"

        upload_dir = os.path.join(os.path.dirname(__file__), "corpus", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)

        # 1. 保存原始文件（二进制保留格式）
        with open(filepath, "wb") as f:
            f.write(content)

        # 2. 解析内容并向量化
        from knowledge_base import parse_file

        parsed_text = parse_file(content, filename)
        kb = get_kb_service()
        kb_result = kb.upload_by_str(parsed_text, filename)

        return {
            "status": "ok",
            "filename": filename,
            "path": filepath,
            "size": len(parsed_text),
            "format": os.path.splitext(filename)[1].lower(),
            "vectorize": kb_result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- 启动入口 ----------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
