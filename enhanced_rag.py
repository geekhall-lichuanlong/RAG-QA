# -*- coding: utf-8 -*-
"""
增强 RAG 模块 - 直接调用已有脚本中的组件
"""
import sys
import os
import io
import re
from concurrent.futures import ThreadPoolExecutor

# 强制标准输出使用 UTF-8 编码，避免 ascii 编码错误
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 确保当前目录在 sys.path 中，以便导入其他脚本
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入你已有的类和函数
from 多样化及事实知识提取deepseek import (
    RetrievalSystem,
    AnswerGenerator,
    extract_factual_info_rag,
    split_sentences
)
from template import general_extract_nolist  # 如果需要

# 尝试导入 CoT 相关函数（如果已重命名文件）
try:
    from query_to_cot import call_deepseek_api as call_cot
    from cot_to_answer import call_deepseek_api as call_answer
    COT_AVAILABLE = True
except ImportError:
    COT_AVAILABLE = False
    print("[EnhancedRAG] 警告: 未找到 query_to_cot 或 cot_to_answer 模块，将使用简化版答案生成。")
    # 定义备用函数（直接调用 generator.client 生成答案）
    def call_cot(messages, client):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].message.content

    def call_answer(messages, client):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].message.content

class EnhancedRAG:
    def __init__(self, k=3):
        print("[EnhancedRAG] 初始化中...")
        self.retriever = RetrievalSystem()   # 会使用你脚本中的路径配置
        self.generator = AnswerGenerator()   # 使用你脚本中的 API Key
        self.k = k
        print("[EnhancedRAG] 初始化完成")

    def answer_question(self, question: str, iterations: int = 1, explain_only: bool = False):
        """
        完整处理用户问题，返回最终答案（字符串）
        iterations: 迭代次数（默认 1，减少 API 调用加快响应）
        explain_only: 仅需推理过程时跳过最终答案生成（用于答题解析）
        """
        original_question = question
        current_question = question
        all_retrieved_snippets = []
        _, _, last_sentence = split_sentences(question)

        for i in range(iterations):
            # 1. 生成候选答案、内容、标题（并行化：三者互不依赖）
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_answers = executor.submit(self.generator.generate_possible_answer, current_question)
                future_content = executor.submit(self.generator.generate_possible_content, current_question)
                future_title = executor.submit(self.generator.generate_possible_title, current_question)
                possible_answers = future_answers.result()
                possible_content = future_content.result()
                possible_title = future_title.result()

            # 2. 检索并合并
            snippets, _ = self.retriever.retrieve(current_question + possible_answers, k=self.k)
            all_retrieved_snippets.extend(snippets)
            snippets, _ = self.retriever.retrieve(current_question + possible_content, k=self.k)
            all_retrieved_snippets.extend(snippets)
            snippets, _ = self.retriever.retrieve(current_question + possible_title, k=self.k)
            all_retrieved_snippets.extend(snippets)

            # 3. 事实提取，更新问题（复用 generator 的 client）
            extracted = extract_factual_info_rag(current_question, all_retrieved_snippets, client=self.generator.client)
            current_question = " ".join(extracted) + ". " + last_sentence

        # 4. 生成 CoT 和最终答案
        # 构建 CoT prompt — 允许模型在文档不相关时使用自身知识
        cot_prompt = (
            f"问题：{original_question}\n"
            f"参考资料：{str(all_retrieved_snippets[:3])}\n"
            f"请一步步推理。如果参考资料与问题无关或不足以回答问题，请基于你自己的医学知识进行推理。"
        )
        cot = call_cot([{"role": "user", "content": cot_prompt}], self.generator.client)

        # 判断是否为选择题（包含 A/B/C/D 选项格式）— 仅在需要最终答案时判断
        if not explain_only:
            has_options = bool(re.search(r'[A-D][.、)]\s*\S', original_question))
            if has_options:
                final_prompt = (
                    f"问题：{original_question}\n"
                    f"推理过程：{cot}\n"
                    f"请给出最终选项字母（A/B/C/D）。如果参考资料不足以判断，请基于你自己的医学知识选择。"
                )
            else:
                final_prompt = (
                    f"问题：{original_question}\n"
                    f"推理过程：{cot}\n"
                    f"请给出完整、专业的回答。如果参考资料不相关，请基于你自己的医学知识作答。"
                )
            answer = call_answer([{"role": "user", "content": final_prompt}], self.generator.client)
            answer_text = answer.strip()
        else:
            answer_text = ""
        return {
            "answer": answer_text,
            "cot": cot.strip(),
            "sources": all_retrieved_snippets[:3]
        }