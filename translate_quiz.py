# -*- coding: utf-8 -*-
"""
批量翻译 MMLU 题库前 100 题为中文
使用 DeepSeek API，支持断点续传
"""
import json
import os
import sys
import time
import urllib.request
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
API_URL = 'https://api.deepseek.com/v1/chat/completions'

def call_deepseek(prompt: str) -> str:
    """直接通过 urllib 调用 DeepSeek API"""
    data = json.dumps({
        'model': 'deepseek-chat',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.0,
        'max_tokens': 4096
    }).encode('utf-8')
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        }
    )
    for attempt in range(3):
        try:
            r = urllib.request.urlopen(req, timeout=120)
            result = json.loads(r.read().decode('utf-8'))
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f'  API 调用失败 (尝试 {attempt+1}/3): {e}')
            if attempt < 2:
                time.sleep(3)
    raise RuntimeError('API 调用失败，已达最大重试次数')

def translate_batch(questions: list) -> list:
    """一次翻译 5 道题"""
    # 构建翻译 prompt
    batch_text = ''
    for i, q in enumerate(questions):
        batch_text += f'[{i+1}]\n'
        batch_text += f'Question: {q["question"]}\n'
        batch_text += f'Options:\n'
        for k, v in q['options'].items():
            batch_text += f'  {k}. {v}\n'
        batch_text += f'Answer: {q["answer"]}\n\n'

    prompt = f'''请将以下英文医学选择题翻译为中文。要求：
1. 题干翻译为流畅专业的中文
2. 选项也全部翻译为中文
3. 答案字母（A/B/C/D）保持不变
4. 输出格式严格为 JSON 数组，每个元素包含 question、options（字典）、answer 三个字段
5. 不要输出任何其他内容，只输出 JSON 数组

英文题目：
{batch_text}

JSON 输出：'''

    print('  发送翻译请求...')
    response = call_deepseek(prompt)

    # 提取 JSON（去掉可能的 markdown 代码块标记）
    response = response.strip()
    if response.startswith('```'):
        response = response.split('\n', 1)[1]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()
        if response.startswith('json'):
            response = response[4:].strip()

    try:
        translated = json.loads(response)
        return translated
    except json.JSONDecodeError:
        print(f'  JSON 解析失败，原始响应:\n{response[:500]}')
        # 尝试逐题翻译作为 fallback
        print('  回退到逐题翻译...')
        results = []
        for q in questions:
            results.append(translate_single(q))
        return results

def translate_single(q: dict) -> dict:
    """翻译单道题（fallback）"""
    prompt = f'''请将以下英文医学选择题翻译为中文，输出格式为 JSON：
{{"question": "中文题干", "options": {{"A": "中文选项A", "B": "中文选项B", "C": "中文选项C", "D": "中文选项D"}}, "answer": "{q['answer']}"}}

英文题目：{q["question"]}
选项：{json.dumps(q["options"])}

只输出 JSON：'''
    response = call_deepseek(prompt)
    response = response.strip()
    if response.startswith('```'):
        response = response.split('\n', 1)[1]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()
        if response.startswith('json'):
            response = response[4:].strip()
    return json.loads(response)

def main():
    # 1. 读取 mmlu 子集
    benchmark_path = os.path.join(os.path.dirname(__file__), 'MIRAGE', 'benchmark.json')
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    mmlu_questions = data.get('mmlu', {})
    all_items = list(mmlu_questions.items())
    print(f'MMLU 共 {len(all_items)} 题，取前 100 题翻译')

    target = all_items[:100]

    # 2. 检查已有进度（断点续传）
    output_path = os.path.join(os.path.dirname(__file__), 'MIRAGE', 'mmlu_cn.json')
    done = {}
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            done = json.load(f)
        print(f'已有 {len(done)} 题已完成，从断点继续')

    # 3. 批量翻译
    batch_size = 5
    for start in range(len(done), len(target), batch_size):
        batch = target[start:start + batch_size]
        batch_dicts = [{'id': qid, 'question': q['question'], 'options': q['options'], 'answer': q['answer']} for qid, q in batch]

        print(f'\n[{start+1}-{min(start+batch_size, len(target))}/{len(target)}] 翻译中...')

        try:
            translated = translate_batch(batch_dicts)
            for t in translated:
                # translate_batch 返回的是不带 id 的，我们需要匹配
                pass
        except Exception as e:
            print(f'  批量翻译失败: {e}，逐题重试...')
            translated = []
            for i, qd in enumerate(batch_dicts):
                print(f'    第 {start+i+1} 题...')
                try:
                    t = translate_single(qd)
                    translated.append(t)
                except Exception as e2:
                    print(f'      失败: {e2}')
                    translated.append(qd)  # 保留原文

        # 保存进度
        for i, qd in enumerate(batch_dicts):
            qid = batch[i][0]
            if i < len(translated):
                done[qid] = {
                    'question': translated[i].get('question', qd['question']),
                    'options': translated[i].get('options', qd['options']),
                    'answer': translated[i].get('answer', qd['answer'])
                }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(done, f, ensure_ascii=False, indent=2)

        print(f'  已保存 {len(done)}/100 题')
        time.sleep(0.5)  # 避免 API 限流

    print(f'\n完成！已保存到 {output_path}')

    # 4. 复制一份替换 .env 指向（可选）
    print(f'\n完成后在 .env 中修改：QUIZ_DATA_PATH=./MIRAGE/mmlu_cn.json')

if __name__ == '__main__':
    main()
