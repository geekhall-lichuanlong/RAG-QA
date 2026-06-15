# import pandas as pd
# import json
# import re

# jsonl_file_path = r"C:\Users\SUS\KnowledgeBase-RAG-LLM-System\test4mmlu.jsonl"

# n = 0
# x = 0
# # Read JSON Lines file and create a DataFrame
# json_lines = []
# with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
#     for line in jsonl_file:
#         n+=1
#         data = json.loads(line)
#         ground_truth = data['ground_truth']
#         print("ground_truth:", ground_truth)
#         answers = str(data['model_answer'])
#         match = re.search(r'[A-D]', answers)
#         answers = match.group(0)
#         print("answers:", answers)
#         if any(answer.lower() in answers.lower() for answer in ground_truth):
#             x+=1

# print(x/n)

#改后
import json
import re

jsonl_file_path = r"C:\Users\SUS\KnowledgeBase-RAG-LLM-System\test4mmlu.jsonl"

n = 0
x = 0
with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
    for line in jsonl_file:
        n += 1
        data = json.loads(line)
        ground_truth = data.get('ground_truth', '')
        print("ground_truth:", ground_truth)
        
        model_answer = data.get('model_answer', '')
        # 提取选项字母（支持 "(A)"、"A"、"{'answer_choice': 'A'}" 等格式）
        answers = ''
        if isinstance(model_answer, str):
            # 匹配括号内的字母或单独的字母
            match = re.search(r'\(?([A-D])\)?', model_answer)
            if match:
                answers = match.group(1)
            else:
                # 尝试解析 JSON
                try:
                    obj = json.loads(model_answer)
                    answers = obj.get('answer_choice', '')
                except:
                    pass
        else:
            answers = str(model_answer)
        
        print("answers:", answers)
        if answers and any(answer.lower() in answers.lower() for answer in ground_truth):
            x += 1

print(f"Accuracy: {x}/{n} = {x/n:.2%}")