import json


input_file_path =  r'C:\Users\SUS\KnowledgeBase-RAG-LLM-System\test1mmlu.jsonl'
output_file_path =  r'C:\Users\SUS\KnowledgeBase-RAG-LLM-System\test2mmlu.jsonl'


with open(input_file_path, 'r', encoding='utf-8') as infile:

    with open(output_file_path, 'w', encoding='utf-8') as outfile:

        for line in infile:

            data = json.loads(line)

            data['passages'] = data.pop('passage', [])

            for passage in data.get('passages', []):
                passage['segment'] = passage.pop('content', '')

            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')

print(f"数据已从 {input_file_path} 读取并修改后保存到 {output_file_path}")