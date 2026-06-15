import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from template import PROMPT_DICT
from torch.utils.data import Dataset, DataLoader
from openai import OpenAI


def call_deepseek_api(messages, client):
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    return completion.choices[0].message.content

def custom_json_decoder(obj):
    if 'id' in obj:
        obj['id'] = str(obj['id'])
    return obj

class llmDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def process_prompt(self, item):
        id = item['id']
        query = item['query']
        passages = item['passages']
        ground_truth = item['ground_truth']
        cot = item['model_output']
        template = PROMPT_DICT['QA_queryCoT_to_answer']
        template = template.format(question=query, CoT=cot)
        messages = [
            {"role": "user", "content": template},
        ]
        item['input_prompt'] = template
        return item

    def __getitem__(self, index):
        item = self.data[index]
        item = self.process_prompt(item)
        if index == 0:
            print(item)
        return item

    def __len__(self):
        return len(self.data)

    def Collactor(self, batch):
        id = [f['id'] for f in batch]
        query = [f['query'] for f in batch]
        passages = [f['passages'] for f in batch]
        ground_truth = [f['ground_truth'] for f in batch]
        COT = [f['model_output'] for f in batch]
        input_prompt = [f['input_prompt'] for f in batch]
        return {'id': id, 'query': query, 'passages': passages, 'COT': COT, 'ground_truth': ground_truth, 'input_prompt': input_prompt}

def inference(args):

    client = OpenAI(
        base_url='https://api.deepseek.com',
        api_key=os.getenv('DEEPSEEK_API_KEY', 'sk-db9e2fb08a7f4b34bee4772ec1deacc7')
    )

    # Load data from the JSONL file
    with open(args.data_path, 'r') as file:
        data = [json.loads(line, object_hook=custom_json_decoder) for line in file]


    dataset = llmDataset(data)
    dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=dataset.Collactor)

    output_data = []

    for batch in tqdm(dataloader):
        input_prompt = batch['input_prompt'][0]
        messages = [{"role": "user", "content": input_prompt}]
        model_output = call_deepseek_api(messages, client)

        id = batch['id'][0]
        query = batch['query'][0]
        passages = batch['passages'][0]
        cot = batch['COT'][0]
        ground_truth = batch['ground_truth'][0]

        output_item = {
            "id": id,
            "query": query,
            "passages": passages,
            "COT": cot,
            "model_answer": model_output,
            "ground_truth": ground_truth
        }
        output_data.append(output_item)

    with open(args.output_name, 'w') as outfile:
        for item in output_data:
            json.dump(item, outfile)
            outfile.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r"C:\Users\SUS\KnowledgeBase-RAG-LLM-System\test3mmlu.jsonl")
    parser.add_argument('--output_name', type=str, default=r"C:\Users\SUS\KnowledgeBase-RAG-LLM-System\test4mmlu.jsonl")
    args = parser.parse_args()
    inference(args)

