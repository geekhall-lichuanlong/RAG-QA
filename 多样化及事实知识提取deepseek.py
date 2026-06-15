import json
import re
from openai import OpenAI
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer
import os
import faiss
import tqdm
import numpy as np
import torch
from template import *

corpus_names = {
    "Textbooks": ["textbooks"]
}

retriever_names = {
    "MedCPT": [r"C:\Users\SUS\KnowledgeBase-RAG-LLM-System\models\MedCPT-Query-Encoder"]
}


# def embed(chunk_dir, index_dir, model_name, **kwarg):
#     # index_dir = "/home/qluai/lcl/RGAR-C613/moxing/MedCPT-Article-Encoder"######新加
#     # chunk_dir = "/home/qluai/lcl/RGAR-C613/corpus/textbooks/chunk"
#     # model_name="/home/qluai/lcl/RGAR-C613/moxing/MedCPT-Article-Encoder"
#     save_dir = os.path.join(index_dir, "embedding")#save_dir=/home/qluai/lcl/RGAR-C613/MedCPT-Article-Encoder/embedding

#     model = CustomizeSentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

#     model.eval()#将模型设置为评估模式，关闭一些在训练时使用的特殊层（如 Dropout）。

#     fnames = sorted([fname for fname in os.listdir(chunk_dir) if fname.endswith(".jsonl")])
# #fnames=[Anatomy_Gray.jsonl,Biochemistry_Lippincott.jsonl]
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     with torch.no_grad():
#         for fname in tqdm.tqdm(fnames):#遍历每个jsonl文件。
#             fpath = os.path.join(chunk_dir, fname)
#             save_path = os.path.join(save_dir, fname.replace(".jsonl", ".npy"))
#             if os.path.exists(save_path):
#                 continue
#             if open(fpath).read().strip() == "":
#                 continue
#             texts = [json.loads(item) for item in open(fpath).read().strip().split('\n')]
#             if "medcpt" in model_name.lower():
#                 texts = [[item["title"], item["content"]] for item in texts]
#             else:
#                 texts = [concat(item["title"], item["content"]) for item in texts]
#             embed_chunks = model.encode(texts, **kwarg)
#             np.save(save_path, embed_chunks)
#         embed_chunks = model.encode([""], **kwarg)
#     return embed_chunks.shape[-1]

#改
def embed(chunk_dir, index_dir, model_name, **kwarg):
    save_dir = os.path.join(index_dir, "embedding")
    model = CustomizeSentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    fnames = sorted([fname for fname in os.listdir(chunk_dir) if fname.endswith(".jsonl")])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for fname in tqdm.tqdm(fnames):
            fpath = os.path.join(chunk_dir, fname)
            save_path = os.path.join(save_dir, fname.replace(".jsonl", ".npy"))
            if os.path.exists(save_path):
                continue
            # 使用 UTF-8 编码读取文件，并跳过空行
            with open(fpath, 'r', encoding='utf-8-sig') as f:
                content = f.read().strip()
                if content == "":
                    continue
                lines = content.split('\n')
                items = []
                for line in lines:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
            if "medcpt" in model_name.lower():
                texts = [[item["title"], item["content"]] for item in items]
            else:
                texts = [concat(item["title"], item["content"]) for item in items]
            embed_chunks = model.encode(texts, **kwarg)
            np.save(save_path, embed_chunks)
        embed_chunks = model.encode([""], **kwarg)
    return embed_chunks.shape[-1]


def construct_index(index_dir, model_name, h_dim=768, HNSW=False,
                    M=32):

    with open(os.path.join(index_dir, "metadatas.jsonl"), 'w') as f:
        f.write("")

    if HNSW:
        M = M
        index = faiss.IndexHNSWFlat(h_dim, M)
        index.metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        index = faiss.IndexFlatIP(h_dim)

    for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(index_dir, "embedding")))):
        curr_embed = np.load(os.path.join(index_dir, "embedding", fname))
        index.add(curr_embed)
        with open(os.path.join(index_dir, "metadatas.jsonl"), 'a+') as f:
            f.write("\n".join([json.dumps({'index': i, 'source': fname.replace(".npy", "")}) for i in
                               range(len(curr_embed))]) + '\n')

    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    return index


def extract_factual_info_rag(question, retrieved_snippets, client=None):
    num_sentences, other_sentences, last_sentence = split_sentences(question)
    contexts = [
        "Document [{:d}] (Title: {:s}) {:s}".format(
            idx,
            retrieved_snippets[idx]["title"],
            retrieved_snippets[idx]["content"]
        )
        for idx in range(len(retrieved_snippets))
    ]

    #

    context_str = "\n".join(contexts)
    answers = []

    prompt_extract = general_extract_nolist.render(
        context=context_str,
        ehr=other_sentences,
        question=last_sentence
    )

    # 使用 DeepSeek API（复用传入的 client，未传入时自动创建）
    if client is None:
        import ssl as _ssl
        import httpx as _httpx
        ctx = _ssl.create_default_context()
        ctx.minimum_version = _ssl.TLSVersion.TLSv1_2
        ctx.maximum_version = _ssl.TLSVersion.TLSv1_2
        _http_client = _httpx.Client(verify=ctx, proxy=None, trust_env=False, timeout=60.0)
        client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY', ''),
            base_url="https://api.deepseek.com/v1",
            http_client=_http_client,
        )
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt_extract}],
            temperature=0.0,
            max_tokens=8192
        )
        answer = response.choices[0].message.content
        answers.append(re.sub(r"\s+", " ", answer).strip())
    except Exception as e:
        print(f"提取事实信息时出错: {str(e)}")
        answers.append("")
    return answers

def split_sentences(text):

        text = text.rstrip('"').strip()

        pattern = r'(.*?[.!?。\n])'
        sentences = re.findall(pattern, text, re.DOTALL)
        #print("sentences:", sentences)
        if not sentences:
            return 0, "", ""

        last_sentence = sentences[-1].strip()
        other_sentences = "".join(sentences[:-1]).strip()

        return len(sentences), other_sentences, last_sentence


def data_save(id,data_type,question,answer,passage):
                json_line = {
                    "id": id,
                    "data_type": data_type,
                    "question": question,
                    "answer": answer,
                    "passage": passage
                }
                out_f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

class CustomizeSentenceTransformer(SentenceTransformer):

    def _load_auto_model(self, model_name_or_path, *args, **kwargs):

        print("No sentence-transformers model found with name {}. Creating a new one with CLS pooling.".format(model_name_or_path))
        token = kwargs.get('token', None)
        cache_folder = kwargs.get('cache_folder', None)
        revision = kwargs.get('revision', None)
        trust_remote_code = kwargs.get('trust_remote_code', False)
        if 'token' in kwargs or 'cache_folder' in kwargs or 'revision' in kwargs or 'trust_remote_code' in kwargs:
            transformer_model = Transformer(
                model_name_or_path,
                cache_dir=cache_folder,
                model_args={"token": token, "trust_remote_code": trust_remote_code, "revision": revision},
                tokenizer_args={"token": token, "trust_remote_code": trust_remote_code, "revision": revision},
            )
        else:
            transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'cls')
        return [transformer_model, pooling_model]


class Retriever:

    def __init__(self, retriever_name="ncbi/MedCPT-Query-Encoder", corpus_name="textbooks", db_dir="./corpus",
                 HNSW=False, **kwarg):
        self.retriever_name = r"C:\Users\SUS\KnowledgeBase-RAG-LLM-System\models\MedCPT-Query-Encoder"
        self.corpus_name = corpus_name

        self.db_dir = db_dir  # db_dir=./corpus
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
        self.chunk_dir = os.path.join(self.db_dir, self.corpus_name, "chunk")  # chunk_dir=./corpus/textbooks/chunk
        self.index_dir = os.path.join(self.db_dir, self.corpus_name, "index",
                                      self.retriever_name.replace("Query-Encoder","Article-Encoder"))
        if "bm25" in self.retriever_name.lower():
            print("不会执行")
        else:
            if os.path.exists(os.path.join(self.index_dir, "faiss.index")):
                self.index = faiss.read_index(os.path.join(self.index_dir, "faiss.index"))
                self.metadatas = [json.loads(line) for line in
                                  open(os.path.join(self.index_dir, "metadatas.jsonl")).read().strip().split('\n')]
            else:
                print("[In progress] Embedding the {:s} corpus with the {:s} retriever...".format(self.corpus_name,self.retriever_name.replace("Query-Encoder","Article-Encoder")))

                h_dim = embed(chunk_dir=self.chunk_dir, index_dir=self.index_dir,model_name=self.retriever_name.replace("Query-Encoder", "Article-Encoder"),**kwarg)  # 对指定目录下的文本进行编码，生成嵌入向量，并获取这些嵌入向量的维度，将维度值赋给变量 h_dim

                print("[In progress] Embedding finished! The dimension of the embeddings is {:d}.".format(h_dim))
                self.index = construct_index(index_dir=self.index_dir,model_name=self.retriever_name.replace("Query-Encoder", "Article-Encoder"),h_dim=h_dim, HNSW=HNSW)
                print("[Finished] Corpus indexing finished!")
                self.metadatas = [json.loads(line) for line in open(os.path.join(self.index_dir, "metadatas.jsonl")).read().strip().split('\n')]


            self.embedding_function = CustomizeSentenceTransformer(self.retriever_name, device="cpu")
            self.embedding_function.eval()

    def get_relevant_documents(self, question, k=32, id_only=False,**kwarg):
        assert type(question) == str
        question = [question]

        if "bm25" in self.retriever_name.lower():
            print("不会执行这里")
        else:
            with torch.no_grad():
                query_embed = self.embedding_function.encode(question, **kwarg)
            res_ = self.index.search(query_embed,k=k)
            ids = ['_'.join([self.metadatas[i]["source"], str(self.metadatas[i]["index"])]) for i in res_[1][0]]
            indices = [self.metadatas[i] for i in res_[1][0]]

        scores = res_[0][0].tolist()
        if id_only:
            return [{"id": i} for i in ids], scores
        else:
            return self.idx2txt(
                indices), scores

    def idx2txt(self, indices):
        return [json.loads(
            open(os.path.join(self.chunk_dir, i["source"] + ".jsonl")).read().strip().split('\n')[i["index"]]) for i in indices]


class RetrievalSystem:

    def __init__(self, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", HNSW=False,
                 cache=False):
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        assert self.corpus_name in corpus_names
        assert self.retriever_name in retriever_names
        self.retrievers = []
        for retriever in retriever_names[self.retriever_name]:
            self.retrievers.append([])
            for corpus in corpus_names[self.corpus_name]:
                self.retrievers[-1].append(Retriever(retriever, corpus, db_dir, HNSW=HNSW))
        self.cache = cache
        if self.cache:
            self.docExt = DocExtracter(cache=True, corpus_name=self.corpus_name, db_dir=db_dir)
        else:
            self.docExt = None

    def retrieve(self, question, k=32, id_only=False):
        '''
            Given questions, return the relevant snippets from the corpus
        '''
        assert type(question) == str

        output_id_only = id_only
        if self.cache:
            id_only = True

        texts = []
        scores = []

        k_ = k
        for i in range(len(retriever_names[self.retriever_name])):
            texts.append([])
            scores.append([])
            for j in range(len(corpus_names[self.corpus_name])):
                t, s = self.retrievers[i][j].get_relevant_documents(question, k=k_,
                                                                    id_only=id_only)
                texts[-1].append(t)
                scores[-1].append(s)

        all_texts = []
        all_scores = []

        for i in range(len(retriever_names[self.retriever_name])):
            for j in range(len(corpus_names[self.corpus_name])):
                all_texts.extend(texts[i][j])
                all_scores.extend(scores[i][j])

        # Sort by score descending
        sorted_indices = np.argsort(all_scores)[::-1]
        texts = [all_texts[i] for i in sorted_indices[:k]]
        scores = [all_scores[i] for i in sorted_indices[:k]]
        if self.cache:
            texts = self.docExt.extract(texts)
        return texts, scores

    def retrieve_multi(self, questions, k, id_only=False):
        assert isinstance(questions, list)

        # Store all retrieved docs and their scores
        all_docs = {}  # id -> {doc_info, max_score, query_scores}

        # Retrieve for each question
        for question in questions:
            texts, scores = self.retrieve(question, k, id_only=id_only)

            # Process each retrieved document
            for doc, score in zip(texts, scores):
                doc_id = doc['id']

                if doc_id not in all_docs:
                    # First time seeing this document
                    all_docs[doc_id] = {
                        'doc': doc,
                        'max_score': score,
                        'query_scores': {question: score}
                    }
                else:
                    # Update existing document info
                    all_docs[doc_id]['max_score'] = max(all_docs[doc_id]['max_score'], score)
                    all_docs[doc_id]['query_scores'][question] = score

        # Sort documents by their maximum scores
        sorted_docs = sorted(all_docs.items(), key=lambda x: x[1]['max_score'],reverse=True)

        # Format output
        merged_texts = []
        merged_scores = []

        for doc_id, doc_info in sorted_docs:
            merged_texts.append(doc_info['doc'])

            merged_scores.append(doc_info['max_score'])

            if not id_only:
                doc_info['doc']['query_scores'] = doc_info['query_scores']

        return merged_texts[:k], merged_scores[:k]



class AnswerGenerator:
    def __init__(self):
        import ssl
        import httpx
        # 强制 TLS 1.2，禁用代理（Windows 上 TLS 1.3 和系统代理可能不兼容）
        ctx = ssl.create_default_context()
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.maximum_version = ssl.TLSVersion.TLSv1_2
        http_client = httpx.Client(verify=ctx, proxy=None, trust_env=False, timeout=60.0)
        self.client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY', ''),
            base_url="https://api.deepseek.com/v1",
            http_client=http_client,
        )

    def generate_possible_answer(self, question):
        prompt = '''Please give 4 options for the question. Each option should be a concise description of a key detail, formatted as: A. "key detail 1" B. "key detail 2" C. "key detail 3" D. "key detail 4"'''
        messages = [
            {"role": "user", "content": question + "\n" + prompt},
        ]
        ans = self.generate(messages)
        cleaned = re.sub(r"\s+", " ", ans).strip()
        return cleaned

    def generate_possible_content(self, question):
        prompt = '''Please generate some knowledge that might address the above question. please give me only the knowledge."'''
        messages = [
            {"role": "user", "content": question + "\n" + prompt},
        ]
        ans = self.generate(messages)
        cleaned = re.sub(r"\s+", " ", ans).strip()
        return cleaned

    def generate_possible_title(self, question):
        prompt = '''Please generate some titles of references that might address the above question. Please give me only the titles, formatted as: ["title 1", "title 2", ..., "title N"]. Please be careful not to give specific content and analysis, just the title.'''
        messages = [
            {"role": "user", "content": question + "\n" + prompt},
        ]
        ans = self.generate(messages)
        cleaned = re.sub(r"\s+", " ", ans).strip()
        return cleaned

    def generate(self, messages):
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    # 读取 benchmark.json 文件
    with open(r"C:\Users\SUS\KnowledgeBase-RAG-LLM-System\MIRAGE\mmlu.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 初始化生成器
    generator = AnswerGenerator()

    k = 3
    # 遍历 medqa 的前几个样本


    retrieval_system = RetrievalSystem()
    output_path = r"C:\Users\SUS\KnowledgeBase-RAG-LLM-System\test1mmlu.jsonl"
    with open(output_path, "w", encoding="utf-8") as out_f:
        for dataset_name, dataset_samples in data.items():
             print(f"Processing dataset: {dataset_name}")
             for qid, sample in list(dataset_samples.items()):
                  i=0
                  question = sample["question"]
                  num_sentences, other_sentences, last_sentence = split_sentences(question)
                  while True:
                    all_retrieved_snippets = []
                    print(f"\nQuestion ID: {qid}")
                    #print(f"Original Question:\n{question}")

                    possible_answers = generator.generate_possible_answer(question)
                    print(f"\nGenerated Possible Answers:\n{possible_answers}")

                    retrieved_snippets, scores = retrieval_system.retrieve(question + possible_answers, k=k)
                    #print("可能答案retrieved_snippets:", retrieved_snippets)
                    print("scores:", scores)
                    all_retrieved_snippets.extend(retrieved_snippets)

                    possible_content = generator.generate_possible_content(question)
                    print(f"\nGenerated Possible Content:\n{possible_content}")

                    retrieved_snippets, scores = retrieval_system.retrieve(question + possible_content, k=k)
                    # print("retrieved_snippets:", retrieved_snippets)
                    # print("scores:", scores)
                    all_retrieved_snippets.extend(retrieved_snippets)

                    possible_title = generator.generate_possible_title(question)
                    print(f"\nGenerated Possible Title:\n{possible_title}")


                    retrieved_snippets, scores = retrieval_system.retrieve(question + possible_title, k=k)
                   # print("retrieved_snippets:", retrieved_snippets)
                   # print("scores:", scores)
                    all_retrieved_snippets.extend(retrieved_snippets)

                    print("all_retrieved_snippets:", all_retrieved_snippets)

                    extract_sentences = extract_factual_info_rag(question, all_retrieved_snippets)
                    print("extract_sentences:", extract_sentences)


                    question = "\n".join(extract_sentences) +"."+ last_sentence
                    #print("question:", question)
                    i+=1
                    if i>=2:
                        options_str = "\nOptions:\n" + "\n".join([f"({k}) {v}" for k, v in sample["options"].items()])
                        full_question = sample["question"] + options_str

                        id = qid
                        data_type=dataset_name
                        question = full_question
                        answer = sample["answer"]
                        passage=all_retrieved_snippets

                        data_save(id, data_type, question,  answer, passage)
                        #print("当前问题最终的all_retrieved_snippets:", all_retrieved_snippets)
                        break






