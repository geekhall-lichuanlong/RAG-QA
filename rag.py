# from langchain_core.documents import Document
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from file_history_store import get_history
# from vector_stores import VectorStoreService
# from langchain_community.embeddings import DashScopeEmbeddings
# import config_data as config
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_community.chat_models.tongyi import ChatTongyi


# def print_prompt(prompt):
#     print("="*20)
#     print(prompt.to_string())
#     print("="*20)

#     return prompt


# class RagService(object):
#     def __init__(self):

#         self.vector_service = VectorStoreService(
#             embedding=DashScopeEmbeddings(model=config.embedding_model_name)
#         )

#         self.prompt_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", 
#          "你是一个专业的问答助手。请基于提供的参考资料进行简洁专业的回答。\n"
#          "参考资料：{context}\n"
#          "以下是用户的历史对话："),
#         MessagesPlaceholder("history"),
#         ("user", "请回答用户提问：{input}")
#     ]
# )

#         self.chat_model = ChatTongyi(model=config.chat_model_name)

#         self.chain = self.__get_chain()

#     def __get_chain(self):
#         """获取最终的执行链"""

#         retriever = self.vector_service.get_retriever()

#         def format_document(docs: list[Document]):
#             if not docs:
#                 return "无相关参考资料"

#             formatted_str = ""
#             for doc in docs:
#                 formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"

#             return formatted_str

#         def format_for_retriever(value: dict)->str:

#             return value["input"]

#         def format_for_prompt_template(value):
#             # {input, context, history}
#             new_value = {}
#             new_value["input"] = value["input"]["input"]
#             new_value["context"] = value["context"]
#             new_value["history"] = value["input"]["history"]
#             return new_value


#         chain = (
#             {
#                 "input": RunnablePassthrough(),
#                 "context": RunnableLambda(format_for_retriever) | retriever | format_document
#             }| RunnableLambda(format_for_prompt_template) |self.prompt_template | print_prompt |self.chat_model | StrOutputParser()
#         )

#         conversation_chain = RunnableWithMessageHistory(       # 增强的链
#             chain,
#             get_history,
#             input_messages_key="input",
#             history_messages_key="history",
#         )

#         return conversation_chain


# if __name__ == '__main__':
#     # session id 配置
#     session_config ={
#         "configurable":{
#             "session_id":"user_001",
#         }
#     }
#     res = RagService().chain.invoke({"input":"我之前问了什么"},session_config)
#     print(res)

from langchain_core.documents import Document
from langchain_core.runnables.history import RunnableWithMessageHistory
from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config

# 新增
from enhanced_rag import EnhancedRAG


class RagService(object):

    def __init__(self):

        # 向量库（保留原项目）
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(
                model=config.embedding_model_name
            )
        )

        # 你的增强RAG
        self.enhanced_rag = EnhancedRAG()

    # =====================================
    # 新聊天接口
    # =====================================

    def chat(self, question):

        try:

            # =====================================
            # 1. 原项目 retriever
            # =====================================

            retriever = self.vector_service.get_retriever()

            docs = retriever.invoke(question)

            # =====================================
            # 2. 转成你的 passage 格式
            # =====================================

            retrieved_docs = []

            for doc in docs:

                retrieved_docs.append({
                    "title": str(doc.metadata),
                    "content": doc.page_content
                })

            # =====================================
            # 3. 使用你的增强RAG
            # =====================================

            result = self.enhanced_rag.chat(
                question=question,
                external_docs=retrieved_docs
            )

            return result["answer"]

        except Exception as e:

            return f"系统错误: {str(e)}"


if __name__ == '__main__':

    rag = RagService()

    res = rag.chat("什么是肺癌")

    print(res)