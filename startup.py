import operator
import os
# from abc import ABC
# from typing import List, Tuple, Optional, Any
from abc import ABC, abstractmethod
from langchain_core.vectorstores import VectorStore
import pandas as pd
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.chat_models import QianfanChatEndpoint
# from langchain_community.document_loaders import TextLoader
# from langchain_core.prompts import PromptTemplate
# from configs import EMBEDDING_MODEL
# from server.kb_service.base import EmbeddingsFunAdapter
# from server.model_workers import qianfan
# from server.embeddings import initialize_and_load_text_embeddings, create_connection
from langchain_community.vectorstores.milvus import Milvus
from sentence_transformers import SentenceTransformer
# import logging
from pymilvus import connections, Collection, utility
from configs import kbs_config, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD
# from langchain.docstore.document import Document
# from langchain_core.embeddings import Embeddings
# from langchain_core.vectorstores import VectorStore
# # EmbeddingsFunAdapter = EMBEDDING_MODEL
# 配置日志记录器
# from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# milvus_config = kb_config.get("milvus", {})

# CUSTOM_PROMPT_TEMPLATE = """
#         使用下面的语料来回答本模板最末尾的问题。如果你不知道问题的答案，直接回答 "我不知道"，禁止随意编造答案。
#         请注意！在每次回答结束之后，你都必须接上 "感谢你的提问" 作为结束语
#         以下是一对问题和答案的样例：
#             请问：秦始皇的原名是什么
#             秦始皇原名嬴政。感谢你的提问。
#         以下是语料：
#         {context}
#         请问：{question}
#     """
#
# knowledge_base_dir = "/home/zhouzhixiang1/zhuimi1-AIGC/pythonProject/knowledge_base/samples/content/test_files"







# def main():
    # os.environ["QIANFAN_AK"] = "BhW3TK98jwfFDKB5GmUdGUuB"
    # os.environ["QIANFAN_SK"] = "F0wcGyPGkEl207VZ6qMDsSnP6TIP6Kbg"
    # llm = QianfanChatEndpoint(streaming=True, model="ERNIE-Bot-4")
    # chat = QianfanChatEndpoint(model="ERNIE-Bot-4")

    # os.environ["https_proxy"] = "socks5://192.168.0.11:20170"
    # EMBEDDING_MODEL = SentenceTransformer('moka-ai/m3e-base')

    # logger.info("正在向量化文件...")
    # initialize_and_load_text_embeddings('127.0.0.1', '19530')
    # connections.connect(host='127.0.0.1', port='19530')
    # logger.info("文件向量化完成,启动大模型服务...")
    # 启动问答交互服务
    # 输入问题

    # question = input("请输入您的问题：")
    # 将用户提出的问题向量化

    # embeddings = EMBEDDING_MODEL.encode(question)

    # 连接到Milvus
    # search_params = {
    #     "metric_type": "L2",
    #     "params": {"nprobe": 10},
    # }
    # schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
    # hello_milvus = Collection("demo")
    # hello_milvus.load()
    # query = "虚拟墙如何设置"
    # top_k = 5
    # score_threshold = 1.0
    # # MilvusKBService1 = MilvusKBService()
    # os.environ["https_proxy"] = "socks5://192.168.0.11:20170"
    # EMBEDDING_MODEL = SentenceTransformer('moka-ai/m3e-base')
    # embeddings = EMBEDDING_MODEL.encode(query)
    # print(201)
    # print(embeddings)
    # print(102)



    # def similarity_search_with_score_by_vector1(
    #         self,
    #         embedding: List[float],
    #         k: int = 4,
    #         param: Optional[dict] = None,
    #         expr: Optional[str] = None,
    #         timeout: Optional[int] = None,
    #         **kwargs: Any,
    # ) -> List[Tuple[Document, float]]:
    #     col = Collection("demo")
    #
    #     param = kb_config.get("milvus_kwargs")["search_params"]

        # Determine result metadata fields.
        # output_fields = [x for x in self.fields if x != self._primary_field]
        # output_fields = ["text_field"]
        # _vector_field = "float_vector_field"
        # output_fields.remove(_vector_field)

        # Perform the search.
        # res = col.search(
        #     data=[embedding],
        #     anns_field=_vector_field,
        #     param=param,
        #     limit=k,
        #     expr=expr,
        #     output_fields=output_fields,
        #     timeout=timeout,
        #     **kwargs,
        # )
        # search_params = {
        #     "metric_type": "L2",
        #     "params": {"nprobe": 10},
        # }
        # print(200)
        # print(embedding)
        # print(100)
        # embedding = embeddings
        #
        # res = hello_milvus.search([embedding], "float_vector_field", search_params, limit=5,
        #                           output_fields=["text_field"])
        # for hits in res:
        #     for hit in hits:
        #         print(hit)
        #
        # print()

        # Organize results.

        # ret = []
        # for result in res[0]:
        #     data = {x: result.entity.get(x) for x in output_fields}
        #     doc = self._parse_document(data)
        #     pair = (doc, result.score)
        #     ret.append(pair)
        #
        # return ret

    # embeddings_list = embeddings.tolist()
    # print(embeddings)
    # docs = similarity_search_with_score_by_vector1(embeddings, top_k)
    # docs1 = score_threshold_process(score_threshold, top_k, docs)
    # print(docs)
    # print(docs1)

    # return score_threshold_process(score_threshold, top_k, docs)

    # result = MilvusKBService.do_search(MilvusKBService1, query, 5, 1.0)
    # print(result)
    # return result
    # milvus = connections.connect(host="127.0.0.1", port=19530)

    # docs = hello_milvus.search([embeddings], "float_vector_field", search_params, limit=5, output_fields=["text_field"])
    # for hits in docs:
    #     for hit in hits:
    #         print(hit)
    # 处理匹配结果
    # print(docs)

    # chat = QianfanChatEndpoint(model='ERNIE-Bot-4', temperature=0.7)
    # prompt = generate_prompt(docs, question)
    # answer = generate_answer(prompt, question)
    # qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=hello_milvus)

    # print(answer.content)
    # 将问题输出给大模型，大模型在向量数据库中寻找匹配的答案，问答形式按提供的prompt模板返回
    # prompt_template = prompt_config.PROMPT_TEMPLATES["llm_chat"]["default"]

    # 查询大模型的回答
    # answer_from_model = qianfan.query_baidu_qianfan(prompt_template.format(input=question))
    # 返回大模型的回答
    # print("Answer from model:", answer_from_model)


# def generate_prompt(context, question):
#     # 提取文本信息
#     context_texts = []
#
#     for hits in context:
#         for hit in hits:
#             context_texts.append(hit.entity.get('text_field'))
#
#     return _PROMPT_TEMPLATE.format(context="\n".join(context_texts), question=question)
#
#
# def generate_answer(context, question):
#     # 设置环境变量
#     os.environ["QIANFAN_AK"] = "BhW3TK98jwfFDKB5GmUdGUuB"
#     os.environ["QIANFAN_SK"] = "F0wcGyPGkEl207VZ6qMDsSnP6TIP6Kbg"
#
#     # 创建大模型实例
#     chat = QianfanChatEndpoint(model="ERNIE-Bot-4")
#
#     # 提交问题给大模型
#     response = chat.invoke(input=context + " " + question)
#     print(response.content)
#     return response

# 根据相似度阈值，过滤出符合条件的相似度结果，并返回其中最相似的前 k 个结果。
# def score_threshold_process(score_threshold, k, docs):
#     if score_threshold is not None:
#         cmp = (
#             operator.le
#         )
#         docs = [
#             (doc, similarity)
#             for doc, similarity in docs
#             if cmp(similarity, score_threshold)
#         ]
#     return docs[:k]


# class KBService(ABC):
#     def search_docs(self,
#                     query: str,
#                     top_k: int = VECTOR_SEARCH_TOP_K,
#                     score_threshold: float = SCORE_THRESHOLD,
#                     ) -> List[Document]:
#         docs = self.do_search(query, top_k, score_threshold)
#         return docs
#
#     @abstractmethod
#     def do_search(self,
#                   query: str,
#                   top_k: int,
#                   score_threshold: float,
#                   ) -> List[Tuple[Document, float]]:
#         """
#
#         """


# if __name__ == "__main__":
#     main()
