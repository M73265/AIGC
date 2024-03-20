import numpy as np
from langchain.chains.qa_with_sources import vector_db

from configs.model_config import ONLINE_LLM_MODEL, LLM_MODEL
import httpx
from typing import Union, Dict
import logging
import os
from langchain_community.chat_models import QianfanChatEndpoint
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from pymilvus import Milvus
from langchain_core.documents import Document
from langchain.chains.question_answering import load_qa_chain
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from configs import KB_ROOT_PATH

# 创建milvus检索器
# def find_similar_vectors(query_vector, vectors_to_use):
#     similarities = cosine_similarity([query_vector], vectors_to_use)
#     most_similar_index = similarities.argmax()
#     most_similar_vector = vectors_to_use[most_similar_index]
#     return most_similar_vector
#
#
# class MilvusRetriever:
#     def __init__(self, milvus):
#         self.milvus = milvus
#         self.model = SentenceTransformer('moka-ai/m3e-base')
#
#     def text_to_vector(self, text):
#         vector = self.model.encode(text)
#         return vector
#
#     def retrieve_vectors(self, collection_name):
#         # 从原始集合中检索向量数据
#         query_vector_list = []  # 存储检索到的向量数据
#         query_result = self.milvus.query(collection_name=collection_name, expr=None)
#         for entity in query_result:
#             vector = entity.entity.get("embedding", [])
#             query_vector_list.append(vector)
#         return query_vector_list
#
#     def find_similar_vectors(self, query_vector, vectors_to_use):
#         similarities = cosine_similarity([query_vector], vectors_to_use)
#         most_similar_index = similarities.argmax()
#         most_similar_vector = vectors_to_use[most_similar_index]
#         return most_similar_vector
#
#     def get_document_by_vector(self, target_vector, documents):
#         for document in documents:
#             if np.array_equal(target_vector, document.vector):
#                 return document
#         return None
#
#
# def query_baidu_qianfan(question):
#     os.environ["QIANFAN_AK"] = "BhW3TK98jwfFDKB5GmUdGUuB"
#     os.environ["QIANFAN_SK"] = "F0wcGyPGkEl207VZ6qMDsSnP6TIP6Kbg"
#     chat = QianfanChatEndpoint(model='ERNIE-Bot-4', temperature=0.7,max_tokens=1024)
#
#
#     # 将查询语句转化为特征向量
#     question_vector = retriever.text_to_vector(question)
#     # 使用相似度检索找到与查询向量最相似的向量数据
#     most_similar_vector = retriever.find_similar_vectors(question_vector, vectors_to_use)
#     documents = os.path.join(KB_ROOT_PATH, "test_files")
#     similar_document = retriever.get_document_by_vector(most_similar_vector, documents)
#     prompt = ChatPromptTemplate.from_template("""使用下面的语料来回答本模板最末尾的问题。如果你不知道问题的答案，直接回答 "我不知道"，禁止随意编造答案。
#             为了保证答案尽可能简洁，你的回答必须不超过三句话，你的回答中不可以带有星号。
#             请注意！在每次回答结束之后，你都必须接上 "感谢你的提问" 作为结束语
#             以下是一对问题和答案的样例：
#                 请问：秦始皇的原名是什么
#                 秦始皇原名嬴政。感谢你的提问。
#             以下是语料：
#     <context>
#     {context}
#     </context>
#     Question: {input}""")

    # qianfan_config = ONLINE_LLM_MODEL.get("qianfan-api", {})
    # api_key = qianfan_config.get("api_key")
    # secret_key = qianfan_config.get("secret_key")
    # model_version = "ernie-bot-4"
    #
    # # access_token = get_baidu_access_token(api_key, secret_key)
    # access_token = "24.f38cc5dd1986f6ae64f37e7b2ff347f9.2592000.1710848426.282335-48744446"
    # if not access_token:
    #     print("Failed to obtain access token. Please check your API key and secret key.")
    #     return None
    #
    # url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model_version}?access_token={access_token}"
    # headers = {"Content-Type": "application/json"}
    # payload = {
    #     "messages": [{"role": "user", "content": question}],
    #     "temperature": 0.7,  # 温度参数，控制生成答案的 kreativity，取值范围 [0.0, 1.0]，值越大越 creative。
    #     "stream": True  # 是否返回多轮对话结果。若设为 False 则只返回一次对话结果。若设为 True 则返回多轮对话结果，且对话结果中包含历史轮对话结果。
    # }
    #
    # try:
    #     response = requests.post(url, json=payload, headers=headers)
    #     if response.status_code == 200:
    #         data = response.json()
    #         answer = data.get("result")
    #         return answer
    #     else:
    #         logging.info("Failed to query Baidu QianFan API:", response.status_code)
    #         return None
    # except Exception as e:
    #     logging.info("Exception occurred while querying Baidu QianFan API:", str(e))
    #     return None

# def get_baidu_access_token(api_key: str, secret_key: str) -> str:
#     """
#     使用 AK，SK 生成鉴权签名（Access Token）
#     :return: access_token，或是None(如果错误)
#     """
#     url = "https://aip.baidubce.com/oauth/2.0/token"
#     params = {"grant_type": "client_credentials", "client_id": api_key, "client_secret": secret_key}
#     try:
#         with get_httpx_client() as client:
#             return client.get(url, params=params).json().get("access_token")
#     except Exception as e:
#         print(f"failed to get token from baidu: {e}")
#
#
#
#
# def get_httpx_client(
#         use_async: bool = False,
#         proxies: Union[str, Dict] = None,
#         timeout: float = 30.0,
#         **kwargs,
# ) -> Union[httpx.Client, httpx.AsyncClient]:
#
#     default_proxies = {
#
#         "http://127.0.0.1": None,
#         "http://localhost": None,
#     }
#
#     for x in [
#         fschat_controller_address(),
#         fschat_model_worker_address(),
#     ]:
#         host = ":".join(x.split(":")[:2])
#         default_proxies.update({host: None})
#
#     if proxies is None:
#         proxies = default_proxies
#     else:
#
#         proxies = {**default_proxies, **proxies}
#
#     if use_async:
#         return httpx.AsyncClient(proxies=proxies, timeout=timeout, **kwargs)
#     else:
#         return httpx.Client(proxies=proxies, timeout=timeout, **kwargs)
#
# def fschat_controller_address() -> str:
#     from configs.server_config import FSCHAT_CONTROLLER
#
#     host = FSCHAT_CONTROLLER["host"]
#     if host == "0.0.0.0":
#         host = "127.0.0.1"
#     port = FSCHAT_CONTROLLER["port"]
#     return f"http://{host}:{port}"
# def fschat_model_worker_address(model_name: str = LLM_MODEL[0]) -> str:
#     if model := get_model_worker_config(model_name):
#         host = model["host"]
#         if host == "0.0.0.0":
#             host = "127.0.0.1"
#         port = model["port"]
#         return f"http://{host}:{port}"
#     return ""
#
#
#
# def get_model_worker_config(model_name: str = None,  log_verbose: bool = False) -> dict:
#     '''
#     加载model worker的配置项。
#     '''
#     from configs.model_config import ONLINE_LLM_MODEL
#
#     from server import model_workers
#
#     config = ONLINE_LLM_MODEL.get("qianfan-api", {}).copy()
#
#     if model_name:
#         logging.warning(f"Ignoring model_name '{model_name}'. Using 'qianfan-api' from ONLINE_LLM_MODEL.")
#
#     if provider := config.get("provider"):
#         try:
#             config["worker_class"] = getattr(model_workers, provider)
#             config["online_api"] = True
#         except Exception as e:
#             msg = f"在线模型 ‘qianfan-api’ 的 provider 没有正确配置"
#             logging.error(f'{e.__class__.__name__}: {msg}', exc_info=e if log_verbose else None)
#
#     return config
