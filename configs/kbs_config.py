import os

from pymilvus import connections

# 默认使用的知识库
DEFAULT_KNOWLEDGE_BASE = "sample"

# 默认向量库，目前可用：milvus
DEFAULT_VS_TYPE = "milvus"

# 知识库中单段文本长度
CHUNK_SIZE = 250

# 知识库中相邻文本重合长度
OVERLAP_SIZE = 50

# 知识库匹配向量数量
VECTOR_SEARCH_TOP_K = 10

# 知识库匹配的距离阈值，一般取值范围0-1之间，SCORE越小，距离越小从而相关读越高。一般为了兼容性默认设为1
SCORE_THRESHOLD = 1.0

# 百度搜索需要KEY
# BAIDU_API_KEY=""
# 每个知识库的初始化介绍，用于在初始化知识库时显示和Agent调用，没写则没有介绍，不会被Agent调用。
KB_INFO = {
    "知识库名称": "知识库介绍",
    "samples": "关于本项目issue的解答",
}

# 知识库默认存储路径
KB_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")
VECTOR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_base")
if not os.path.exists(KB_ROOT_PATH):
    os.mkdir(KB_ROOT_PATH)
# 向量数据库默认存储路径
# VECTOR_DB_PATH = os.path.join(KB_ROOT_PATH, "vector_db")


# 向量数据库配置
kbs_config = {
    "milvus": {
        "host": "127.0.0.1",
        "port": "19530",
        "user": "",
        "password": "",
        "secure": False,
    },

    "milvus_kwargs": {
        "search_params": {"HNSW": {"metric_type": "L2", "params": {"nprobe": 8}}},  # 在此处增加search_params
        "index_params": {"metric_type": "L2", "index_type": "HNSW", "params":  {"nlist": 128}}  # 在此处增加index_params
    },
}

# TEXT_SPLITTER 名称
TEXT_SPLITTER_NAME = "ChineseRecursiveTextSplitter"
