import logging
import os

import numpy as np
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.vectorstores.milvus import Milvus
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)
from sentence_transformers import SentenceTransformer

from configs import TEXT_SPLITTER_NAME, KB_ROOT_PATH

import importlib

_DIM = 1792
_COLLECTION_NAME = 'demo'
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'vector_field'
_TEXT_FIELD_NAME = 'text_field'


def read_and_split_text_files(directory):
    splitter_name: str = TEXT_SPLITTER_NAME
    text_splitter_module = importlib.import_module('text_splitter')
    TextSplitter = getattr(text_splitter_module, splitter_name)
    text_splitter = TextSplitter(chunk_size=200, chunk_overlap=50)
    files_chunks = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith('.xlsx'):
            #  加载 Excel 文件
            loader = UnstructuredExcelLoader(file_path)
            doc = loader.load()

            docs = text_splitter.split_documents(doc)
            files_chunks.extend(docs)
            print((type(docs)))
            print(docs)
            # page_contents = [doc.page_content for doc in docs]
            return docs


def has_collection(name):
    return utility.has_collection(name)


def drop_collection(name):
    collection = Collection(name)
    collection.drop()
    print("\nDrop collection: {}".format(name))


def initialize_and_load_text_embeddings(
        milvus_host,
        milvus_port,
):
    try:
        # 连接到 Milvus

        connections.connect(host=milvus_host, port=milvus_port)
        logging.info("向量数据库连接成功！")
        if has_collection(_COLLECTION_NAME):
            drop_collection(_COLLECTION_NAME)

        collection = create_collection(_COLLECTION_NAME, _ID_FIELD_NAME, _VECTOR_FIELD_NAME, _TEXT_FIELD_NAME)

        docs = read_and_split_text_files(os.path.join(KB_ROOT_PATH, "samples", "content", "test_files"))
        page_contents = [doc.page_content for doc in docs]
        os.environ["https_proxy"] = "socks5://192.168.0.11:20170"

        EMBEDDING_MODEL = SentenceTransformer('infgrad/stella-mrl-large-zh-v3.5-1792d')
        # embeddings = embed_func.embed_documents(docs)

        embeddings = EMBEDDING_MODEL.encode(page_contents, normalize_embeddings=False)


        embeddings_np = np.array(embeddings).tolist()
        print(type(embeddings_np))
        print(embeddings_np)

        data = [
            [i for i in range(len(embeddings_np))],  # 选择要插入的索引范围
            embeddings_np[0:len(embeddings_np)],
            page_contents[0:len(embeddings_np)]
        ]
        print(data)
        collection.insert(data)
    except Exception as e:
        # 打印错误信息或者记录日志
        logging.error(f"An error occurred: {e}")


def create_collection(name, id_field, vector_field, text_field):
    field1 = FieldSchema(name=id_field, dtype=DataType.INT64, description="int64", is_primary=True)
    field2 = FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, description="vector_field", dim=_DIM,
                         is_primary=False)
    field3 = FieldSchema(name=text_field, dtype=DataType.VARCHAR, description="text_field", auto_id=False,
                         max_length=1000
                         )
    schema = CollectionSchema(fields=[field1, field2, field3], description="collection description")
    collection = Collection(name=name, data=None, schema=schema, properties={"collection.ttl.seconds": 15})
    print("\ncollection created:", name)
    return collection


initialize_and_load_text_embeddings('127.0.0.1', '19530')
