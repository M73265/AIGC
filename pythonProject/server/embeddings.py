import logging
import os
from configs import KB_ROOT_PATH
from document_loaders.mypdfloader import RapidOCRPDFLoader
from text_splitter import ChineseRecursiveTextSplitter
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

from langchain.document_loaders import UnstructuredExcelLoader
import numpy as np




_COLLECTION_NAME = 'demo'
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'float_vector_field'
_TEXT_FIELD_NAME = 'text_field'
_DIM = 768


#首先读取本地文本文件并进行切割处理

_HOST = '127.0.0.1'
_PORT = '19530'


def create_connection(milvus_host, milvus_port):
    print(f"\nCreate connection...")
    connections.connect(host=_HOST, port=_PORT)
    print(f"\nList connections:")
    print(connections.list_connections())


def read_and_split_text_files(directory):
    global docs
    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=200,
        chunk_overlap=0
    )
    files_chunks = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith('.pdf'):
            # pdf文件要进行一个数据清洗
            loader = RapidOCRPDFLoader(file_path)
            docs = loader.load()
            output_file_path = "/home/zhouzhixiang1/zhuimi1-AIGC/pythonProject/knowledge_base/samples/content/test_files/output.txt"
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                for doc in docs:
                    output_file.write(doc.page_content + "\n\n")  # 写入文档内容并换行
            with open(output_file_path, "r", encoding="utf-8") as input_file:
                file_content = input_file.read()
            chunks = text_splitter.split_text(file_content)
            # text_splitter.split_documents(docs)
            files_chunks.extend(chunks)
        elif filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                file_data = file.read()
                chunks = text_splitter.split_text(file_data)
                files_chunks.extend(chunks)
        elif filename.endswith('.xlsx'):
            #  加载 Excel 文件
            loader = UnstructuredExcelLoader(file_path)
            docs1 = loader.load()
            print(docs1)
            chunks = text_splitter.split_documents(docs1)
            print(chunks)
            files_chunks.extend(chunks)


            # 将 Excel 数据转换为文本

            # output_file_path1 = "/home/zhouzhixiang1/zhuimi1-AIGC/pythonProject/knowledge_base/samples/content/test_files/output1.txt"  # 指定输出文件路径
            # with open(output_file_path1, "w", encoding="utf-8") as output_file1:
            #     for doc in docs1:
            #         output_file1.write(doc.page_content + "\n\n")  # 写入文档内容并换行
            # with open(output_file_path1, "r", encoding="utf-8") as input_file1:
            #     file_content1 = input_file1.read()
            # 使用 text_splitter 分割文本内容
            # chunks = text_splitter.split_text(docs1)
            # chunks = text_splitter.split_text(file_content1)
            # files_chunks.extend(chunks)
    return files_chunks


def initialize_and_load_text_embeddings(
    milvus_host,
    milvus_port,
):
    try:
        # 连接到 Milvus

        connections.connect(host=milvus_host, port=milvus_port)
        print(connections.list_connections())
        logging.info("向量数据库连接成功！")

        # 如果向量数据库不存在，则创建
        if has_collection(_COLLECTION_NAME):
            drop_collection(_COLLECTION_NAME)

        collection = create_collection(_COLLECTION_NAME, _ID_FIELD_NAME, _VECTOR_FIELD_NAME, _TEXT_FIELD_NAME)


        # 本地文本文件切割
        files_chunks = read_and_split_text_files(os.path.join(KB_ROOT_PATH, "samples", "content", "test_files"))
        print(files_chunks)
        # files_chunks = [str(i) for i in files_chunks]
        #指定本地模型路径
        # model_path = 'remote_model/m3e-base'
        # 使用预训练模型进行向量化并存储到 Milvus 中
        os.environ["https_proxy"] = "socks5://192.168.0.11:20170"
        EMBEDDING_MODEL = SentenceTransformer('moka-ai/m3e-base')
        # 对文本数据进行向量化
        embeddings = EMBEDDING_MODEL.encode(files_chunks)
        print(type(embeddings))

        embeddings_np = np.array(embeddings).tolist()
        # collection.insert([embeddings_np])

        insert(collection, embeddings_np, files_chunks)
        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }
        collection.create_index("float_vector_field", index)

        # milvus.insert(collection_name=collection_name, records=embeddings)
    except Exception as e:
        # 打印错误信息或者记录日志
        logging.error(f"An error occurred: {e}")


def insert(collection, embeddings_np,files_chunks):
    batch_size = 100
    for i in range(0, len(embeddings_np), batch_size):
        data = [
            list(range(i, min(i + batch_size, len(embeddings_np)))),  # 选择要插入的索引范围
            embeddings_np[i:min(i + batch_size, len(embeddings_np))],
            files_chunks[i:min(i + batch_size, len(embeddings_np))]
        ]
        collection.insert(data)





def has_collection(name):
    return utility.has_collection(name)


def drop_collection(name):
    collection = Collection(name)
    collection.drop()
    print("\nDrop collection: {}".format(name))


def create_collection(name, id_field, vector_field, text_field):
    field1 = FieldSchema(name=id_field, dtype=DataType.INT64, description="int64", is_primary=True)
    field2 = FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, description="float vector", dim=_DIM,
                         is_primary=False)
    field3 = FieldSchema(name=text_field, dtype=DataType.VARCHAR, description="string", auto_id=False, max_length=1000
                         )
    schema = CollectionSchema(fields=[field1, field2, field3], description="collection description")
    collection = Collection(name=name, data=None, schema=schema, properties={"collection.ttl.seconds": 15})
    print("\ncollection created:", name)
    return collection


def list_collections():
    print("\nlist collections:")
    print(utility.list_collections())



