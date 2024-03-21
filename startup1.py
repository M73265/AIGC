
import os


from langchain_community.document_loaders import UnstructuredExcelLoader


from langchain_community.vectorstores.chroma import Chroma


from configs import TEXT_SPLITTER_NAME, KB_ROOT_PATH, EMBEDDING_MODEL, VECTOR_PATH

import importlib

from server.knowledge_base.kb_service.base import EmbeddingsFunAdapter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}


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

            # page_contents = [doc.page_content for doc in docs]
            return docs


def initialize_and_load_text_embeddings():
    # 连接到 Milvus

    docs = read_and_split_text_files(os.path.join(KB_ROOT_PATH, "samples", "content", "test_files"))
    # page_contents = [doc.page_content for doc in docs]
    os.environ["https_proxy"] = "socks5://192.168.0.11:20170"

    # embeddings = embed_func.embed_documents(docs)

    # embeddings = EMBEDDING_MODEL.encode(page_contents, normalize_embeddings=False)
    persist_dir = os.path.join(VECTOR_PATH, ".vectordb")
    embeddings = EmbeddingsFunAdapter(EMBEDDING_MODEL)
    if os.path.exists(persist_dir):
        # 从本地持久化文件中加载
        print("从本地向量加载数据...")
        vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        vector_store = Chroma.from_documents(documents=docs,
                                             embedding=embeddings,
                                             persist_directory=persist_dir)

        vector_store.persist()
    query = "如何设置虚拟墙"

    docs1 = vector_store.similarity_search_with_score(query)
    print(docs1)


initialize_and_load_text_embeddings()
