import importlib
import os
import uuid
from typing import List, Dict

from langchain.vectorstores.milvus import Milvus
from langchain.document_loaders import UnstructuredExcelLoader
from configs import kbs_config, EMBEDDING_MODEL, TEXT_SPLITTER_NAME, KB_ROOT_PATH
from server.knowledge_base.kb_service.base import EmbeddingsFunAdapter
from langchain.schema import Document


class MilvusEmbedding:
    def __init__(self):
        self.embed_model = EMBEDDING_MODEL

    def do_init(self):
        self._load_milvus()

    milvus: Milvus

    def _load_milvus(self):
        self.milvus = Milvus(embedding_function=EmbeddingsFunAdapter(self.embed_model),
                             collection_name="demo",
                             connection_args=kbs_config.get("milvus"),
                             index_params=kbs_config.get("milvus_kwargs")["index_params"],
                             search_params=kbs_config.get("milvus_kwargs")["search_params"],
                             auto_id=True
                             )

    def do_add_doc(self, docs: List[Document], **kwargs) -> List[Dict]:
        # ids = self.milvus.add_documents(docs)
        # return ids
        for doc in docs:
            for k, v in doc.metadata.items():
                doc.metadata[k] = str(v)
            for field in self.milvus.fields:
                doc.metadata.setdefault(field, "")
            doc.metadata.pop(self.milvus._text_field, None)
            doc.metadata.pop(self.milvus._vector_field, None)

        ids = self.milvus.add_documents(docs)
        print(ids)
        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        print(doc_infos)
        return doc_infos


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


def initialize_and_load_text_embeddings(**kwargs):
    docs = read_and_split_text_files(os.path.join(KB_ROOT_PATH, "samples", "content", "test_files"))

    milvus_embedding = MilvusEmbedding()
    milvus_embedding.do_init()
    ids_added = milvus_embedding.do_add_doc(docs, **kwargs)

    print(ids_added)


initialize_and_load_text_embeddings()
