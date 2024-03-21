from langchain.docstore.document import Document
from configs import EMBEDDING_MODEL

from server.utils import BaseResponse
from typing import Dict, List
from server.utils import load_local_embeddings


def embed_texts(
        texts: List[str],
        embed_model: str = EMBEDDING_MODEL,
) -> BaseResponse:
    """
    对文本进行向量化。返回数据格式：BaseResponse(data=List[List[float]])
    """

    embeddings = load_local_embeddings(model=embed_model)
    return BaseResponse(data=embeddings.embed_documents(texts))


def embed_documents(
        docs: List[Document],
        embed_model: str = EMBEDDING_MODEL,
) -> Dict:
    """
    将 List[Document] 向量化，转化为 VectorStore.add_embeddings 可以接受的参数
    """
    texts = [x.page_content for x in docs]
    metadatas = [x.metadata for x in docs]
    embeddings = embed_texts(texts=texts, embed_model=embed_model).data
    if embeddings is not None:
        return {
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }


async def aembed_texts(
        texts: List[str],
        embed_model: str = EMBEDDING_MODEL,
) -> BaseResponse:
    '''
    对文本进行向量化。返回数据格式：BaseResponse(data=List[List[float]])
    '''

    embeddings = load_local_embeddings(model=embed_model)
    return BaseResponse(data=await embeddings.aembed_documents(texts))
