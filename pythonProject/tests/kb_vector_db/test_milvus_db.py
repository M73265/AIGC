from server.knowledge_base.kb_service.milvus_kb_service import MilvusKBService

search_content = "如何设置虚拟墙"

kbService = MilvusKBService("demo")


def test_search_db():
    result = kbService.search_docs(search_content)
    print(result)


test_search_db()
