from pymilvus import utility
from pymilvus import connections, Collection

# 连接到 Milvus
connections.connect(host='localhost', port='19530')
utility.drop_collection("demo")
