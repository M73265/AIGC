from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)
import logging
_DIM = 1792
_COLLECTION_NAME = 'demo'
_ID_FIELD_NAME = 'pk'
_VECTOR_FIELD_NAME = 'vector'
_TEXT_FIELD_NAME = 'text'


def has_collection(name):
    return utility.has_collection(name)
def drop_collection(name):
    collection = Collection(name)
    collection.drop()
    print("\nDrop collection: {}".format(name))
connections.connect(host='127.0.0.1', port='19530')
logging.info("向量数据库连接成功！")
if has_collection(_COLLECTION_NAME):
    drop_collection(_COLLECTION_NAME)
def create_collection(name, primary_field, vector_field, text_field):
    field1 = FieldSchema(name=primary_field, dtype=DataType.INT64, description="primary_field", is_primary=True)
    field2 = FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, description="vector_field", dim=_DIM,
                         is_primary=False)
    field3 = FieldSchema(name=text_field, dtype=DataType.VARCHAR, description="text_field", auto_id=False,
                         max_length=1000
                         )
    schema = CollectionSchema(fields=[field1, field2, field3], description="collection description")
    collection = Collection(name=name, data=None, schema=schema, properties={"collection.ttl.seconds": 15})
    print("\ncollection created:", name)
    return collection

create_collection(_COLLECTION_NAME, _ID_FIELD_NAME, _VECTOR_FIELD_NAME, _TEXT_FIELD_NAME)

