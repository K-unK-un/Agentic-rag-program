"""
创建chroma数据库
"""
from langchain_chroma import Chroma

from models.llm import embeddings

vectorstore = Chroma(
    host="localhost",
    port=8081,
    embedding_function=embeddings,
    collection_name="default",
    collection_metadata={"hnsw:space": "cosine"}
)
