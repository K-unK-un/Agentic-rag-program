from typing import List

from langchain_core.documents import Document

from retriever.vector_retriever import VectorRetriever
from utils.logger import setup_logger

logger = setup_logger(__name__)


class VectorIndex:
    def __init__(self, chunks: List[Document], vectorstore, delete_flag: bool = True):
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.delete_flag = delete_flag

    # 删除向量数据库中的数据
    def delete_vectors(self):
        ids = self.vectorstore.get()["ids"]
        if len(ids) > 0:
            self.vectorstore.delete(ids=ids)
            logger.info("删除向量数据库完成")

    # 添加数据到向量数据库
    def add_documents(self):
        if self.delete_flag:
            self.delete_vectors()

        batch_size = 200  # 如果报错再设小一点
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            self.vectorstore.add_documents(batch)

        logger.info("向量数据库索引创建完成")

    def get_retriever(self):
        vector_retrieve = VectorRetriever(self.vectorstore)

        vector_retriever = vector_retrieve.get_retriever()
        return vector_retriever
