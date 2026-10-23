"""
"""
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from utils.logger import setup_logger

logger = setup_logger(__name__)

class CustomBM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def get_retriever(self):
        if isinstance(self.chunks[0], Document):
            retriever = BM25Retriever.from_documents(self.chunks, k=3)
        else:
            retriever = BM25Retriever.from_texts(self.chunks, k=3)

        return retriever


