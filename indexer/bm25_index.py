import json
import os

from langchain_core.documents import Document

from models.chroma import vectorstore
from retriever.bm25_retriever import CustomBM25Retriever
from utils.logger import setup_logger

logger = setup_logger(__name__)


class Bm25Index:
    def get_data(self):
        all_data = vectorstore.get(include=["documents"])
        texts = all_data.get("documents", [])

        os.makedirs("./data", exist_ok=True)
        with open("./data/chunks.json", "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)

        logger.info(f"bm25获取数据成功")

    def load_bm25(self):
        """从JSON文件加载文本构建BM25"""
        self.get_data()
        with open("./data/chunks.json", "r", encoding="utf-8") as f:
            texts = json.load(f)

        # 构建BM25检索器
        chunks = [Document(page_content=t) for t in texts]
        bm25_retriever = CustomBM25Retriever(chunks)
        logger.info(f"bm25构建成功")
        retriever = bm25_retriever.get_retriever()
        return retriever
