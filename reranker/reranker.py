"""
创建reranker
"""
import os

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from utils.logger import setup_logger

logger = setup_logger(__name__)

# 模块级单例：所有 Reranker 实例共享同一个 CrossEncoderReranker
_reranker_instance = None


def _get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        model_path = os.path.join(
            os.getenv("HF_HOME"),
            "hub",
            "models--BAAI--bge-reranker-large",
            "snapshots",
            "55611d7bca2a7133960a6d3b71e083071bbfc312"
        )
        cross_encoder = HuggingFaceCrossEncoder(model_name=model_path)
        _reranker_instance = CrossEncoderReranker(model=cross_encoder, top_n=5)
        logger.info("reranker模型加载完成")
    return _reranker_instance


class Reranker:
    def __init__(self):
        # 复用模块级单例，不再每次加载模型
        self._reranker = _get_reranker()

    def get_reranker(self):
        return self._reranker

    def rerank(self, query, docs):
        return self._reranker.compress_documents(docs, query)
