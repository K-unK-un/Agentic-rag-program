"""
队减速结果进行rerank精排
"""
from typing import Annotated, List

from langchain_core.tools import tool

from reranker.reranker import Reranker
from tools.doc_store import DocStore
from utils.config import max_rerank_docs
from utils.logger import setup_logger

logger = setup_logger(__name__)

def create_rerank_tool(doc_store: DocStore):
    reranker = Reranker()
    @tool
    def rerank_doc(
            query: Annotated[str, "用户的问题"],
            doc_ids: Annotated[List[str], "doc_store中存储的文档id列表"]
    ):
        '''
        根据检索的结果进行排序，得到排序后的结果，只要进行检索后，都要调用该工具
        '''
        # 根据doc_ids获取原文档
        document = doc_store.get_batch(doc_ids)
        # 如果文档个数超过设定的词频，则将超过的文档抛弃掉
        if len(document) > max_rerank_docs:
            document = document[:max_rerank_docs]

        # 重排（复用单例模型；参数顺序由 Reranker.rerank 封装保证）
        ranked = reranker.rerank(query, document)

        if len(ranked) == 0:
            return []

        results = []
        for doc in ranked:
            score = doc.metadata.get("relevance_score", 0)
            dic = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": score
            }
            results.append(dic)

        logger.info(f"reranker工具被调用")

        return results

    return rerank_doc

