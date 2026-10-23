"""
基于 EnsembleRetriever 的 RRF 融合检索器。

langchain-classic 的 EnsembleRetriever 已实现：
1. 多路并行召回（arank_fusion 内部为 asyncio.gather）
2. 文档去重（id_key=None 时按 page_content 内容去重）
3. 加权倒数排序融合（score += weight / (rank + c)，c默认60）并排序

唯一缺口：融合分数不写回文档metadata。
本子类仅补写 rrf_score，其余逻辑全部复用父类。
"""
from collections import defaultdict

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

'''
自定义混合检索类，继承EnsembleRetriever
作用：在检索到文档片段并进行RRF融合时，将融合分数补写回文档metadata
'''

class RrfEnsembleRetriever(EnsembleRetriever):

    # 去重键与父类保持一致：id_key为None时按内容，否则取指定元数据字段（缺失时兜底为内容）
    def _make_key(self, doc: Document) -> str:
        if self.id_key:
            return str(doc.metadata.get(self.id_key, doc.page_content))
        return doc.page_content

    def weighted_reciprocal_rank(self, doc_lists):
        # 按父类相同的RRF公式预先计算各文档的融合分数
        rrf_score = defaultdict(float)
        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score[self._make_key(doc)] += weight / (rank + self.c)

        # 去重、排序完全复用父类
        # 父类进行RRF计算后返回的文档默认没有分数(rrf_score)
        ranked_docs = super().weighted_reciprocal_rank(doc_lists)

        # 把融合分数补写进metadata
        for doc in ranked_docs:
            # 在每个文档的metadata中添加rrf_score
            doc.metadata["rrf_score"] = round(rrf_score[self._make_key(doc)], 6)

        return ranked_docs