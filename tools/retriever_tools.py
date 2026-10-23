"""
检索器工具

把 BM25 / 向量 / Neo4j 三路检索器封装为 @tool，供 agent 自主判断调用。
每个工具的 docstring 即是 agent 选择该工具的依据，故需写清"适用场景"。
"""

from langchain_core.documents import Document
from langchain_core.tools import tool

from indexer.bm25_index import Bm25Index
from models.chroma import vectorstore
from retriever.hybrid_retriever import RrfEnsembleRetriever
from retriever.neo4j_retriever import Neo4jRetriever
from retriever.vector_retriever import VectorRetriever
from tools.doc_store import DocStore
from utils.logger import setup_logger

logger = setup_logger(__name__)


def build_retriever_tools(doc_store: DocStore):

    bm25_retriever = Bm25Index().load_bm25()
    vector_retriever = VectorRetriever(vectorstore).get_retriever()
    neo4j_retriever = Neo4jRetriever()

    def to_compact(doc: Document, route: str, score: float | None = None):
        doc_id = doc_store.add_doc(doc)

        snippet = doc_store.get_snippet(doc)

        item = {
            'doc_id': doc_id,
            'route': route,
            'snippet': snippet,
        }

        if score is not None:
            item['score'] = score

        return item

    @tool
    def bm25_search(query: str):
        """基于 BM25 的关键词倒排检索。
        适用场景：用户提问包含精确术语、编号、条款名、法规名、专有名词
        （如 21.75、医疗期、住房公积金、TIERED_BY 等），需要按字面精确命中含该词的条款时使用。
        不擅长理解同义或模糊表达。
        需要单独进行BM25检索时调用该工具。"""
        chunks = bm25_retriever.invoke(query)
        logger.info(f"工具 bm25_search 命中 {len(chunks)} 条")

        results = []

        for doc in chunks:
            result = to_compact(doc, 'bm25', score=doc.metadata.get("score"))
            results.append(result)
        return results

    @tool
    def vector_search(query: str):
        """基于向量嵌入的语义相似度检索。
        适用场景：用户用自然语言描述需求、意图较模糊、需按语义理解匹配条款
        （如"请假怎么扣钱""加班给多少""怎么评职称"等泛问），不依赖字面一致。
        当需要单独进行向量检索时调用该工具。"""
        chunks = vector_retriever.invoke(query)
        logger.info(f"工具 vector_search 命中 {len(chunks)} 条")
        results = []
        for doc in chunks:
            result = to_compact(doc, 'vector')
            results.append(result)

        return results

    # 混合检索工具
    @tool
    def hybrid_search(query: str):
        """
        基于 RRF（倒数排名融合）的向量 + BM25 混合检索，融合语义相似与字面命中两路结果。
        适用场景：用户问题既含具体术语/编号/数值又含模糊语义，单走向量或 BM25 可能漏召回，
        需同时兼顾字面精确与语义理解（如"年休假10天怎么算""病假工资按什么标准发"
        "住宿费400元/天谁审批"等带关键词又需语义匹配的问题），或不确定该用哪一路检索时使用。
        返回融合排序后的 final_top_k（默认3）个文档片段。
        当需要综合语义+关键词混合检索时调用该工具。
        """
        # 初始化混合检索器
        hybrid_retriever = RrfEnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        chunks = hybrid_retriever.invoke(query)
        logger.info(f"工具 hybrid_search 命中 {len(chunks)} 条")
        chunks = chunks[:3]
        results = []
        for doc in chunks:
            result = to_compact(doc, 'hybrid', score=doc.metadata.get("rrf_score"))
            results.append(result)
        return results

    @tool
    def neo4j_search(query: str):
        """基于 Neo4j 图谱的关系 / 多跳检索。
        适用场景：涉及跨制度串联、阶梯阈值核算（工龄/天数/金额/比例）、触发链
        （如旷工→解除合同）、审批层级、实体间关系推理的问题
        （如"工龄6年病假10天工资怎么发""连续旷工几天解除合同""住宿费报销谁审批"）。
        前需要单独进行图知识库检索时调用该工具。"""
        triples = neo4j_retriever.get_paths(query)

        results = []

        for triple in triples:
            doc = Document(page_content=triple, metadata={"source": "neo4j"})
            result = to_compact(doc, 'graph')
            results.append(result)

        return results

    return [bm25_search, vector_search, hybrid_search, neo4j_search]
