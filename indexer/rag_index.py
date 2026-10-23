"""
创建索引：
1.向量数据库索引
2.BM25索引
3.图知识库(neo4j)
4.混合检索器(EnsembleRetriever)

- create_index：全量构建（写向量库 + 抽取三元组建图），耗时较高，由 /ingest 调用
- load_retrievers：廉价加载（连接已存在的 Chroma/Neo4j，仅重建内存中的 BM25），由服务启动调用
"""
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from langchain_core.documents import Document

from indexer.neo4j_index import Neo4jIndex
from indexer.vector_index import VectorIndex
from loader.document_loader import DocumentLoader
from loader.document_splitter import DocumentSplitter
from models.chroma import vectorstore

# 项目根目录（indexer/ 的上一级）
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Retrievers:
    """索引构建产出的检索器集合"""
    bm25: Any  # BM25 关键词检索器
    vector: Any  # 向量语义检索器
    neo4j: Any  # Neo4j 图谱检索器
    hybrid: Any  # 三路加权混合检索器


def _default_files() -> List[str]:
    """默认加载 resources/ 下的全部文件"""
    return sorted(glob.glob(str(BASE_DIR / "resources" / "*")))


def _load_chunks(files: List[str]) -> List[Document]:
    """加载并分割文档为 chunk（供 BM25 等内存检索器使用）"""
    docs = []
    for path in files:
        docs.extend(DocumentLoader(path).load())
    return DocumentSplitter(docs).header_split()


def create_index(
        vector_delete_flag: bool = True,
        neo4j_delete_flag: bool = False
):
    # 加载并分割文档（默认 resources/*.md）
    files_path = _default_files()
    chunks = _load_chunks(files_path)

    # 向量数据库（按需清空后写入）
    vector_index = VectorIndex(
        chunks,
        vectorstore=vectorstore,
        delete_flag=vector_delete_flag,
    )

    # 词嵌入模型
    vector_index.add_documents()

    # 图知识库(neo4j)：用提示词驱动大模型抽取三元组并写入
    neo4j_index = Neo4jIndex(chunks, delete_flag=neo4j_delete_flag)
    neo4j_index.build()


if __name__ == "__main__":
    # create_index()
    pass

