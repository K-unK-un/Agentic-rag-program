"""
文档分割
"""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

from utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentSplitter:
    def __init__(self, docs: List[Document]):
        self.docs = docs

    # 先按标题切割，再把切割后的文本块按照RecursiveCharacterTextSplitter切割
    def header_split(self):
        # 按标题切割的切割器：
        headers_splitter_on = [
            ("#", "H1"),
            ("##", "H2"),
            ("###", "H3")
        ]

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_splitter_on
        )
        text = "".join([doc.page_content for doc in self.docs])

        header_chunks = header_splitter.split_text(text)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "？"]
        )

        chunks = splitter.split_documents(header_chunks)

        logger.info(f"按标题文档分割完成，共分割{len(chunks)}个文档")
        return chunks

    def split(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "？"]
        )

        chunks = splitter.split_documents(self.docs)

        logger.info(f"文档分割完成，共分割{len(chunks)}个文档")
        return chunks
