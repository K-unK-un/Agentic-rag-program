"""
文档加载
"""
from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader
from langchain_mineru import MinerULoader

from utils.logger import setup_logger

# 创建日志对象
logger = setup_logger(__name__)


class DocumentLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    # 文档加载
    def markdown_load(self):
        md_loader = UnstructuredMarkdownLoader(file_path=self.file_path)
        docs = md_loader.load()

        logger.info("加载markdown文档成功")

        return docs

    # minerU加载器：可以加载pdf、word等
    def mineru_load(self):
        mu_loader = MinerULoader(source=self.file_path)
        docs = mu_loader.load()

        logger.info("minerU加载文档成功")

        return docs

    def text_load(self):
        text_loader = TextLoader(self.file_path, encoding='utf-8')

        docs = text_loader.load()

        logger.info("加载文本文档成功")
        return docs

    # 定义通用加载器
    def load(self):
        # 获取文件名的后缀
        suffix = self.file_path.split(".")[-1]
        if suffix == "md":
            return self.markdown_load()
        elif suffix == "txt":
            return self.text_load()
        elif suffix in ["pdf", "doc", "docx"]:
            return self.mineru_load()
        else:
            logger.error("不支持该文件格式")
            raise Exception("不支持该文件格式")
