"""
检索向量数据库
如果单独进行向量检索，可以使用：vector_retrieve、vector_retrieve_with_score
混合检索只需要获取retriever即可，可以使用：get_retriever
"""


class VectorRetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def vector_retrieve(self, query):
        # 使用向量数据库做相似度查询
        # self.vectorstore.similar_search(self.query, k=3)

        # 定义检索器，在其中可以设置检索策略
        retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.5
            }
        )

        chunks = retriever.invoke(query)

        return chunks

    # similarity_search_with_score:返回分数的相似度检索，返回结果是[(Document, score),(Document, score),...]
    # 根据分数做相对灵活的处理
    def vector_retrieve_with_score(self, query):
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=5
        )
        # cosine相似度计算：计算出cosine值之后做(1-cosine)，分数越小越相似
        # 判断：分数大于0.5的文档被过滤
        chunks = [doc for doc, score in results if score <= 0.5]

        return chunks

    # 获取相似度的检索器
    def get_retriever(self):
        retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.5
            }
        )
        return retriever
