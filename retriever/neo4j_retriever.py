"""
根据用户问题检索三元组:
1.大模型从用户问题中提取实体
2.根据提取的实体在neo4j中检索三元组
3.对检索到的三元组去重
4.将三元组转换为文本,作为大模型回答的上下文

本类不继承任何基类，直接通过 get_context(query) 获取上下文字符串，
由 agent 的 neo4j_search 工具直接调用。
图 schema 对应 indexer/neo4j_index.py 建的节点(中文标签 + name 属性)与关系(带 中文键 属性)。
"""
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph

from models.llm import qwen_llm
from utils.config import neo4j_database, neo4j_password, neo4j_url, neo4j_username
from utils.logger import setup_logger

logger = setup_logger(__name__)


# 模块级单例：所有 Neo4jRetriever 实例共享同一个 Neo4jGraph（底层一个 driver）
# 避免每次请求新建 driver 导致 neo4j-python-driver C 扩展资源泄漏 → 0xC0000005 崩溃
_graph_instance = None


def _get_graph():
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database or None,
        )
    return _graph_instance


class Neo4jRetriever:

    def __init__(self, graph: Neo4jGraph = None, top_k=10):
        # 未传入 graph 时复用模块级单例
        if graph is None:
            graph = _get_graph()
        self.graph = graph
        self.top_k = top_k

        # 实体抽取提示词（适配本项目图 schema：抽取图里实际存在的具体实体名）
        prompt = ChatPromptTemplate.from_template("""
                从以下问题中抽取用于在知识图谱中检索的实体名，返回JSON数组。
                只抽取以下类别的具体实体名，并把数值条件写进词中：
                - 假期类型：年休假/病假/事假/婚假/产假/法定节假日 等
                - 薪资项：基本工资/岗位工资/工龄工资/全勤奖/月度绩效奖金/年终奖/项目专项奖/销售提成奖 等
                - 报销类型：差旅/住宿费/交通费/业务招待费/日常办公费 等
                - 培训类型：入职培训/岗位技能培训/管理进阶培训/外部研修 等
                - 职称等级：初级/中级/副高级/高级
                - 审批角色/部门：部门负责人/人力资源部/财务部/分管领导/总经理/行政部 等
                - 阈值条件（务必把数值写进名称）：如 工龄满10年、年休假5天、住宿费400元/天、连续旷工3天、费用5000元 等
                - 违规行为：代打卡/伪造材料/旷工/虚假发票 等

                要求：
                1. 抽取问题中明确出现的具体实体名，不要返回"制度""条款"这类泛标签。
                2. 数值条件把指标和数值一起写进一个词，如"工龄满10年"。
                3. 去重，返回 2~6 个；若问题无可用实体，返回 []。
                4. 只返回JSON数组，不要任何其他内容。
                例如：["年休假", "工龄满10年"]

                问题：{query}
                """)
        self._extract_chain = prompt | qwen_llm | JsonOutputParser()

    # 从用户问题中提取实体
    def extract(self, query: str) -> list:
        try:
            result = self._extract_chain.invoke({"query": query})
        except Exception as e:
            logger.warning(f"实体抽取失败: {e}")
            return []
        return result if isinstance(result, list) else []

    # 根据实体检索（双向、标签无关，取全部关系属性；与 neo4j_index 建的图一致）
    def retrieve_triples(self, entity_list: list) -> list:
        if len(entity_list) == 0:
            return []

        triples = []
        for entity in entity_list:
            result = self.graph.query(
                """
                MATCH (s {name: $entity_name}) - [r] -> (o)
                RETURN s.name AS subject,
                       type(r) AS predicate,
                       o.name AS object,
                       properties(r) AS props
                UNION
                MATCH (s) <- [r] - (o {name: $entity_name})
                RETURN s.name AS subject,
                       type(r) AS predicate,
                       o.name AS object,
                       properties(r) AS props
                """,
                params={
                    "entity_name": entity
                }
            )
            triples.extend(result or [])
        return triples

    # 三元组去重
    def dedup_triples(self, triples) -> list:
        unique_triples = []
        filter_set = set()

        for t in triples:
            key = (t.get("subject"), t.get("predicate"), t.get("object"))
            if key not in filter_set:
                filter_set.add(key)
                unique_triples.append(t)

        return unique_triples

    # 单条三元组 -> 文本
    def _format_triple(self, t: dict) -> str:
        props = t.get("props") or {}
        prop_str = "; ".join(f"{k}: {v}" for k, v in props.items())
        content = f"{t.get('subject')} - {t.get('predicate')} -> {t.get('object')}"
        if prop_str:
            content += f" ({prop_str})"
        return content

    # 三元组封装
    def get_paths(self, query: str):
        # 提取用户问题中的实体
        entity_list = self.extract(query)
        # 根据实体检索三元组
        triples = self.retrieve_triples(entity_list)
        logger.info(f"工具 neo4j_search 命中 {len(triples)} 条")
        # 去重
        unique_triples = self.dedup_triples(triples)

        paths = [self._format_triple(t) for t in unique_triples[: self.top_k]]
        return paths

    # 把检索到三元组去重, 转成字符串, 作为大模型的上下文
    def get_context(self, query: str):
        paths = self.get_paths(query)
        context = "\n".join(paths)
        return context
