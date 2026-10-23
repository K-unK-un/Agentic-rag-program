"""
从文档中提取三元组，存入neo4j数据库：
1.文档加载和分割(由外部完成，传入 chunks)
2.大模型提取三元组
3.三元组处理(清洗/去重)
4.存入neo4je

图 schema（与 retriever/neo4j_retriever.py 对齐）：
- 节点: 统一用 :ENTITY 标签 + name 属性（检索器按 name 匹配，标签无关）
- 关系: 类型用 UPPER_SNAKE（Neo4j 友好，反引号引用保安全），属性用 constraint 键存条件
- 检索器读取 type(r) 作 predicate、properties(r) 作 props，与本类写入完全一致
"""
import re
from typing import List, Dict

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph

from models.llm import qwen_llm
from utils.config import neo4j_url, neo4j_username, neo4j_password, neo4j_database
from utils.logger import setup_logger

logger = setup_logger(__name__)


class Neo4jIndex:
    def __init__(self, chunks: List[Document], delete_flag: bool = False):
        self.chunks = chunks
        self.delete_flag = delete_flag
        self.graph = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database or None,
        )

        # 三元组抽取提示词（适配企业员工制度文档）
        prompt = ChatPromptTemplate.from_template("""
            你是一个知识图谱抽取专家。请从以下"企业员工制度"文档片段中，提取所有关键知识三元组。

            **实体类型**（subject / object 须是以下类别的具体实体名，并把数值条件写进词中）：
            - 假期类型：年休假/病假/事假/婚假/产假/法定节假日 等
            - 薪资项：基本工资/岗位工资/工龄工资/全勤奖/月度绩效奖金/年终奖/项目专项奖/销售提成奖 等
            - 报销类型：差旅/住宿费/交通费/业务招待费/日常办公费 等
            - 培训类型：入职培训/岗位技能培训/管理进阶培训/外部研修 等
            - 职称等级：初级/中级/副高级/高级
            - 审批角色/部门：部门负责人/人力资源部/财务部/分管领导/总经理/行政部 等
            - 阈值条件（务必把数值写进名称）：如 工龄满10年、年休假5天、住宿费400元/天、连续旷工3天、费用5000元 等
            - 违规行为：代打卡/伪造材料/旷工/虚假发票 等

            **关系类型**（predicate 必须是以下 UPPER_SNAKE 之一，按语义选择）：
            - APPLIES_TO：适用于（制度/条款适用于某类人员或情形）
            - GOVERNED_BY：依据/受约束于（某事项依据某制度）
            - REQUIRES_APPROVAL：需经...审批
            - HANDLED_BY：由...办理/负责
            - TIERED_BY：按...阶梯分级（工龄/天数/金额/比例等阈值）
            - TRIGGERS：触发（某行为触发某后果，如旷工→解除合同）
            - RESTRICTS：限制（限制某行为/额度）
            - EXCLUDES：排除/不适用
            - CONFIGURES_WITH：与...配套/结合
            - CALCULATES_WITH：按...计算（薪资/报销公式）
            - FEEDS_INTO：归入/计入（如某假计入年休假）
            - CROSS_REF：跨制度引用
            - ARCHIVED_BY：归档于

            **输出格式**：JSON 数组，每个元素 {{subject, predicate, object, constraint}}
            - subject：头实体名
            - predicate：上面的 UPPER_SNAKE 关系类型
            - object：尾实体名
            - constraint：附加条件（如天数/金额/比例/工龄阈值），没有则留空字符串

            **示例**：
            文档片段："带薪年休假。工龄满1年不满10年的，年休假5天；工龄满10年不满20年的，年休假10天；工龄满20年以上的，年休假15天。年休假由人力资源部统筹安排。"
            提取：
            [
              {{"subject":"年休假","predicate":"TIERED_BY","object":"工龄满1年不满10年","constraint":"年休假5天"}},
              {{"subject":"年休假","predicate":"TIERED_BY","object":"工龄满10年不满20年","constraint":"年休假10天"}},
              {{"subject":"年休假","predicate":"TIERED_BY","object":"工龄满20年以上","constraint":"年休假15天"}},
              {{"subject":"年休假","predicate":"HANDLED_BY","object":"人力资源部","constraint":""}}
            ]

            **要求**：
            1. 只返回 JSON 数组，不要任何其他内容。
            2. 不要遗漏文档中的阈值、比例、天数、金额、审批层级等关键信息。
            3. 数值条件务必写进 object 或 constraint，不要丢失数值。

            **待处理的文档片段**：
            {content}
            """)
        self._chain = prompt | qwen_llm | JsonOutputParser()

    # 大模型提取三元组（遍历所有 chunk，不再只取第一个）
    def extract_triples(self) -> List[Dict]:
        triples = []
        for chunk in self.chunks:
            content = chunk.page_content.strip()
            # 跳过过短的片段，避免浪费调用
            if len(content) < 15:
                continue
            try:
                res = self._chain.invoke({"content": content})
                logger.info(f"单条三元组抽取完成，{res}")
            except Exception as e:
                logger.warning(f"三元组抽取失败(chunk={content[:20]}...): {e}")
                continue
            if isinstance(res, list):
                triples.extend(res)
        logger.info(f"三元组抽取完成，共 {len(triples)} 条")
        return triples

    # 三元组数据处理
    def clean_triples(self, triples: List[Dict]) -> List[Dict]:
        """清洗提取的三元组：去空白、过滤不完整、关系类型归一化、去重"""

        # 停用词，用来过滤掉不相关的三元组
        STOP_WORDS = {"的", "了", "是", "在", "和", "与", "或", "等", "可以", "需要", "应该"}

        # 实体同义词，把意思相近的实体映射为统一名称
        SYNONYMS = {
            "带薪年休假": "年休假",
            "带薪年假": "年休假",
            "年假": "年休假",
            "迟到早退": "迟到",
            "伪造材料": "伪造",
            "虚假发票": "假发票",
        }

        # 关系类型归一化：LLM 若输出中文谓词，回退映射为 UPPER_SNAKE
        REL_MAP = {
            "适用于": "APPLIES_TO", "适用": "APPLIES_TO",
            "依据": "GOVERNED_BY", "约束": "GOVERNED_BY", "受约束": "GOVERNED_BY",
            "审批": "REQUIRES_APPROVAL", "需审批": "REQUIRES_APPROVAL", "经审批": "REQUIRES_APPROVAL",
            "办理": "HANDLED_BY", "负责": "HANDLED_BY",
            "阶梯": "TIERED_BY", "分级": "TIERED_BY", "分档": "TIERED_BY",
            "触发": "TRIGGERS", "导致": "TRIGGERS",
            "限制": "RESTRICTS", "不得": "RESTRICTS",
            "排除": "EXCLUDES", "不适用": "EXCLUDES",
            "计算": "CALCULATES_WITH",
            "归入": "FEEDS_INTO", "计入": "FEEDS_INTO",
            "引用": "CROSS_REF", "参见": "CROSS_REF",
            "归档": "ARCHIVED_BY",
        }

        def _sanitize_predicate(p: str) -> str:
            """中文->UPPER_SNAKE 回退，并清洗为合法关系类型字符"""
            p = REL_MAP.get(p, p)
            # 只保留大写字母/数字/下划线
            p = re.sub(r"[^A-Z0-9_]", "_", p.upper())
            p = re.sub(r"_+", "_", p).strip("_")
            return p

        cleaned, seen = [], set()
        for item in triples:
            # 去除三元组元素中的空格
            s = str(item.get("subject", "")).strip()
            p = str(item.get("predicate", "")).strip()
            o = str(item.get("object", "")).strip()
            c = str(item.get("constraint", "")).strip()

            # 判断三元组是否完整, 不完整则过滤掉
            if not all([s, p, o]) or len(s) < 2 or len(o) < 2:
                continue

            # 判断实体中是否存在停用词, 如果存在则过滤掉
            if s in STOP_WORDS or o in STOP_WORDS:
                continue

            # 把三元组实体中意思相近的标签名映射为统一的标签名
            s, o = SYNONYMS.get(s, s), SYNONYMS.get(o, o)

            # 关系类型归一化 + 清洗；清洗后为空则丢弃
            p = _sanitize_predicate(p)
            if not p:
                continue

            # 去重, 如果在列表存在则过滤,不存在则添加到结果列表中
            key = (s, p, o, c)
            if key not in seen:
                seen.add(key)
                cleaned.append({"subject": s, "predicate": p, "object": o, "constraint": c})

        logger.info(f"三元组清洗完成，保留 {len(cleaned)} 条")
        return cleaned

    # 保存三元组到neo4j
    def add_triples(self, triples: List[Dict]) -> Dict:
        try:
            success_count = 0
            for item in triples:
                s = item.get("subject", "")
                p = item.get("predicate", "")
                o = item.get("object", "")
                c = item.get("constraint", "")
                # 只在有条件时才写 constraint 属性，保持图谱干净
                props = {"constraint": c} if c else {}

                self.graph.query(
                    f"""
                    MERGE (s:ENTITY {{name: $s_name}})
                    MERGE (o:ENTITY {{name: $o_name}})
                    MERGE (s) - [r:`{p}`] -> (o)
                    SET r += $props
                    """,
                    params={
                        "s_name": s,
                        "o_name": o,
                        "props": props,
                    }
                )
                success_count += 1

            logger.info(f"neo4j 写入完成，成功 {success_count}/{len(triples)} 条")
            return {
                "status": "success",
                "success_count": success_count,
                "total_count": len(triples)
            }
        except Exception as e:
            logger.error(f"三元组保存失败: {e}")
            return {
                "status": "error",
                "msg": f"三元组保存失败: {e}"
            }

    # 保存三元组逻辑整合（对齐 rag_index.py 的调用：build() + get_graph()）
    def build(self):
        # 删除开关为 True，则清空 neo4j 的所有数据
        if self.delete_flag:
            self.graph.query("MATCH (n) DETACH DELETE n")
            logger.info("已清空 neo4j 全部数据")

        results = self.extract_triples()
        cleaned_triples = self.clean_triples(results)
        add_result = self.add_triples(cleaned_triples)

        logger.info("neo4j 三元组创建完成")
        return add_result

