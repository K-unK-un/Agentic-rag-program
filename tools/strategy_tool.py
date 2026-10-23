'''
意图识别工具：根据用户的问题，通过大模型推理，判断应该调用哪些工具执行检索
'''
import hashlib
from typing import Annotated

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel

from agent.prompts import ANALYZE_PROMPT
from models.llm import strategy_llm
from service.redis_service import RedisService
from utils.logger import setup_logger

logger = setup_logger(__name__)
class StrategyAnalyzer(BaseModel):
    retrieval_types: list[str] = ['vector']
    need_decomposition: bool = False
    sub_questions: list[str]
    reasoning: str

@tool
def strategy_tool(query: Annotated[str, "用户的问题"]):
    '''
    分析用户问题的检索策略，输出结构化策略（不检索文档）。
    返回字段：retrieval_types(应使用的检索方式列表) / need_decomposition(是否需拆子问题) / sub_questions(子问题列表) / reasoning(判断依据)。
    必须在调用任何检索工具(bm25_search/vector_search/hybrid_search/neo4j_search)之前先调用本工具，据此结果选择后续检索工具。
    '''
    # 引入redis缓存机制，如果用户的问题在缓存中能命中则直接返回缓存中保存的识别结果
    # 如果未命中，则使用大模型进行意图识别，并将识别结果保存到redis缓存中
    redis_service = RedisService()

    # 根据用户问题生成redis缓存的key
    cache_key = hashlib.md5(query.strip().lower().encode('utf-8')).hexdigest()
    # 从缓存中获取key
    cache_result = redis_service.get_strategy(cache_key)

    if cache_result is not None:
        logger.info(f"从缓存中获取策略识别结果：{cache_result}")
        return cache_result

    logger.info(f"未找到缓存")
    # 大模型进行意图识别
    prompt_template = ChatPromptTemplate.from_template(ANALYZE_PROMPT)

    chain = prompt_template | strategy_llm.with_structured_output(StrategyAnalyzer)
    # 得到对象数据类型：StrategyAnalyzer
    result = chain.invoke({"query": query})

    if isinstance(result, StrategyAnalyzer):
        result = result.model_dump()

    # 将大模型识别的结果存入redis缓存
    redis_service.set_strategy(cache_key, result)

    return result
