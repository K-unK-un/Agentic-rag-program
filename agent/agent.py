"""
检索路由 Agent

基于 langchain 1.x 的 create_agent（LangGraph 风格）。
agent 依据用户问题自主决定调用哪个检索器工具（BM25 / 向量 / 图谱 / 混合），
检索后可调用 rerank 工具精排，最后给出有依据的回答。
"""
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

from models.llm import qwen_llm
from agent.prompts import AGENT_SYSTEM_PROMPT
from tools.doc_store import DocStore
from tools.reranker_tool import create_rerank_tool
from tools.retriever_tools import build_retriever_tools
from tools.strategy_tool import strategy_tool
from utils.logger import setup_logger

logger = setup_logger(__name__)


def create_retriever_agent(doc_store: DocStore):
    # 策略分析工具（1个，前置）+ 检索工具（4个）+ 重排工具（1个）
    tools = [strategy_tool] + build_retriever_tools(doc_store) + [create_rerank_tool(doc_store)]
    agent = create_agent(
        model=qwen_llm,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
        middleware=[TodoListMiddleware()]
    )
    logger.info(f"检索路由 Agent 构建完成，挂载 {len(tools)} 个工具")
    return agent



