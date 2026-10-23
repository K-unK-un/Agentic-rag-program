"""
Agent RAG 聊天服务。

与classic版本差异：
- 不再是"固定检索管线+LLM回答"，而是由Agent自主编排：策略分析→(拆解)→多路并行
  检索→合并→统一rerank→生成回答
- SSE输出为结构化事件流：strategy / todos / tool_start / tool_end / answer / done
- 每个请求创建独立的DocStore与Agent实例（方案A：doc_id+服务端Store）
"""
import json

from langchain_core.messages import AIMessageChunk, ToolMessage

from agent.agent import create_retriever_agent
from tools.doc_store import DocStore
from utils.logger import setup_logger

logger = setup_logger(__name__)


def sse_event(event: str, data) -> str:
    """构造SSE格式事件帧"""
    if not isinstance(data, str):
        data = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


class AgentChatService:

    # ==================== 非流式调用 ====================
    async def invoke(self, query: str) -> dict:
        doc_store = DocStore()
        agent = create_retriever_agent(doc_store)

        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": query}]
        })

        todos = result.get("todos", [])
        answer = result["messages"][-1].content
        references = self._collect_references(doc_store)

        logger.info(f"Agent执行完成: todos={len(todos)}, 引用文档={len(references)}")

        return {
            "answer": answer,
            "todos": todos,
            "references": references
        }

    # ==================== SSE流式调用 ====================
    async def generator(self, query: str):
        """SSE事件流：
        strategy → todos → tool_start/tool_end（可并行多路） → answer(token流) → done
        """
        doc_store = DocStore()
        agent = create_retriever_agent(doc_store)

        try:
            # 已看到工具结果消息后，后续AI文本视为最终答案token，实时下发；
            # 此之前的AI文本属于工具调用前的过渡语，不下发
            seen_tool_result = False
            latest_todos = []

            # stream_mode="messages" 单模式：只接收消息流，避免 LangGraph 1.x 多模式解包兼容问题
            async for chunk in agent.astream(
                    {"messages": [{"role": "user", "content": query}]},
                    stream_mode="messages"
            ):
                # messages 模式下 chunk 是 (message, metadata) 2元组
                msg = chunk[0] if isinstance(chunk, tuple) else chunk

                # ToolMessage 表示工具执行完成，之后产生的 AIMessage 即为最终答案
                if isinstance(msg, ToolMessage):
                    seen_tool_result = True
                    # 推送工具完成事件，前端工具chip变绿
                    yield sse_event("tool_end", {"name": msg.name})

                elif isinstance(msg, AIMessageChunk):
                    content = msg.content if isinstance(msg.content, str) else ""
                    if seen_tool_result and content:
                        # 工具执行后产生的 AI 文本，作为答案 token 流式下发
                        yield sse_event("answer", content)

            # 结束事件：附带todos最终状态与引用文档
            yield sse_event("done", {
                "todos": [
                    {"content": t.get("content"), "status": t.get("status")}
                    for t in latest_todos
                ],
                "references": self._collect_references(doc_store)
            })
        except Exception as e:
            import traceback
            logger.error(f"Agent执行异常: {e}\n{traceback.format_exc()}")
            yield sse_event("error", {"message": f"服务执行异常: {str(e)[:200]}"})
            yield sse_event("done", {"todos": [], "references": []})

    # 解析策略分析工具的返回内容
    @staticmethod
    def _parse_strategy(content) -> dict | None:
        try:
            raw = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            return json.loads(raw)
        except Exception:
            return None

    # 收集本请求DocStore中的引用文档（doc_id + 摘要）
    @staticmethod
    def _collect_references(doc_store: DocStore) -> list[dict]:
        references = []
        for doc_id, doc in doc_store.items():
            references.append({
                "doc_id": doc_id,
                "snippet": doc_store.get_snippet(doc),
                "source": doc.metadata.get("source", "")
            })
        return references


