"""
Agent RAG 聊天接口：
1. GET  /api/chat/ask   SSE流式问答（JWT认证）
2. POST /api/chat/invoke 非流式问答，一次性返回答案、todo列表与引用文档
"""
from fastapi import APIRouter, Query

from starlette.responses import StreamingResponse


from service.chat_service import AgentChatService


chat_router = APIRouter(prefix="/chat")


# SSE流式问答
@chat_router.get("/ask")
async def ask(query: str = Query(...)):
    chat_service = AgentChatService()
    return StreamingResponse(
        chat_service.generator(query),
        media_type="text/event-stream"
    )



