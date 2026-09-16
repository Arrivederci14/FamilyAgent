"""家庭健康问答智能体的 FastAPI 路由。

对外提供一个接口：
- POST /agent/healthy/stream/ 流式问答，逐段返回智能体生成的内容

前端页面统一由智能体分发助手（GET /agent/）承载，自动把健康问题路由到这里。
"""

import logging
from collections.abc import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from familyagent.func.agent_health import SERVICE_UNAVAILABLE, stream_health_answer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent/healthy", tags=["家庭健康问答智能体"])


class HealthQuery(BaseModel):
    """健康问答请求体。"""

    query: str


def _stream_answer(question: str) -> Iterator[str]:
    """包装智能体流式输出：中途异常只记日志并正常收尾，避免响应半路断裂。"""
    question = question.strip()
    if not question:
        yield "请输入要咨询的健康问题。"
        return

    try:
        yield from stream_health_answer(question)
    except Exception:
        logger.exception("健康问答流式输出失败：question=%s", question)
        yield SERVICE_UNAVAILABLE


@router.post("/stream/", summary="健康问答流式输出")
async def health_stream(payload: HealthQuery) -> StreamingResponse:
    """以流式方式返回健康问答结果，前端可边接收边渲染。"""
    return StreamingResponse(
        _stream_answer(payload.query),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache"},
    )
