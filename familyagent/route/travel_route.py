"""家庭旅游规划智能体的 FastAPI 路由。

对外提供一个接口：
- POST /agent/travel/stream/ 流式生成 3 天家庭旅游路线，逐段返回智能体生成的内容

前端页面统一由智能体分发助手（GET /agent/）承载，自动把旅游需求路由到这里。
"""

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from familyagent.func.agent_travel import SERVICE_UNAVAILABLE, stream_travel_plan

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent/travel", tags=["家庭旅游规划智能体"])


class TravelQuery(BaseModel):
    """旅游规划请求体。"""

    query: str


async def _stream_plan(question: str) -> AsyncIterator[str]:
    """包装智能体流式输出：中途异常只记日志并正常收尾，避免响应半路断裂。"""
    question = question.strip()
    if not question:
        yield "请输入旅游需求，例如：帮我规划一个北京 3 天家庭游，同行有一位老人和一个小孩。"
        return

    try:
        async for piece in stream_travel_plan(question):
            yield piece
    except Exception:
        logger.exception("旅游规划流式输出失败：question=%s", question)
        yield SERVICE_UNAVAILABLE


@router.post("/stream/", summary="家庭旅游规划流式输出")
async def travel_stream(payload: TravelQuery) -> StreamingResponse:
    """以流式方式返回 3 天家庭旅游路线，前端可边接收边渲染。"""
    return StreamingResponse(
        _stream_plan(payload.query),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache"},
    )
