"""对话历史的 FastAPI 路由：侧边栏的历史会话列表与详情。

对外提供三个接口（数据存在 familyagent/history/，一个会话一个 JSON 文件，不进版本库）：
- GET    /agent/history/          历史会话摘要列表（按最近更新倒序）
- GET    /agent/history/{id}      单个会话的完整消息
- DELETE /agent/history/{id}      删除会话

写入不在这里：问答过程中由 dispatch_route 落盘，本模块只负责读取与删除。
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from familyagent.func import history_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent/history", tags=["对话历史"])


@router.get("/", summary="历史会话列表")
async def list_sessions() -> dict:
    """返回全部会话的摘要，供侧边栏渲染。"""
    sessions = await asyncio.to_thread(history_store.list_sessions)
    return {"sessions": sessions}


@router.get("/{session_id}", summary="会话详情")
async def get_session(session_id: str) -> dict:
    """返回单个会话的完整消息列表，用于继续上一次对话。"""
    session = await asyncio.to_thread(history_store.get_session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


@router.delete("/{session_id}", summary="删除会话")
async def delete_session(session_id: str) -> dict:
    """删除单个会话；不存在或 id 非法返回 404。"""
    deleted = await asyncio.to_thread(history_store.delete_session, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True}
