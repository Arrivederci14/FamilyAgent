"""家庭档案的 FastAPI 路由：家人的增删改查。

对外提供四个接口（数据存在 familyagent/familydata/profiles.json，不进版本库）：
- GET    /agent/family/            档案列表 + 表单字段定义
- POST   /agent/family/            新增家人
- PUT    /agent/family/{id}        编辑家人
- DELETE /agent/family/{id}        删除家人

档案内容不是在这里用的：问答时由 dispatch_route 调用 family_profile.select_profiles
挑出相关成员并注入智能体提示词，本模块只负责维护数据。
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from familyagent.func import family_profile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent/family", tags=["家庭档案"])


class ProfileIn(BaseModel):
    """家人档案请求体；除姓名外都可留空。

    age 允许 int / 字符串 / 空：前端表单清空后传的是空串，交给 family_profile 统一校验。
    """

    name: str
    relation: str = ""
    gender: str = ""
    age: int | str | None = None
    conditions: str = ""
    allergies: str = ""
    medications: str = ""
    diet: str = ""
    notes: str = ""


def _fields_meta() -> list[dict[str, str]]:
    """把字段定义下发给前端，表单与后端校验共用一份，避免两边写两遍。"""
    return [{"key": key, "label": label} for key, label in family_profile.FIELD_LABELS.items()]


@router.get("/", summary="家庭档案列表")
async def list_profiles() -> dict:
    """返回全部家人档案，以及表单需要的字段定义。"""
    profiles = await asyncio.to_thread(family_profile.list_profiles)
    return {"fields": _fields_meta(), "profiles": profiles}


@router.post("/", summary="新增家人档案")
async def create_profile(payload: ProfileIn) -> dict:
    """新增一位家人；校验不通过返回 400 并带上具体原因。"""
    try:
        profile = await asyncio.to_thread(
            family_profile.create_profile, payload.model_dump()
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"profile": profile}


@router.put("/{profile_id}", summary="编辑家人档案")
async def update_profile(profile_id: str, payload: ProfileIn) -> dict:
    """编辑指定家人；档案不存在返回 404，校验不通过返回 400。"""
    try:
        profile = await asyncio.to_thread(
            family_profile.update_profile, profile_id, payload.model_dump()
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if profile is None:
        raise HTTPException(status_code=404, detail="档案不存在")
    return {"profile": profile}


@router.delete("/{profile_id}", summary="删除家人档案")
async def delete_profile(profile_id: str) -> dict:
    """删除指定家人；档案不存在返回 404。"""
    deleted = await asyncio.to_thread(family_profile.delete_profile, profile_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="档案不存在")
    return {"ok": True}
