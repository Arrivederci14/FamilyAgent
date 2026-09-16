"""智能体分发路由：先用大模型识别意图，再把问题转发给对应的智能体。

对外提供两个接口（智能体分发助手挂在 /agent 下，是全站唯一的前端入口）：
- GET  /agent/        前端页面，向模板 index.html 注入页面文案与各分支运行状态
- POST /agent/stream/ 识别问题意图（旅游规划 / 健康问答 / 闲聊），流式返回对应智能体的回答

一次请求的完整流程：
1. 取会话（前端带 session_id 就复用，否则新建，通过 X-Session-Id 回传）；
2. 并发做两件独立的事：识别意图 + 挑出本次相关的家庭档案；
3. 把会话历史与档案拼进提问，转发给对应智能体，流式返回；
4. 回答开头写明参考了哪几位家人，并把这轮问答落盘到 familyagent/history/。
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Iterator
from urllib.parse import quote

from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

from familyagent.func import family_profile, history_store
from familyagent.func.agent_health import AI_DISCLAIMER
from familyagent.func.agent_health import SERVICE_UNAVAILABLE as HEALTH_UNAVAILABLE
from familyagent.func.agent_health import health_agent, stream_health_answer
from familyagent.func.agent_travel import SERVICE_UNAVAILABLE as TRAVEL_UNAVAILABLE
from familyagent.func.agent_travel import stream_travel_plan, travel_agent
from familyagent.func.intent_recognition import HEALTH, TRAVEL, recognize_intent
from familyagent.func.llm_config import llm
from familyagent.route import AGENT_STATUS_ERROR, AGENT_STATUS_OK

logger = logging.getLogger(__name__)

# 项目根目录：route/dispatch_route.py -> route/ -> familyagent/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

router = APIRouter(prefix="/agent", tags=["智能体分发"])

# 页面文案，改文案不用动模板；三个分支的可用性由路由实时算出来再注入
PAGE = {
    "title": "家庭智能体分发助手",
    "hints": [
        "一个问题自动找到合适的智能体：旅游规划、健康问答，其余交给通用大模型",
        "例如：帮我规划一个北京 3 天家庭游 / 小孩发烧怎么办 / 陪我聊聊天",
    ],
    "placeholder": "请输入问题，Enter 发送，Shift + Enter 换行",
    "dispatch_url": "/agent/stream/",
    "offline_tip": "意图识别所需的大模型不可用，请检查后端日志与模型配置后刷新页面。",
    "note": "回答由大模型自动分发与生成，仅供参考；健康问题请咨询专业医生。",
    # 健康智能体的回答自带免责标注，前端会把它拆成醒目提示条
    "disclaimer": AI_DISCLAIMER,
}

# 闲聊分支不走智能体，自己兜底一条通用文案
CHITCHAT_UNAVAILABLE = "服务暂时不可用，请稍后再试。"

# 闲聊分支的人设：只闲聊和答通用问题，不碰需要专业智能体的活儿
CHITCHAT_SYSTEM_PROMPT = """你是家庭助手，负责和用户闲聊、解答与家庭生活相关的通用问题。

回答要求：
1. 语气轻松自然，回答简洁，几句话讲完，不要长篇大论。
2. 不确定的内容直接说明，不要编造事实。
3. 涉及疾病、用药、诊疗的问题，提醒用户咨询专业医生，不给出诊疗结论。
4. 涉及具体出行安排时，提示用户可以让我帮忙规划行程。"""

# 分支内部异常时回给用户的最后一句话，按意图取对应文案
UNAVAILABLE = {
    TRAVEL: TRAVEL_UNAVAILABLE,
    HEALTH: HEALTH_UNAVAILABLE,
}

# 同步生成器迭代结束的哨兵值（不能用 None，正文片段可能是空串以外的任意值）
_SENTINEL = object()

# 意图识别只看「在聊什么」，历史每条截断，避免把整篇行程塞进分类提示词
CONTEXT_ITEMS = 4
CONTEXT_ITEM_LEN = 120


class Question(BaseModel):
    """智能体分发请求体。"""

    question: str
    # 会话 id：前端首次提问不传，后端建好后通过 X-Session-Id 返回，之后每轮都带上
    session_id: str | None = None


def _chunk_text(message: object) -> str:
    """取出消息正文，兼容 content 为字符串或分块列表两种形式。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


async def _as_async(source: Iterator[str]) -> AsyncIterator[str]:
    """把同步生成器逐段搬到线程里取，避免长时间阻塞事件循环。

    健康问答智能体是同步生成器，直接在协程里迭代会把事件循环占满，
    导致其他请求（含流式输出）全部排队，所以这里每次取下一段都放到线程里执行。
    """
    iterator = iter(source)
    while True:
        piece = await asyncio.to_thread(next, iterator, _SENTINEL)
        if piece is _SENTINEL:
            return
        yield piece  # type: ignore[misc]


async def _stream_chitchat(
    question: str,
    profile_context: str = "",
    history: list[dict] | None = None,
) -> AsyncIterator[str]:
    """闲聊分支：直接用大模型实例流式作答，不经过智能体与工具。"""
    if llm is None:
        logger.error("闲聊分支不可用：LLM 未初始化")
        yield CHITCHAT_UNAVAILABLE
        return

    messages = [
        {"role": "system", "content": CHITCHAT_SYSTEM_PROMPT},
        *family_profile.build_agent_messages(question, profile_context, history),
    ]
    async for chunk in llm.astream(messages):
        # 只透传有正文的片段，忽略模型返回的其他内容
        text = _chunk_text(chunk)
        if text:
            yield text


def _intent_context(history: list[dict]) -> str:
    """把最近的对话压成几行摘要，供意图识别判断追问的归属。"""
    lines = []
    for message in history[-CONTEXT_ITEMS:]:
        speaker = "用户" if message.get("role") == "user" else "助手"
        text = " ".join(str(message.get("content", "")).split())
        if text:
            lines.append(f"{speaker}：{text[:CONTEXT_ITEM_LEN]}")
    return "\n".join(lines)


async def _detect_intent(question: str, history: list[dict]) -> str | None:
    """第一步：识别意图，空问题不上模型、直接返回 None。

    识别是阻塞的网络调用，放到线程里跑，避免卡住事件循环。
    recognize_intent 内部已做兜底，识别失败会返回「闲聊」，这里不用再判空。
    """
    if not question:
        return None

    intent = await asyncio.to_thread(recognize_intent, question, _intent_context(history))
    logger.info("智能体分发：intent=%s, question=%s", intent, question)
    return intent


def _inherit_profiles(session_id: str) -> list[dict]:
    """当轮没选到档案时，沿用上一轮回答参考过的成员（按最新档案内容还原）。"""
    names = history_store.last_profile_names(session_id)
    if not names:
        return []

    inherited = [p for p in family_profile.list_profiles() if p["name"] in names]
    if inherited:
        logger.info("家庭档案沿用上一轮：%s", [p["name"] for p in inherited])
    return inherited


async def _stream_reply(
    question: str,
    intent: str | None,
    profile_context: str,
    profile_note: str,
    profile_names: list[str],
    history: list[dict],
    session_id: str,
) -> AsyncIterator[str]:
    """第二步：按意图分发并流式返回回答，分支异常统一收口成兜底文案。

    intent 由接口层识别后传入；为空只可能出现在空问题时，这里会提前返回提示语。
    整个回答（含开头的档案说明）会在结束时落盘到会话历史。
    """
    if not question:
        yield "请输入问题内容。"
        return

    # 三个标签严格匹配，其余（含「闲聊」）走通用大模型
    stream: AsyncIterator[str]
    if intent == TRAVEL:
        stream = stream_travel_plan(question, profile_context, history)
    elif intent == HEALTH:
        stream = _as_async(stream_health_answer(question, profile_context, history))
    else:
        stream = _stream_chitchat(question, profile_context, history)

    # 边流边攒，结束时整体写入历史；中途断开也能把已生成的部分存下来
    collected: list[str] = []
    try:
        if profile_note:
            # 回答开头就说明这次参考了谁，页面和直接调接口的调用方都能看到
            note = f"{profile_note}\n\n"
            collected.append(note)
            yield note

        async for piece in stream:
            collected.append(piece)
            yield piece
    except Exception:
        logger.exception("智能体分发失败：intent=%s, question=%s", intent, question)
        fallback = UNAVAILABLE.get(intent, CHITCHAT_UNAVAILABLE)
        collected.append(fallback)
        yield fallback
    finally:
        await asyncio.to_thread(
            history_store.append_message,
            session_id,
            history_store.ROLE_AI,
            "".join(collected),
            intent=intent or "",
            profiles=profile_names,
        )


def _branch_status() -> tuple[bool, str]:
    """汇总三个分支的可用性，返回（整体是否可用，前端展示的状态文案）。"""
    branches = [
        ("旅游规划", travel_agent is not None),
        ("健康问答", health_agent is not None),
        ("闲聊", llm is not None),
    ]
    detail = " / ".join(f"{name}{AGENT_STATUS_OK if ok else AGENT_STATUS_ERROR}" for name, ok in branches)
    # 只要大模型在，意图识别和闲聊就能跑；单个智能体不可用时由它自己返回提示语
    return llm is not None, detail


@router.get("/", summary="智能体分发前端页面")
async def dispatch_page(request: Request) -> Response:
    """全站唯一前端页面：渲染页面文案与各分支运行状态。"""
    ready, detail = _branch_status()
    context = {
        "page": PAGE,
        "agent_ready": ready,
        "agent_status": AGENT_STATUS_OK if ready else AGENT_STATUS_ERROR,
        "branch_status": detail,
    }
    return templates.TemplateResponse(request, "index.html", context)


@router.post("/stream/", summary="智能体分发接口（基于大模型意图识别）")
async def dispatch(payload: Question) -> StreamingResponse:
    """识别问题意图，转发给对应智能体，以流式方式返回回答。

    旅游规划 → 旅游规划智能体；健康问答 → 健康问答智能体；
    其他 / 闲聊 → 通用大模型直接作答。
    返回头 X-Session-Id 为本轮会话 id，X-Intent 为命中的分支。
    """
    question = payload.question.strip()

    # 会话：前端没带 id（或 id 已失效）时新建；标题取首条提问
    session = await asyncio.to_thread(
        history_store.get_or_create_session,
        payload.session_id,
        question[: history_store.MAX_TITLE_LEN] or "新对话",
    )
    session_id = session["id"]
    history = await asyncio.to_thread(history_store.build_context, session_id)
    if question:
        await asyncio.to_thread(history_store.append_message, session_id, history_store.ROLE_USER, question)

    # 意图识别与档案选择互不依赖，并发跑，首字延迟只等于其中较慢的一个
    intent, profiles = await asyncio.gather(
        _detect_intent(question, history),
        asyncio.to_thread(family_profile.select_profiles, question),
    )
    if not profiles and intent in (TRAVEL, HEALTH):
        # 追问句通常不带称呼（「那她能吃海鲜吗」），沿用上一轮参考过的档案，
        # 避免多轮对话里「在说谁」中途丢失；闲聊（如「谢谢」）不沿用，
        # 否则一句寒暄也会被标成「参考了家庭档案：奶奶」
        profiles = await asyncio.to_thread(_inherit_profiles, session_id)

    headers = {
        "Cache-Control": "no-cache",
        "X-Session-Id": session_id,
    }
    if intent:
        # HTTP 头只允许 latin-1，中文标签必须百分号编码后再下发
        headers["X-Intent"] = quote(intent)

    return StreamingResponse(
        _stream_reply(
            question,
            intent,
            family_profile.format_profile_context(profiles),
            family_profile.format_profile_note(profiles),
            [profile["name"] for profile in profiles],
            history,
            session_id,
        ),
        media_type="text/plain; charset=utf-8",
        headers=headers,
    )
