"""对话历史的本地持久化：一个会话一个 JSON 文件，存在 familyagent/history/ 下。

目录结构（已加入 .gitignore，不进版本库）：
    familyagent/history/<session_id>.json

会话文件结构：
    {"id", "title", "created_at", "updated_at",
     "messages": [{"role", "content", "ts", "intent", "profiles"}]}

前端侧边栏直接读 list_sessions() 的结果渲染历史列表，
继续对话时用 build_context() 把最近若干轮拼成模型可读的消息列表。
"""

import json
import logging
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from familyagent.config import config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = PROJECT_ROOT / config.history_dir

# 会话 id 来自前端，必须校验，否则 "../../某文件" 会写到目录外
SESSION_ID_PATTERN = re.compile(r"^s_[0-9a-f]{32}$")

# 侧边栏标题取首条提问的前若干字
MAX_TITLE_LEN = 24
# 送给模型的历史消息条数上限，避免上下文随对话无限增长
MAX_CONTEXT_MESSAGES = 12
# 模型侧的角色名，与 LangChain 的消息 role 对齐
ROLE_USER = "user"
ROLE_AI = "ai"


def _now() -> str:
    """本地时间戳，精确到秒，便于前端展示与排序。"""
    return datetime.now().isoformat(timespec="seconds")


def _session_path(session_id: str) -> Path:
    return HISTORY_DIR / f"{session_id}.json"


def is_valid_session_id(session_id: object) -> bool:
    """会话 id 必须是 s_ + 32 位十六进制，挡住路径穿越与非法文件名。"""
    return isinstance(session_id, str) and bool(SESSION_ID_PATTERN.match(session_id))


def new_session_id() -> str:
    return f"s_{uuid.uuid4().hex}"


def _read(session_id: str) -> dict[str, Any] | None:
    """读取会话文件，文件损坏或不存在时返回 None（而不是抛异常）。"""
    path = _session_path(session_id)
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        logger.exception("会话文件读取失败：%s", path)
        return None
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        logger.warning("会话文件结构异常，已忽略：%s", path)
        return None
    return data


def _write(session: dict[str, Any]) -> None:
    """原子写：先写临时文件再替换，避免写一半被读到。"""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = _session_path(session["id"])
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# 同一进程内的并发写保护：流式接口在线程里落盘，没有锁会出现写覆盖
_write_lock = threading.Lock()


def create_session(title: str = "新对话") -> dict[str, Any]:
    """新建会话并落盘，返回会话字典。"""
    now = _now()
    session = {
        "id": new_session_id(),
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    with _write_lock:
        _write(session)
    logger.info("新建会话：id=%s, title=%s", session["id"], title)
    return session


def get_session(session_id: str) -> dict[str, Any] | None:
    """按 id 取会话，id 非法时直接返回 None。"""
    if not is_valid_session_id(session_id):
        return None
    return _read(session_id)


def get_or_create_session(session_id: str | None, title: str = "新对话") -> dict[str, Any]:
    """存在就复用，不存在或 id 非法就新建；前端首次提问不带 session_id。"""
    session = get_session(session_id) if session_id else None
    if session is not None:
        return session
    return create_session(title)


def list_sessions() -> list[dict[str, Any]]:
    """列出全部会话的摘要，按最近更新倒序；读不动的文件跳过不影响其他会话。"""
    if not HISTORY_DIR.exists():
        return []

    summaries = []
    for path in HISTORY_DIR.glob("s_*.json"):
        session = _read(path.stem)
        if session is None:
            continue
        messages = session["messages"]
        summaries.append(
            {
                "id": session["id"],
                "title": session.get("title") or "新对话",
                "created_at": session.get("created_at", ""),
                "updated_at": session.get("updated_at", ""),
                # 侧边栏展示：最后一条提问的一句话摘要
                "preview": next(
                    (m["content"] for m in reversed(messages) if m.get("role") == ROLE_USER),
                    "",
                )[:40],
                "count": len(messages),
            }
        )

    summaries.sort(key=lambda item: item["updated_at"], reverse=True)
    return summaries


def append_message(
    session_id: str,
    role: str,
    content: str,
    **extra: Any,
) -> dict[str, Any] | None:
    """追加一条消息并落盘，返回更新后的会话；id 非法或会话不存在返回 None。

    额外字段（intent、profiles 等）原样存进消息里，供前端渲染标签。
    首条用户提问会顺便作为会话标题。
    """
    if not is_valid_session_id(session_id):
        logger.warning("非法的会话 id，已忽略写入：%s", session_id)
        return None

    with _write_lock:
        session = _read(session_id)
        if session is None:
            logger.warning("会话不存在，无法追加消息：%s", session_id)
            return None

        session["messages"].append(
            {"role": role, "content": content, "ts": _now(), **extra}
        )
        # 首条提问定为标题，后续消息不再覆盖
        if role == ROLE_USER and session.get("title", "新对话") == "新对话":
            session["title"] = content.strip()[:MAX_TITLE_LEN] or "新对话"
        session["updated_at"] = _now()
        _write(session)
        return session


def last_profile_names(session_id: str) -> list[str]:
    """取本会话上一次回答参考过的家人称呼，供追问时沿用。

    追问句往往不带称呼（「那她能吃海鲜吗」），光靠当轮提问选不出档案，
    沿用上一轮选中的人，多轮对话里「在说谁」才不会漂移。
    """
    session = get_session(session_id)
    if session is None:
        return []

    for message in reversed(session["messages"]):
        if message.get("role") != ROLE_AI:
            continue
        # 只看最近一条回答：它为空就是上一轮本来就没参考档案，不再往前翻
        return [str(name) for name in message.get("profiles") or []]
    return []


def delete_session(session_id: str) -> bool:
    """删除会话文件，返回是否删除成功。"""
    if not is_valid_session_id(session_id):
        return False
    path = _session_path(session_id)
    if not path.exists():
        return False
    try:
        path.unlink()
    except OSError:
        logger.exception("会话删除失败：%s", path)
        return False
    logger.info("已删除会话：%s", session_id)
    return True


def build_context(session_id: str | None, limit: int = MAX_CONTEXT_MESSAGES) -> list[dict]:
    """取最近 limit 条消息，转成模型可读的 [{"role", "content"}]，按时间正序。

    role 用 LangChain 约定的 user / assistant；带档案前缀的历史回答原样保留，
    模型能看到上一轮参考了哪些家人档案，追问时更连贯。
    """
    if not session_id:
        return []

    session = get_session(session_id)
    if session is None:
        return []

    recent = session["messages"][-limit:] if limit > 0 else session["messages"]
    return [
        {
            "role": ROLE_USER if message.get("role") == ROLE_USER else "assistant",
            "content": message.get("content", ""),
        }
        for message in recent
        if message.get("content")
    ]
