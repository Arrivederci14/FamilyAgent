"""家庭档案：保存家人的基本信息，并按提问挑出相关成员注入到智能体提示词中。

数据落在 familyagent/familydata/profiles.json（已加入 .gitignore，含家庭隐私，不进版本库）。

一次问答的完整链路：
1. 路由调用 select_profiles(question) 挑出本次相关的成员；
2. format_profile_context() 把选中的成员拼成【家庭成员档案】文本块；
3. 智能体把该文本块拼在用户问题前面（系统提示词保持固定，便于复用）；
4. format_profile_note() 生成一行说明放在回答开头，让用户知道答案参考了谁。
"""

import json
import logging
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from familyagent.config import config
from familyagent.func.llm_config import llm

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAMILY_DIR = PROJECT_ROOT / config.family_dir
PROFILE_FILE = FAMILY_DIR / "profiles.json"

# 档案 id 由后端生成，校验一次挡住非法文件名
PROFILE_ID_PATTERN = re.compile(r"^p_[0-9a-f]{32}$")

# 档案字段与展示名：前端表单、提示词拼装、校验共用这一份定义
FIELD_LABELS: dict[str, str] = {
    "name": "姓名/称呼",
    "relation": "与家庭的关系",
    "gender": "性别",
    "age": "年龄",
    "conditions": "慢病/既往史",
    "allergies": "过敏史",
    "medications": "长期用药",
    "diet": "饮食忌口",
    "notes": "其他说明（行动能力、作息等）",
}
FIELDS = tuple(FIELD_LABELS)
MAX_FIELD_LEN = 200

# 写在回答开头的说明，回答里出现它，用户就知道这次参考了哪几位家人
PROFILE_NOTE_TEMPLATE = "> 📋 本次回答参考了家庭档案：{names}"

# 模型回复里可能带上的引号、标点
_STRIP_CHARS = "「」『』【】\"'“”‘’。，、.：: \n\r\t"

SELECT_SYSTEM_PROMPT = """你是家庭档案检索助手。下面会给出家庭成员名单和用户提问，
你只需判断回答这个问题需要参考哪几位成员的信息，不需要回答问题本身。

判断规则：
- 提问里明确提到某位成员，或问的是某类人群（老人、小孩、婴儿、孕产妇等）时，选出对应成员。
- 提问涉及某位成员的健康、用药、饮食、出行或作息时，应该把他/她选上。
- 提问与具体成员无关时（通用知识、目的地推荐、闲聊等），一个都不要选。

输出要求：只输出被选中成员的「姓名/称呼」，多个用「、」分隔；一个都不选就只输出「无」。
不要输出解释、原因、标点说明或其他任何内容。"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _message_text(message: object) -> str:
    """取出模型回复的正文，兼容 content 为字符串或分块列表两种形式。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


def is_valid_profile_id(profile_id: object) -> bool:
    return isinstance(profile_id, str) and bool(PROFILE_ID_PATTERN.match(profile_id))


def _read_all() -> list[dict[str, Any]]:
    """读取全部档案；文件不存在或损坏时返回空列表，不影响主流程。"""
    if not PROFILE_FILE.exists():
        return []
    try:
        with PROFILE_FILE.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        logger.exception("家庭档案读取失败：%s", PROFILE_FILE)
        return []

    profiles = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(profiles, list):
        logger.warning("家庭档案文件结构异常，已忽略：%s", PROFILE_FILE)
        return []
    return [p for p in profiles if isinstance(p, dict) and p.get("id")]


def _write_all(profiles: list[dict[str, Any]]) -> None:
    """原子写：先写临时文件再替换。"""
    FAMILY_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PROFILE_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump({"profiles": profiles}, f, ensure_ascii=False, indent=2)
    tmp.replace(PROFILE_FILE)


# 并发写保护：档案接口可能被多个请求同时打到
_write_lock = threading.Lock()


def _clean_age(value: Any) -> int | None:
    """年龄允许留空；填了就必须是 0~120 的整数。"""
    if value is None or str(value).strip() == "":
        return None
    try:
        age = int(str(value).strip())
    except ValueError as exc:
        raise ValueError("年龄必须是整数") from exc
    if not 0 <= age <= 120:
        raise ValueError("年龄应在 0~120 之间")
    return age


def _clean(data: Any) -> dict[str, Any]:
    """校验并归一化前端传来的字段，非法时抛 ValueError（路由转成 400 返回）。"""
    if not isinstance(data, dict):
        raise ValueError("档案数据格式不正确")

    cleaned: dict[str, Any] = {}
    for key, label in FIELD_LABELS.items():
        if key == "age":
            cleaned[key] = _clean_age(data.get(key))
            continue
        value = data.get(key)
        text = "" if value is None else str(value).strip()
        if len(text) > MAX_FIELD_LEN:
            raise ValueError(f"{label}最多 {MAX_FIELD_LEN} 个字")
        cleaned[key] = text

    if not cleaned["name"]:
        raise ValueError("姓名/称呼不能为空")
    return cleaned


def list_profiles() -> list[dict[str, Any]]:
    return _read_all()


def get_profile(profile_id: str) -> dict[str, Any] | None:
    if not is_valid_profile_id(profile_id):
        return None
    return next((p for p in _read_all() if p["id"] == profile_id), None)


def create_profile(data: Any) -> dict[str, Any]:
    """新增一位家人，返回落库后的档案。"""
    cleaned = _clean(data)
    now = _now()
    profile = {"id": f"p_{uuid.uuid4().hex}", **cleaned, "created_at": now, "updated_at": now}

    with _write_lock:
        profiles = _read_all()
        profiles.append(profile)
        _write_all(profiles)

    logger.info("新增家庭档案：%s（%s）", profile["name"], profile["relation"] or "未填关系")
    return profile


def update_profile(profile_id: str, data: Any) -> dict[str, Any] | None:
    """编辑档案，档案不存在或 id 非法返回 None。"""
    if not is_valid_profile_id(profile_id):
        return None
    cleaned = _clean(data)

    with _write_lock:
        profiles = _read_all()
        target = next((p for p in profiles if p["id"] == profile_id), None)
        if target is None:
            return None
        # id 与创建时间不改写，其余字段整体替换，保证清空的字段真的被清空
        target.update({**cleaned, "updated_at": _now()})
        _write_all(profiles)

    logger.info("更新家庭档案：%s", target["name"])
    return target


def delete_profile(profile_id: str) -> bool:
    if not is_valid_profile_id(profile_id):
        return False

    with _write_lock:
        profiles = _read_all()
        remaining = [p for p in profiles if p["id"] != profile_id]
        if len(remaining) == len(profiles):
            return False
        _write_all(remaining)

    logger.info("删除家庭档案：%s", profile_id)
    return True


def _roster_line(profile: dict[str, Any]) -> str:
    """名单里的一行：称呼 + 关系/性别/年龄 + 健康标签，供模型判断相关性。"""
    parts = [p for p in (profile.get("relation"), profile.get("gender")) if p]
    if profile.get("age") is not None:
        parts.append(f"{profile['age']}岁")
    line = f"- {profile['name']}（{'，'.join(parts)}）" if parts else f"- {profile['name']}"

    tags = [t for t in (profile.get("conditions"), profile.get("allergies")) if t]
    return f"{line}：{'；'.join(tags)}" if tags else line


def _parse_selection(reply: str, profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从模型回复里挑出被点名的成员，保持档案原顺序；回复里没有名字就是没选中。"""
    text = reply.strip()
    if not text or text.strip(_STRIP_CHARS) == "无":
        return []
    return [p for p in profiles if p.get("name") and p["name"] in text]


def match_profiles_by_text(question: str, profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """确定性匹配：提问里直接写了某位家人的称呼就算命中，不用调模型。"""
    return [p for p in profiles if p.get("name") and p["name"] in question]


def select_profiles(question: str, profiles: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """挑出与提问相关的家庭档案：先按称呼直连，没命中再交给模型判断。

    没人在提问里被直接提到时才会调模型，这个调用与意图识别并发执行，
    不额外增加首字延迟；任何异常都退化成「不带档案回答」，不影响主流程。
    """
    profiles = list_profiles() if profiles is None else profiles
    if not profiles or not question.strip():
        return []

    matched = match_profiles_by_text(question, profiles)
    if matched:
        logger.info("家庭档案选择（称呼直连）：%s", [p["name"] for p in matched])
        return matched

    if llm is None:
        logger.warning("家庭档案选择跳过：LLM 未初始化")
        return []

    roster = "\n".join(_roster_line(p) for p in profiles)
    try:
        reply = llm.invoke(
            [
                SystemMessage(content=SELECT_SYSTEM_PROMPT),
                HumanMessage(content=f"【家庭成员名单】\n{roster}\n\n【用户提问】\n{question}"),
            ]
        )
    except Exception:
        logger.exception("家庭档案选择失败，本次回答不带档案：question=%s", question)
        return []

    selected = _parse_selection(_message_text(reply), profiles)
    logger.info("家庭档案选择（模型判断）：%s", [p["name"] for p in selected] or "无")
    return selected


def format_profile_context(profiles: list[dict[str, Any]]) -> str:
    """把选中的成员拼成注入用的文本块，没选中成员时返回空串。"""
    if not profiles:
        return ""

    lines = ["【家庭成员档案】以下是本次回答需要参考的家人信息，请结合这些信息作答："]
    for index, profile in enumerate(profiles, start=1):
        parts = [p for p in (profile.get("relation"), profile.get("gender")) if p]
        if profile.get("age") is not None:
            parts.append(f"{profile['age']}岁")
        lines.append(f"{index}. {profile['name']}（{'，'.join(parts)}）" if parts else f"{index}. {profile['name']}")
        for key in ("conditions", "allergies", "medications", "diet", "notes"):
            value = profile.get(key)
            if value:
                lines.append(f"   - {FIELD_LABELS[key]}：{value}")

    lines.append("档案中没有的信息不要编造；与本次问题无关的档案信息不要强行套用。")
    return "\n".join(lines)


def format_profile_note(profiles: list[dict[str, Any]]) -> str:
    """回答开头那行说明，没有选中成员时返回空串（什么都不加）。"""
    if not profiles:
        return ""
    return PROFILE_NOTE_TEMPLATE.format(names="、".join(p["name"] for p in profiles))


def merge_into_question(question: str, profile_context: str) -> str:
    """把档案块拼在用户问题前面，作为模型的用户输入。

    拼在用户输入而不是系统提示词里，是为了让系统提示词保持固定（两个智能体共用同一个
    agent 实例），同时明确用【用户问题】分隔，避免模型把档案当成问题本身。
    """
    if not profile_context:
        return question
    return f"{profile_context}\n\n【用户问题】\n{question}"


def build_agent_messages(
    question: str,
    profile_context: str = "",
    history: list[dict] | None = None,
) -> list[dict]:
    """拼出一次请求的完整消息列表：历史对话 + 带家庭档案的当前问题。

    放在本模块是因为「历史 + 档案注入」是同一件事的两面（都是为了给模型补上下文），
    两个智能体共用同一份拼装逻辑，避免各写一遍导致格式不一致。
    """
    return [
        *(history or []),
        {"role": "user", "content": merge_into_question(question, profile_context)},
    ]
