"""健康问答智能体：基于 LangChain create_agent 的检索增强问答。

检索到本地知识库内容时按知识库作答，检索不到时退回模型自身知识并加免责标注。
"""

import logging
from collections.abc import Iterator

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk

from familyagent.func.family_profile import build_agent_messages
from familyagent.func.llm_config import llm
from familyagent.func.tool_health import health_tool

logger = logging.getLogger(__name__)

AI_DISCLAIMER = "【⚠️ AI生成内容，仅供参考，请咨询专业医生】"
SERVICE_UNAVAILABLE = "健康问答服务暂时不可用，请稍后再试。"

SYSTEM_PROMPT = f"""你是家庭健康护理助手，负责解答儿童、老人、婴儿的日常健康护理问题。

回答规则：
1. 收到健康问题时，必须先调用 health_knowledge_search 工具检索本地权威健康知识库。
2. 检索到有效内容时，直接整理工具返回的内容作为回答，禁止添加任何自己生成的内容或建议。
3. 检索不到内容时，才可以用你自己的知识回答，并且必须在回答的最开头加上这一行：
{AI_DISCLAIMER}
4. 回答要简洁专业，只保留核心信息，不要客套话。"""


def create_health_agent():
    """创建健康问答智能体，失败时记日志并返回 None。"""
    if llm is None:
        logger.error("健康问答智能体创建失败：LLM 未初始化")
        return None

    try:
        agent = create_agent(model=llm, tools=[health_tool], system_prompt=SYSTEM_PROMPT)
    except Exception:
        logger.exception("健康问答智能体创建失败：model=%s", llm.model_name)
        return None

    logger.info("健康问答智能体创建成功：model=%s, tools=%s", llm.model_name, [health_tool.name])
    return agent


def _text_of(message: object) -> str:
    """取出消息正文，兼容 content 为字符串或分块列表两种形式。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


def ask_health_answer(
    question: str,
    profile_context: str = "",
    history: list[dict] | None = None,
) -> str:
    """非流式问答：一次调用拿到完整回答。

    profile_context 为路由挑好的家庭档案文本块，history 为该会话的历史消息。
    """
    if health_agent is None:
        return SERVICE_UNAVAILABLE

    result = health_agent.invoke(
        {"messages": build_agent_messages(question, profile_context, history)}
    )
    # 从后往前找最后一条不带工具调用的 AI 消息，即模型最终答案
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return _text_of(message)
    return ""


def stream_health_answer(
    question: str,
    profile_context: str = "",
    history: list[dict] | None = None,
) -> Iterator[str]:
    """流式返回回答片段，逐段吐出模型最终回答（不含工具调用过程）。

    profile_context 为路由挑好的家庭档案文本块，history 为该会话的历史消息。
    """
    if health_agent is None:
        yield SERVICE_UNAVAILABLE
        return

    for chunk, _metadata in health_agent.stream(
        {"messages": build_agent_messages(question, profile_context, history)},
        stream_mode="messages",
    ):
        # 跳过工具调用产生的片段，只保留最终答案
        if isinstance(chunk, AIMessageChunk) and chunk.content and not chunk.tool_call_chunks:
            yield chunk.content


health_agent = create_health_agent()
