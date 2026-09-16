"""用户问题意图识别工具：仅区分「旅游规划」「健康问答」「闲聊」三类。

实现方式：复用项目已有的大模型实例，用少量样例约束模型只输出一个标签，
再对返回文本做解析与兜底校正，保证任何情况下都返回三个合法标签之一。

用法：from familyagent.func.intent_recognition import recognize_intent
测试：python -m familyagent.test.test_intent_recognition
"""

import logging
import sys
from pathlib import Path

# 直接把本文件当脚本运行时（python familyagent/func/intent_recognition.py），
# 解释器只会把 familyagent/func 加入 sys.path，导致 familyagent 包导入失败。
# 这里提前把项目根目录（familyagent 包的上级目录）插到 sys.path 最前面，
# 让两种运行方式（脚本 / python -m）都能正常导入，导入语句只能放在这段之后。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from familyagent.func.llm_config import llm  # noqa: E402

logger = logging.getLogger(__name__)

# 三个合法意图标签，函数返回值只会是其中之一
TRAVEL = "旅游规划"
HEALTH = "健康问答"
CHITCHAT = "闲聊"

INTENT_LABELS = (TRAVEL, HEALTH, CHITCHAT)

# 兜底默认值：模型未初始化、调用失败或返回内容无法解析时，统一按闲聊处理
DEFAULT_INTENT = CHITCHAT

# 模型回复里可能被带上的引号、书名号、标点，解析前先剥掉
_STRIP_CHARS = "「」『』【】\"'“”‘’。，、.：:；;!！?？\n\r\t "

INTENT_SYSTEM_PROMPT = f"""你是意图分类器，只判断用户的问题属于下面哪一类，不做任何其他任务。

可选类别只有三个：

1. {TRAVEL}：与出行相关的需求，例如行程安排、目的地选择、景点、交通、住宿、
   出行天气、带老人小孩出游的注意事项。
   例：帮我规划北京 3 天家庭游；去杭州玩需要带什么；三亚这几天下雨吗。
2. {HEALTH}：与身体健康相关的咨询，例如疾病、症状、发烧、用药、护理、
   饮食营养、孕产、婴幼儿喂养、老人慢性病。
   例：孩子发烧 38 度怎么办；老人高血压饮食要注意什么；宝宝起湿疹怎么护理。
3. {CHITCHAT}：以上两类之外的问候、寒暄、闲聊、情感陪伴和其他话题。
   例：你好；今天心情不太好；给我讲个笑话。

判定规则：
- 一个问题同时提到旅游和健康时，看用户的主要诉求：以安排行程为主选「{TRAVEL}」，
  以疾病护理为主选「{HEALTH}」。
- 如果给出了「最近对话」，它是判断当前问题的重要依据：当前问题若是上一轮话题的追问
  （例如上一轮在规划行程，这一轮问「那第二天呢」「换成室内的」），应与上一轮归为同类，
  不要因为问题本身很短就归为「{CHITCHAT}」。
- 但单纯的致谢与寒暄（如「谢谢」「好的」「收到」）仍归为「{CHITCHAT}」，不沿用上一轮话题。
- 无法确定归属时选「{CHITCHAT}」。

输出要求：只输出类别名称本身，即「{TRAVEL}」「{HEALTH}」「{CHITCHAT}」三者之一。
禁止输出解释、原因、标点、引号、换行或任何其他多余内容。"""


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


def _parse_label(reply: str) -> str | None:
    """从模型回复中解析出意图标签，解析不出返回 None。

    先做精确匹配（正常情况模型只会输出标签本身，最多带点引号标点）；
    若模型多说了几句，再退化为包含匹配，取最先出现的一个标签，避免选错。
    """
    text = reply.strip().strip(_STRIP_CHARS)
    if text in INTENT_LABELS:
        return text

    hits = [(text.find(label), label) for label in INTENT_LABELS if label in text]
    if hits:
        # 按出现位置排序，取最靠前的标签
        return min(hits)[1]
    return None


def recognize_intent(question: str, context: str = "") -> str:
    """识别用户问题的意图，返回「旅游规划」「健康问答」「闲聊」三者之一。

    context 是最近的对话摘要（多轮会话下传入），用于把「那第二天呢」这类追问
    归到上一轮的话题上，避免短问题被判成闲聊。

    识别出错（LLM 未初始化、调用异常、回复无法解析、问题为空）时，
    一律返回 DEFAULT_INTENT（闲聊），保证调用方总能拿到合法标签而不用处理异常。
    """
    if not isinstance(question, str) or not question.strip():
        logger.warning("意图识别输入为空，按闲聊处理")
        return DEFAULT_INTENT

    if llm is None:
        logger.error("意图识别失败：LLM 未初始化，按闲聊处理")
        return DEFAULT_INTENT

    question = question.strip()
    # 问题用分隔符包起来，避免用户内容被当成新指令干扰分类
    user_content = f"用户问题：\n<<<\n{question}\n>>>"
    if context.strip():
        user_content = f"最近对话：\n<<<\n{context.strip()}\n>>>\n\n{user_content}"

    try:
        reply = llm.invoke(
            [
                SystemMessage(content=INTENT_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ]
        )
    except Exception:
        logger.exception("意图识别调用失败：question=%s", question)
        return DEFAULT_INTENT

    reply_text = _message_text(reply)
    label = _parse_label(reply_text)
    if label is None:
        logger.warning(
            "意图识别结果无法解析，按闲聊处理：question=%s, reply=%r", question, reply_text
        )
        return DEFAULT_INTENT

    logger.info("意图识别完成：%s -> %s", question, label)
    return label
