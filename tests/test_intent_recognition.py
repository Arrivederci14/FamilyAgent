"""意图识别（intent_recognition）的离线测试。

全部用假 LLM，不联网、不花一分钱：主要验证「永远返回三个合法标签之一」这个承诺——
模型回得花哨、回得没边、直接抛异常、甚至根本没初始化，都不能把异常抛给调用方。

运行：python -m tests.test_intent_recognition
"""

from familyagent.func import intent_recognition
from familyagent.func.intent_recognition import (
    CHITCHAT,
    DEFAULT_INTENT,
    HEALTH,
    INTENT_LABELS,
    TRAVEL,
    recognize_intent,
)

_ORIGINAL_LLM = intent_recognition.llm


class FakeLLM:
    """假 LLM：invoke 返回预设内容；raise_error=True 时抛异常。"""

    def __init__(self, content: str = "", *, raise_error: bool = False) -> None:
        self.content = content
        self.raise_error = raise_error

    def invoke(self, _messages: object) -> object:
        if self.raise_error:
            raise RuntimeError("模拟模型调用失败")

        class _Reply:
            content = self.content

        return _Reply()


class _use_llm:
    """临时替换模块级 llm，退出时还原。"""

    def __init__(self, fake: object) -> None:
        self.fake = fake

    def __enter__(self) -> None:
        intent_recognition.llm = self.fake

    def __exit__(self, *_exc: object) -> None:
        intent_recognition.llm = _ORIGINAL_LLM


def test_parse_tolerance() -> None:
    """模型偶尔带引号、书名号或多说几句，也应解析出正确标签。"""
    cases = [
        ("旅游规划", TRAVEL),
        ("「旅游规划」", TRAVEL),
        ("【健康问答】\n", HEALTH),
        ("这句话属于：健康问答。", HEALTH),
        ("这个问题属于闲聊，可以随便聊。", CHITCHAT),
        # 多个标签同时出现时取最先出现的那个（不猜「真正的答案」是哪个）
        ("先排除旅游规划，答案是健康问答", TRAVEL),
    ]
    for reply, expected in cases:
        with _use_llm(FakeLLM(reply)):
            label = recognize_intent("随便问一句")
        assert label == expected, f"回复 {reply!r} 应解析为 {expected}，实际 {label}"
    print("✓ 回复解析容错")


def test_context_passed_to_model() -> None:
    """最近对话要拼进提示词，追问才能沿用上一轮话题；标签仍来自模型回复。"""
    captured: list = []

    class _Recorder(FakeLLM):
        def invoke(self, messages):
            captured.append(messages)
            return super().invoke(messages)

    with _use_llm(_Recorder(TRAVEL)):
        label = recognize_intent("那第二天呢", "用户：帮我规划北京三日游\n助手：好的……")

    assert label == TRAVEL
    user_content = captured[0][1].content
    assert "最近对话" in user_content and "帮我规划北京三日游" in user_content
    # 当前问题用 <<< >>> 包起来，避免用户内容被当成新指令干扰分类
    assert "<<<\n那第二天呢\n>>>" in user_content

    # 不传 context 就不该出现「最近对话」这一段
    captured.clear()
    with _use_llm(_Recorder(CHITCHAT)):
        recognize_intent("你好")
    assert "最近对话" not in captured[0][1].content
    print("✓ 追问上下文注入")


def test_fallback() -> None:
    """调用报错、回复完全无法解析时，统一兜底返回「闲聊」。"""
    cases = [
        (FakeLLM(raise_error=True), "模型调用抛异常"),
        (FakeLLM("我不确定，好像是出行方面的吧"), "回复里没有出现任何标签"),
        (FakeLLM(""), "回复是空串"),
        (None, "LLM 未初始化"),
    ]
    for fake, desc in cases:
        with _use_llm(fake):
            label = recognize_intent("帮我规划成都三日游")
        assert label == DEFAULT_INTENT, f"{desc} 时应返回 {DEFAULT_INTENT}，实际 {label}"
    print("✓ 异常与不可解析回复兜底")


def test_content_as_blocks() -> None:
    """部分模型把正文放在分块列表里（[{"type": "text", ...}]），也要能取出来。"""

    class _BlockLLM:
        def invoke(self, _messages: object) -> object:
            class _Reply:
                content = [{"type": "text", "text": "健康"}, {"type": "text", "text": "问答"}]

            return _Reply()

    with _use_llm(_BlockLLM()):
        assert recognize_intent("孩子咳嗽怎么办") == HEALTH
    print("✓ 分块形式的回复正文")


def test_empty_input() -> None:
    """空问题或纯空白不调用模型，直接返回「闲聊」。"""

    class _Exploding:
        def invoke(self, _messages: object) -> object:
            raise AssertionError("空输入不应该调用模型")

    with _use_llm(_Exploding()):
        for question in ["", "   ", "\n\t"]:
            label = recognize_intent(question)
            assert label == DEFAULT_INTENT, f"空输入应返回 {DEFAULT_INTENT}，实际 {label}"
    print("✓ 空输入短路")


def test_labels_are_valid() -> None:
    """任何分支的返回值都必须落在三个合法标签里，调用方不必再校验。"""
    replies = ["旅游规划", "健康问答", "闲聊", "乱七八糟", ""]
    for reply in replies:
        with _use_llm(FakeLLM(reply)):
            assert recognize_intent("随便问问") in INTENT_LABELS
    assert (TRAVEL, HEALTH, CHITCHAT) == INTENT_LABELS and DEFAULT_INTENT == CHITCHAT
    print("✓ 返回值恒为合法标签")


def main() -> None:
    try:
        test_parse_tolerance()
        test_context_passed_to_model()
        test_fallback()
        test_content_as_blocks()
        test_empty_input()
        test_labels_are_valid()
        print("\n意图识别（intent_recognition）全部测试通过")
    finally:
        intent_recognition.llm = _ORIGINAL_LLM


if __name__ == "__main__":
    main()
