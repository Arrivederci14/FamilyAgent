"""FamilyAgent 家庭健康问答智能体。"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from familyagent.__main__ import main

__all__ = ["main"]


def __getattr__(name: str) -> Any:
    """惰性转发 main。

    真正取用 familyagent.main 时才导入 __main__，避免单纯 import familyagent
    就构造 FastAPI 应用、初始化 LLM 与向量库。
    """
    if name == "main":
        from familyagent.__main__ import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
