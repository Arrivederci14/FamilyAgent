"""通义千问（DashScope 兼容接口）LLM 配置模块。

对外暴露全局变量 llm 与初始化函数 init_llm：初始化成功 llm 为 ChatOpenAI 实例，失败为 None。
"""

import logging

from langchain_openai import ChatOpenAI

from familyagent.config import config

logger = logging.getLogger(__name__)

# 全局 LLM 实例，供项目其他模块直接导入
llm: ChatOpenAI | None = None


def _setup_logging() -> None:
    """根日志器尚未配置时才初始化，避免覆盖调用方的日志设置。"""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=config.log_level.upper(),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


def init_llm() -> ChatOpenAI | None:
    """按配置初始化全局 llm，任何异常都只记日志并置为 None，不向上抛。"""
    global llm
    try:
        llm = ChatOpenAI(
            model=config.llm_model,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_base_url,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
    except Exception:
        llm = None
        logger.exception(
            "LLM 初始化失败：model=%s, base_url=%s",
            config.llm_model,
            config.dashscope_base_url,
        )
    else:
        logger.info(
            "LLM 初始化成功：model=%s, base_url=%s, temperature=%s, max_tokens=%s",
            config.llm_model,
            config.dashscope_base_url,
            config.llm_temperature,
            config.llm_max_tokens,
        )
    return llm


_setup_logging()
init_llm()
