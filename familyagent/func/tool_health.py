"""健康知识检索工具：把 vector_db_func 的语义检索封装成 LangChain Tool。"""

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from familyagent.config import config
from familyagent.func.vector_db_func import search_vectors


NO_RESULT_TEXT = "知识库中未找到相关健康知识。"


class HealthQueryInput(BaseModel):
    """健康检索工具的输入参数。"""

    query: str = Field(description="用户的健康问题，例如：小孩发烧怎么办")


def format_retrieval_results(results: list[dict]) -> str:
    """把检索结果拼成给模型看的文本，空结果返回固定提示。

    结果行里的相似度分数是距离，越小越相似，仅作观测用。
    """
    if not results:
        return NO_RESULT_TEXT

    return "\n\n".join(
        f"【{item['metadata']['title']}】"
        f"（{item['metadata']['category']}｜相似度分数 {item['score']:.4f}）\n"
        f"{item['content']}"
        for item in results
    )


class HealthKnowledgeTool(BaseTool):
    """查询儿童/老人/婴儿健康护理知识，返回最相关的处理步骤。"""

    name: str = "health_knowledge_search"
    description: str = (
        "查询权威的儿童、老人、婴儿健康护理知识，返回具体处理步骤。"
        "当用户询问发烧、腹泻、高血压、呛奶等健康护理问题时使用。"
    )
    args_schema: type[BaseModel] = HealthQueryInput

    def _run(
        self,
        query: str,
        # 声明该参数，LangChain 才会注入回调管理器以支持链路追踪
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """语义检索健康知识库，返回最相关条目的处理步骤。"""
        return format_retrieval_results(search_vectors(query, k=config.retrieve_top_k))


health_tool = HealthKnowledgeTool()
