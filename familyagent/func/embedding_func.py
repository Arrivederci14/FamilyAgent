"""阿里云百炼嵌入模型（text-embedding-v4）封装。

通过 OpenAI 兼容接口调用，继承 LangChain 的 Embeddings，
可直接用于 FAISS / DashVector 等向量库。
"""

from langchain_core.embeddings import Embeddings
from openai import OpenAI

from familyagent.config import config

# 百炼 text-embedding-v4 单次请求最多 10 条文本，超出需分批
BATCH_SIZE = 10


class DashScopeEmbeddings(Embeddings):
    """基于阿里云百炼获取文本向量。"""

    def __init__(self) -> None:
        self.model = config.embedding_model
        self.dimensions = config.embedding_dimension
        self._client = OpenAI(
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_base_url,
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """按批调用接口，返回与输入顺序一致的向量列表。"""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            resp = self._client.embeddings.create(
                model=self.model,
                input=texts[i : i + BATCH_SIZE],
                dimensions=self.dimensions,
            )
            vectors.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """为多条文档生成嵌入向量。"""
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """为单条查询文本生成嵌入向量。"""
        return self._embed([text])[0]


embedding_model = DashScopeEmbeddings()
