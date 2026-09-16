"""健康知识库向量化与检索（LangChain + 阿里云 DashVector）。

流程：校验 CSV -> 读取 -> 分块 -> 嵌入 -> 写入 DashVector -> 相似度检索
DashVector 接入方式参考：https://help.aliyun.com/zh/document_detail/2510225.html
"""

import csv
import hashlib
import time
from pathlib import Path

import dashvector
from langchain_community.vectorstores import DashVector
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from familyagent.config import config
from familyagent.func.embedding_func import embedding_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / config.data_dir / config.knowledge_file
REQUIRED_COLUMNS = ("title", "category", "content")
TEXT_FIELD = "content"  # DashVector 中存放正文的字段，检索时还原成 page_content
PARTITION = "default"

# init_vector_db 初始化的向量库，供检索函数复用
_vector_store: DashVector | None = None


def validate_csv_file(csv_path: Path = DEFAULT_CSV_PATH) -> None:
    """校验知识库文件是否存在、是否包含必需列。"""
    if not csv_path.exists():
        raise FileNotFoundError(f"知识库文件不存在：{csv_path}")

    with csv_path.open(encoding="utf-8", newline="") as f:
        columns = csv.DictReader(f).fieldnames or []

    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"CSV 缺少必需列 {missing}，实际列为 {columns}")


def load_csv_data(csv_path: Path = DEFAULT_CSV_PATH) -> list[dict[str, str]]:
    """校验并读取 CSV，返回 [{title, category, content}]。"""
    validate_csv_file(csv_path)

    with csv_path.open(encoding="utf-8", newline="") as f:
        data = [
            {column: row[column].strip() for column in REQUIRED_COLUMNS}
            for row in csv.DictReader(f)
            if (row.get("content") or "").strip()
        ]

    if not data:
        raise ValueError(f"知识库文件无有效数据行：{csv_path}")
    return data


def split_text_chunks(data: list[dict[str, str]]) -> list[Document]:
    """按中文标点优先分块，并打印分块数量。"""
    documents = [
        Document(
            page_content=item["content"],
            metadata={"title": item["title"], "category": item["category"]},
        )
        for item in data
    ]
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.chunk_separators,
    ).split_documents(documents)

    print(
        f"[分块] {len(documents)} 篇文档 -> {len(chunks)} 个分块 "
        f"(chunk_size={config.chunk_size}, chunk_overlap={config.chunk_overlap})"
    )
    return chunks


def _chunk_id(chunk: Document) -> str:
    """用标题+内容生成稳定主键，重复入库是覆盖而不是新增。"""
    return hashlib.md5(f"{chunk.metadata['title']}|{chunk.page_content}".encode()).hexdigest()


def _get_collection(collection_name: str) -> dashvector.Collection:
    """获取 DashVector 集合，不存在则按配置维度创建。"""
    client = dashvector.Client(
        api_key=config.dashvector_api_key,
        endpoint=config.dashvector_endpoint,
    )
    collection = client.get(collection_name)
    if not collection:
        resp = client.create(collection_name, dimension=config.embedding_dimension, metric="cosine")
        if not resp:
            raise RuntimeError(f"创建集合失败：{resp.message}")
        collection = client.get(collection_name)
        print(f"[集合] 已创建集合 {collection_name}（维度 {config.embedding_dimension}）")
    return collection


def _wait_index_ready(
    collection: dashvector.Collection,
    expected: int,
    timeout: float = 30.0,
) -> None:
    """等待异步索引完成。

    DashVector 写入后建索引是异步的，索引完成前检索会返回 0 条。
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stats = collection.stats()
        if stats:
            output = stats.output
            if output.index_completeness == 1.0 and output.total_doc_count >= expected:
                print(f"[索引] 索引就绪（文档数 {output.total_doc_count}）")
                return
        time.sleep(0.5)
    print(f"[索引] 等待索引超时（{timeout}s），检索可能暂时查不到数据")


def _get_vector_store(collection_name: str | None = None) -> DashVector:
    """优先复用 init_vector_db 初始化好的向量库，否则按集合名即时构建。"""
    if collection_name is None and _vector_store is not None:
        return _vector_store
    collection = _get_collection(collection_name or config.dashvector_collection)
    return DashVector(collection, embedding_model, TEXT_FIELD)


def _upsert_chunks(collection: dashvector.Collection, chunks: list[Document]) -> None:
    """按 md5 主键写入分块，重复初始化是覆盖而不是新增。

    这里不走 DashVector.add_texts：它传的是 (id, vector, fields) 元组，
    而 dashvector SDK 的元组分支会丢掉 id 并自动生成主键，
    导致每次重新初始化都写入一批重复数据。
    """
    vectors = embedding_model.embed_documents([chunk.page_content for chunk in chunks])
    docs = [
        dashvector.Doc(
            id=_chunk_id(chunk),
            vector=vector,
            fields={**chunk.metadata, TEXT_FIELD: chunk.page_content},
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    resp = collection.upsert(docs, partition=PARTITION)
    if not resp:
        raise RuntimeError(f"写入集合失败：{resp.message}")


def init_vector_db(
    *,
    collection_name: str | None = None,
    csv_path: Path | None = None,
    recreate: bool = False,
) -> DashVector:
    """向量化 CSV 知识库并写入 DashVector，返回可直接检索的向量库对象。"""
    global _vector_store

    collection_name = collection_name or config.dashvector_collection
    chunks = split_text_chunks(load_csv_data(csv_path or DEFAULT_CSV_PATH))

    collection = _get_collection(collection_name)
    if recreate:
        if not collection.delete(delete_all=True, partition=PARTITION):
            raise RuntimeError(f"清空集合 {collection_name} 失败")
        print(f"[集合] 已清空集合 {collection_name} 的旧数据")

    _upsert_chunks(collection, chunks)
    _vector_store = DashVector(collection, embedding_model, TEXT_FIELD)
    print(f"[入库] 已写入 {len(chunks)} 个分块到集合 {collection_name}")
    _wait_index_ready(collection, len(chunks))
    return _vector_store


def _format_results(results: list[tuple[Document, float]]) -> list[dict]:
    """转成 [{content, metadata, score}]，并丢弃低于相关性下限的条目。

    score 是距离，越小越相似，因此大于 config.retrieve_score_threshold 视为不相关。
    """
    return [
        {"content": document.page_content, "metadata": document.metadata, "score": score}
        for document, score in results
        if score <= config.retrieve_score_threshold
    ]


def search_vectors(
    query: str,
    k: int | None = None,
    collection_name: str | None = None,
) -> list[dict]:
    """相似度检索，返回 [{content, metadata, score}]，score 越小越相似。"""
    results = _get_vector_store(collection_name).similarity_search_with_relevance_scores(
        query,
        k=k or config.retrieve_top_k,
        partition=PARTITION,
    )
    return _format_results(results)


def search_vectors_by_category(
    query: str,
    category: str,
    k: int | None = None,
    collection_name: str | None = None,
) -> list[dict]:
    """在指定 category 内做相似度检索。"""
    results = _get_vector_store(collection_name).similarity_search_with_relevance_scores(
        query,
        k=k or config.retrieve_top_k,
        filter=f"category = '{category}'",
        partition=PARTITION,
    )
    return _format_results(results)


def quick_search(query: str, k: int | None = None) -> list[dict]:
    """使用 init_vector_db 初始化好的向量库直接检索。"""
    if _vector_store is None:
        raise RuntimeError("向量库尚未初始化，请先调用 init_vector_db()")
    return search_vectors(query, k)
