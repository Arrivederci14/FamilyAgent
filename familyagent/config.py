from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Config(BaseSettings):
    """项目配置，字段自动从 .env / 环境变量读取（不区分大小写）。"""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 阿里云百炼（LLM 与嵌入模型共用）
    dashscope_api_key: str
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 大语言模型
    llm_model: str = "qwen-plus"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048

    # 嵌入模型
    embedding_model: str = "text-embedding-v4"
    embedding_dimension: int = 1024

    # 阿里向量存储 DashVector
    dashvector_api_key: str
    dashvector_endpoint: str
    dashvector_collection: str = "familyagent"

    # 和风天气 QWeather（家庭旅游规划 Agent 用）
    # 留空表示未启用，get_weather 工具会返回明确的未配置提示而不是报错
    weather_api_key: str = ""
    # 账号专属 API Host，形如 xxxxxx.re.qweatherapi.com，见和风控制台「项目」页
    weather_api_host: str = ""
    weather_timeout: float = 10.0

    # FastAPI 服务
    app_name: str = "FamilyAgent"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_reload: bool = True
    log_level: str = "info"

    # 对话历史与家庭档案：本地持久化目录，均已加入 .gitignore
    # 相对项目根目录，实际位置在 familyagent/history/ 与 familyagent/familydata/
    history_dir: str = "history"
    family_dir: str = "familydata"

    # 知识库 / RAG
    upload_dir: str = "uploads"
    data_dir: str = "data"
    knowledge_file: str = "health_knowledge.csv"
    chunk_size: int = 200
    chunk_overlap: int = 30
    # 分块分隔符优先级：先按分号/句号切，再退到换行、逗号、空格
    # 如需在 .env 中覆盖，请写成 JSON 数组形式
    chunk_separators: list[str] = ["；", "。", "\n", "，", " ", ""]
    retrieve_top_k: int = 1
    # 相关性下限：score 为距离，越小越相似，大于该值的检索结果会被丢弃
    # 实测相关问题的 top1 在 0.26~0.34，无关问题在 0.69 以上，取 0.5 作分界
    # 设为 1.0 表示不过滤
    retrieve_score_threshold: float = 0.5


config = Config()
