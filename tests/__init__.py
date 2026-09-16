"""已发布的功能自测：离线可跑，不需要任何密钥，也不产生 API 费用。

与 familyagent/test/ 的区别
    那个目录不随仓库发布（里面有真实提问与个人化用例，并且会真的调用大模型／向量库）。
    这里的用例全部离线、可复现：用假 LLM 替代模型、把数据写到临时目录，不碰真实数据。

运行方式（在项目根目录）
    python -m tests.test_history_store
    python -m tests.test_family_profile
    python -m tests.test_intent_recognition
    python -m tests.test_routes

为什么需要这个 __init__.py
    familyagent/config.py 一被导入就会实例化 Config，其中 dashscope_api_key、
    dashvector_api_key、dashvector_endpoint 是必填项，本地没有 .env 时会直接抛
    ValidationError —— 那样干净环境里连离线用例都跑不起来。
    所以这里在导入任何 familyagent 模块之前，先给「没有 .env」的情况补上占位值；
    一旦检测到 .env 存在就什么都不做，绝不覆盖真实配置。
"""

import os
import sys
from pathlib import Path

# Windows 控制台默认 GBK，输出里的「✓」等符号会直接抛 UnicodeEncodeError 打断测试，
# 这里统一改成「编不出的字符用 ? 代替」。必须在导入 familyagent 之前做：
# 它的日志配置会抓取当前的 sys.stderr 对象。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

if not ENV_FILE.exists():
    # 占位值，只为让 config 能实例化成功；离线用例不会真的请求这些服务
    os.environ.setdefault("DASHSCOPE_API_KEY", "placeholder-for-offline-tests")
    os.environ.setdefault("DASHVECTOR_API_KEY", "placeholder-for-offline-tests")
    os.environ.setdefault("DASHVECTOR_ENDPOINT", "placeholder.dashvector.invalid")
