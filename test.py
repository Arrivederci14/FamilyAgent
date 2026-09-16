"""依赖自检：确认项目要用到的三方包在当前解释器里都能正常导入。

用途：刚克隆仓库、换了机器、或重装依赖之后跑一遍，快速判断环境有没有装齐。
它只验证「包装了没、能不能 import」，不验证接口是否连通、功能是否正确
—— 那属于 familyagent/test/ 下的功能自测（python -m familyagent.test.test_xxx）。

用法：
    uv sync         # 先把依赖装好
    python test.py  # 全部导入成功打印一行提示；缺包会直接抛 ModuleNotFoundError

注意：python-dotenv 不是本项目直接声明的依赖，它随 pydantic-settings 一起装进来
（config.py 用的是 pydantic-settings 读 .env）。这里一并校验，是为了确认这条间接
依赖链是完整的 —— 哪天它被上游摘掉，这个自检会第一时间报出来。
"""

import platform
import sys

# 验证 Web 框架和基础依赖
import fastapi
import uvicorn
import jinja2
import pypdf
import dotenv
import sse_starlette
import multipart  # python-multipart 的导入名

# 验证 langchain 系列
import langchain
import langchain_community
import langchain_openai
import langchain_core

# 验证 OpenAI 和向量数据库
import openai
import grpc
import dashvector

# 如果所有导入都没有报错，说明所有包都安装成功且可用
print(f"解释器：{sys.executable}")
print(f"Python：{platform.python_version()}")
print("所有库都安装成功并可正常导入！")
