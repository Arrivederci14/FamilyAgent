"""FamilyAgent 服务主入口。

启动方式：python -m familyagent
接口文档：http://127.0.0.1:8000/docs
"""

import logging
from string import Template

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from familyagent.config import config
from familyagent.func.agent_health import health_agent
from familyagent.func.agent_travel import travel_agent
from familyagent.route import (
    AGENT_STATUS_ERROR,
    AGENT_STATUS_OK,
    dispatch_route,
    family_route,
    health_route,
    history_route,
    travel_route,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=config.app_name,
    version="0.1.0",
    description="家庭智能体分发服务：识别问题意图后转给旅游规划 / 健康问答智能体或通用大模型。",
)

# 注册业务路由（均挂在 /agent 下）
# 分发助手在前，它的 /agent/ 页面是全站唯一前端入口
app.include_router(dispatch_route.router)
app.include_router(health_route.router)
app.include_router(travel_route.router)
app.include_router(family_route.router)
app.include_router(history_route.router)


# 根路径页面，$xxx 为 string.Template 占位符（CSS 中不含 $，不会被误替换）
ROOT_PAGE = Template("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>$title</title>
  <style>
    body { margin: 0; min-height: 100vh; display: flex; align-items: center;
           justify-content: center; background: #f5f6f8; color: #222;
           font-family: system-ui, "Microsoft YaHei", sans-serif; }
    .card { padding: 40px 48px; background: #fff; border-radius: 12px;
            box-shadow: 0 2px 16px rgba(0, 0, 0, .08); text-align: center; }
    h1 { margin: 0 0 12px; font-size: 20px; }
    .status { margin: 0 0 16px; font-size: 14px; color: #666; }
    .dot { display: inline-block; width: 8px; height: 8px; margin-right: 6px;
           border-radius: 50%; background: #22c55e; }
    .dot.bad { background: #ef4444; }
    .agents { display: flex; justify-content: center; gap: 10px; margin: 0 0 26px;
              font-size: 13px; }
    .agents span { display: inline-flex; align-items: center; padding: 5px 12px;
                   background: #f7f8fa; border-radius: 999px; color: #555; }
    .links { display: flex; flex-direction: column; gap: 10px; }
    .links a { padding: 10px 20px; border: 1px solid #e2e5ea; border-radius: 8px;
               font-size: 14px; color: #2563eb; text-decoration: none; }
    .links a:hover { background: #f0f5ff; border-color: #b9d0ff; }
  </style>
</head>
<body>
  <div class="card">
    <h1>家庭智能体服务已启动</h1>
    <p class="status"><span class="dot"></span>服务状态：$service_status</p>
    <div class="agents">
      <span><span class="dot$health_dot"></span>健康问答：$health_status</span>
      <span><span class="dot$travel_dot"></span>旅游规划：$travel_status</span>
    </div>
    <div class="links">
      <a href="/agent/">智能体分发助手（自动识别意图 · 前端入口）</a>
      <a href="/docs">API 文档 (Swagger UI)</a>
      <a href="/redoc">API 文档 (ReDoc)</a>
    </div>
  </div>
</body>
</html>
""")


def _agent_dot(ready: bool) -> str:
    """智能体就绪时圆点为绿色，否则附加 bad 类变红。"""
    return "" if ready else " bad"


@app.get("/", summary="服务根路径", tags=["服务信息"], response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """服务状态页：展示运行状态、各智能体可用性与页面/文档入口。"""
    health_ready = health_agent is not None
    travel_ready = travel_agent is not None
    return HTMLResponse(
        ROOT_PAGE.substitute(
            title=config.app_name,
            service_status="运行中",
            health_dot=_agent_dot(health_ready),
            health_status=AGENT_STATUS_OK if health_ready else AGENT_STATUS_ERROR,
            travel_dot=_agent_dot(travel_ready),
            travel_status=AGENT_STATUS_OK if travel_ready else AGENT_STATUS_ERROR,
        )
    )


def main() -> None:
    """启动 uvicorn 服务。

    传导入字符串而非 app 对象，uvicorn 才会开启热重载子进程。
    """
    logger.info(
        "启动 %s：http://%s:%s（reload=%s, env=%s）",
        config.app_name,
        config.app_host,
        config.app_port,
        config.app_reload,
        config.app_env,
    )
    uvicorn.run(
        "familyagent.__main__:app",
        host=config.app_host,
        port=config.app_port,
        reload=config.app_reload,
        log_level=config.log_level,
    )


if __name__ == "__main__":
    main()
