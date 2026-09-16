"""HTTP 接口层（route/）的离线测试。

用 FastAPI 自带的 TestClient 直接打接口，不启动服务、不连数据库、不调大模型：
- 分发链路里的意图识别与三个智能体分支全部换成假实现，只验证「分发逻辑」本身：
  按意图走对分支、会话 id 回传、档案说明写在回答开头、整轮问答落盘。
- 档案与会话数据写到临时目录，真实数据不受影响。

运行：python -m tests.test_routes
"""

import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

from fastapi.testclient import TestClient

from familyagent.__main__ import app
from familyagent.func import family_profile, history_store
from familyagent.func.intent_recognition import CHITCHAT, HEALTH, TRAVEL
from familyagent.route import dispatch_route

# 原始状态，测试结束后还原
_ORIGINAL_HISTORY_DIR = history_store.HISTORY_DIR
_ORIGINAL_FAMILY_DIR = family_profile.FAMILY_DIR
_ORIGINAL_PROFILE_FILE = family_profile.PROFILE_FILE
_ORIGINAL_PROFILE_LLM = family_profile.llm


class _FakeChatStream:
    """假 LLM：只实现闲聊分支用到的 astream。"""

    def __init__(self, pieces: list[str]) -> None:
        self.pieces = pieces
        self.calls: list = []

    async def astream(self, messages):
        self.calls.append(messages)
        for piece in self.pieces:
            class _Chunk:
                content = piece

            yield _Chunk()


def _clear_history() -> None:
    """清空临时历史目录，让列表类断言不受前序用例影响。"""
    for path in history_store.HISTORY_DIR.glob("*"):
        path.unlink()


def test_root_page(client: TestClient) -> None:
    """根路径是服务状态页，带上页面与文档入口。"""
    response = client.get("/")
    assert response.status_code == 200
    assert "家庭智能体服务已启动" in response.text
    assert "/agent/" in response.text and "/docs" in response.text
    print("✓ 根路径状态页")


def test_agent_page(client: TestClient) -> None:
    """前端页面能被 Jinja 正常渲染，并注入分支状态。"""
    response = client.get("/agent/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "家庭智能体分发助手" in response.text
    # 三个分支的可用性由后端实时算出来注入，正常时都写「正常」
    assert "旅游规划正常" in response.text and "闲聊正常" in response.text
    print("✓ /agent/ 前端页面渲染")


def test_openapi_paths(client: TestClient) -> None:
    """接口全集：改动路由时这里会立刻报出来。"""
    paths = client.get("/openapi.json").json()["paths"]
    for path in [
        "/",
        "/agent/",
        "/agent/stream/",
        "/agent/healthy/stream/",
        "/agent/travel/stream/",
        "/agent/family/",
        "/agent/family/{profile_id}",
        "/agent/history/",
        "/agent/history/{session_id}",
    ]:
        assert path in paths, path
    print("✓ 接口路径齐全")


def test_family_api(client: TestClient) -> None:
    """家庭档案的增删改查与错误码。"""
    payload = {"name": "奶奶", "relation": "祖母", "age": "78", "conditions": "高血压"}

    created = client.post("/agent/family/", json=payload)
    assert created.status_code == 200, created.text
    profile = created.json()["profile"]
    assert profile["age"] == 78 and profile["name"] == "奶奶"

    listing = client.get("/agent/family/").json()
    assert [p["name"] for p in listing["profiles"]] == ["奶奶"]
    # 表单字段定义一并下发，前端表单与后端校验共用一份
    assert any(f["key"] == "name" for f in listing["fields"])

    edited = client.put(f"/agent/family/{profile['id']}", json={**payload, "name": "姥姥"})
    assert edited.status_code == 200 and edited.json()["profile"]["name"] == "姥姥"

    # 校验失败回 400，并把中文原因带给调用方
    bad = client.post("/agent/family/", json={"name": "   "})
    assert bad.status_code == 400 and "姓名" in bad.json()["detail"]

    # id 非法或合法但不存在，一律 404（不区分，避免用错误码探测 id 是否存在）
    assert client.put("/agent/family/p_123", json=payload).status_code == 404
    assert client.delete("/agent/family/p_" + "0" * 32).status_code == 404

    assert client.delete(f"/agent/family/{profile['id']}").json() == {"ok": True}
    assert client.get("/agent/family/").json()["profiles"] == []
    print("✓ 家庭档案接口")


def test_history_api(client: TestClient) -> None:
    """历史会话：列表摘要、详情、删除，以及非法 id 的 404。"""
    # 清掉前面分发用例留下的会话，让列表断言只依赖本用例自己的数据
    _clear_history()
    session = history_store.create_session("帮我规划北京三日游")
    history_store.append_message(session["id"], history_store.ROLE_USER, "帮我规划北京三日游")
    history_store.append_message(
        session["id"], history_store.ROLE_AI, "好的……", intent=TRAVEL, profiles=["奶奶"]
    )

    summaries = client.get("/agent/history/").json()["sessions"]
    assert [s["id"] for s in summaries] == [session["id"]]
    assert summaries[0]["title"] == "帮我规划北京三日游" and summaries[0]["count"] == 2

    detail = client.get(f"/agent/history/{session['id']}").json()
    assert len(detail["messages"]) == 2
    assert detail["messages"][1]["intent"] == TRAVEL
    assert detail["messages"][1]["profiles"] == ["奶奶"]

    # 不存在的会话与格式非法的 id 都是 404
    assert client.get("/agent/history/s_" + "0" * 32).status_code == 404
    assert client.delete("/agent/history/s_123").status_code == 404

    assert client.delete(f"/agent/history/{session['id']}").json() == {"ok": True}
    assert client.get("/agent/history/").json()["sessions"] == []
    print("✓ 历史会话接口")


def test_dispatch_empty_question(client: TestClient) -> None:
    """空问题不进分支、不调模型，直接回提示语，但仍会建出一个会话。"""
    original_intent = dispatch_route.recognize_intent
    original_llm = dispatch_route.llm

    def _explode(*_args, **_kwargs):
        raise AssertionError("空问题不应该调用意图识别")

    class _ExplodingLLM:
        async def astream(self, _messages):
            raise AssertionError("空问题不应该调用大模型")

    dispatch_route.recognize_intent = _explode
    dispatch_route.llm = _ExplodingLLM()
    try:
        response = client.post("/agent/stream/", json={"question": "   "})
        assert response.status_code == 200
        assert response.text == "请输入问题内容。"
        # 没有识别出意图就没有 X-Intent 头；会话 id 照常下发
        assert "x-intent" not in response.headers
        assert history_store.is_valid_session_id(response.headers["x-session-id"])
    finally:
        dispatch_route.recognize_intent = original_intent
        dispatch_route.llm = original_llm
    print("✓ 空问题短路")


def test_dispatch_routing(client: TestClient) -> None:
    """按意图走对分支：旅游 / 健康 / 闲聊，且响应头带回意图与会话 id。"""
    calls: dict[str, list] = {"travel": [], "health": [], "question": []}
    original = {
        "recognize_intent": dispatch_route.recognize_intent,
        "stream_travel_plan": dispatch_route.stream_travel_plan,
        "stream_health_answer": dispatch_route.stream_health_answer,
        "llm": dispatch_route.llm,
    }

    async def fake_travel(question, profile_context="", history=None):
        calls["travel"].append(question)
        yield "第 1 天：故宫。"

    def fake_health(question, profile_context="", history=None):
        calls["health"].append(question)
        yield "注意低盐饮食。"

    def pick(intent):
        def _fake(question, context=""):
            calls["question"].append(question)
            return intent

        return _fake

    dispatch_route.stream_travel_plan = fake_travel
    dispatch_route.stream_health_answer = fake_health
    try:
        # 旅游规划：走旅游分支
        dispatch_route.recognize_intent = pick(TRAVEL)
        travel = client.post("/agent/stream/", json={"question": "帮我规划北京三日游"})
        assert travel.text == "第 1 天：故宫。"
        assert calls["travel"] == ["帮我规划北京三日游"] and calls["health"] == []
        assert travel.headers["x-intent"] == quote(TRAVEL)

        # 健康问答：走健康分支
        dispatch_route.recognize_intent = pick(HEALTH)
        health = client.post("/agent/stream/", json={"question": "孩子咳嗽怎么办"})
        assert health.text == "注意低盐饮食。"
        assert calls["health"] == ["孩子咳嗽怎么办"]
        assert health.headers["x-intent"] == quote(HEALTH)

        # 闲聊：不走智能体，直接用大模型流式作答
        fake_llm = _FakeChatStream(["今天", "天气不错"])
        dispatch_route.llm = fake_llm
        dispatch_route.recognize_intent = pick(CHITCHAT)
        chat = client.post("/agent/stream/", json={"question": "陪我聊聊天"})
        assert chat.text == "今天天气不错"
        assert chat.headers["x-intent"] == quote(CHITCHAT)
        # 闲聊也带上了系统人设
        assert fake_llm.calls[0][0]["role"] == "system"
    finally:
        dispatch_route.recognize_intent = original["recognize_intent"]
        dispatch_route.stream_travel_plan = original["stream_travel_plan"]
        dispatch_route.stream_health_answer = original["stream_health_answer"]
        dispatch_route.llm = original["llm"]
    print("✓ 三个分支分发正确")


def test_dispatch_profile_and_history(client: TestClient) -> None:
    """档案注入写在回答开头，整轮问答（含意图与档案）落盘到会话历史。"""
    family_profile.create_profile(
        {"name": "奶奶", "relation": "祖母", "age": "78", "conditions": "高血压", "diet": "低盐"}
    )

    original_intent = dispatch_route.recognize_intent
    original_health = dispatch_route.stream_health_answer
    captured: list = []

    def fake_health(question, profile_context="", history=None):
        captured.append({"question": question, "profile_context": profile_context})
        yield "建议控制盐分摄入。"

    dispatch_route.recognize_intent = lambda *_args, **_kwargs: HEALTH
    dispatch_route.stream_health_answer = fake_health
    try:
        response = client.post("/agent/stream/", json={"question": "奶奶平时饮食要注意什么"})
        assert response.text.startswith("> 📋 本次回答参考了家庭档案：奶奶\n\n")
        assert response.text.endswith("建议控制盐分摄入。")
        # 档案块拼在用户问题里，两个智能体共用固定的系统提示词
        assert "【家庭成员档案】" in captured[0]["profile_context"]
        assert "低盐" in captured[0]["profile_context"]

        session_id = response.headers["x-session-id"]
        session = history_store.get_session(session_id)
        assert session["title"] == "奶奶平时饮食要注意什么"
        assert [m["role"] for m in session["messages"]] == ["user", "ai"]
        assert session["messages"][1]["intent"] == HEALTH
        assert session["messages"][1]["profiles"] == ["奶奶"]
        assert "家庭档案：奶奶" in session["messages"][1]["content"]

        # 第二轮带上同一会话：复用会话而不是新建，并沿用上一轮的档案
        again = client.post(
            "/agent/stream/",
            json={"question": "那能吃海鲜吗", "session_id": session_id},
        )
        assert again.headers["x-session-id"] == session_id
        # 「那能吃海鲜吗」里没有称呼，本轮档案由上一轮沿用而来
        assert again.text.startswith("> 📋 本次回答参考了家庭档案：奶奶\n\n")
        assert len(history_store.get_session(session_id)["messages"]) == 4
    finally:
        dispatch_route.recognize_intent = original_intent
        dispatch_route.stream_health_answer = original_health
    print("✓ 档案注入与多轮记忆")


def main() -> None:
    history_tmp = Path(tempfile.mkdtemp(prefix="fa_routes_history_"))
    family_tmp = Path(tempfile.mkdtemp(prefix="fa_routes_family_"))
    history_store.HISTORY_DIR = history_tmp
    family_profile.FAMILY_DIR = family_tmp
    family_profile.PROFILE_FILE = family_tmp / "profiles.json"
    # 档案挑选在测试里不联网：没有称呼直连命中时直接返回「不带档案」
    family_profile.llm = None

    try:
        with TestClient(app) as client:
            test_root_page(client)
            test_agent_page(client)
            test_openapi_paths(client)
            test_dispatch_empty_question(client)
            test_dispatch_routing(client)
            test_family_api(client)
            test_history_api(client)
            test_dispatch_profile_and_history(client)
        print("\nHTTP 接口层（route/）全部测试通过")
    finally:
        history_store.HISTORY_DIR = _ORIGINAL_HISTORY_DIR
        family_profile.FAMILY_DIR = _ORIGINAL_FAMILY_DIR
        family_profile.PROFILE_FILE = _ORIGINAL_PROFILE_FILE
        family_profile.llm = _ORIGINAL_PROFILE_LLM
        shutil.rmtree(history_tmp, ignore_errors=True)
        shutil.rmtree(family_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
