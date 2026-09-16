"""对话历史持久化（history_store）的离线测试。

全部在临时目录里跑，不会碰 familyagent/history/ 下的真实数据。

运行：python -m tests.test_history_store
"""

import json
import shutil
import tempfile
from pathlib import Path

from familyagent.func import history_store

# 原始目录，测试结束后还原
_ORIGINAL_DIR = history_store.HISTORY_DIR


def _use_temp_dir() -> Path:
    """把历史目录切到临时目录，返回该目录。"""
    tmp = Path(tempfile.mkdtemp(prefix="fa_history_"))
    history_store.HISTORY_DIR = tmp
    return tmp


def _clear_dir() -> None:
    """清空临时目录，让依赖全量列表的用例不受前序用例影响。"""
    for path in history_store.HISTORY_DIR.glob("*"):
        path.unlink()


def _fake_clock():
    """假时钟：每次调用往后走一秒，让排序断言与真实时间无关。"""
    counter = {"tick": 0}

    def now() -> str:
        counter["tick"] += 1
        return f"2026-01-01T00:00:{counter['tick']:02d}"

    return now


def test_session_crud() -> None:
    """建会话、追加消息、标题取首条提问并截断、删除。"""
    session = history_store.create_session()
    assert history_store.is_valid_session_id(session["id"])
    assert history_store.get_session(session["id"])["messages"] == []

    question = "帮我规划一个适合全家老小的北京三日游行程安排，同行有腿脚不便的老人和两岁的宝宝"
    history_store.append_message(session["id"], history_store.ROLE_USER, question)
    history_store.append_message(
        session["id"], history_store.ROLE_AI, "好的，这是行程……", intent="旅游规划", profiles=["奶奶"]
    )

    saved = history_store.get_session(session["id"])
    assert len(saved["messages"]) == 2
    assert saved["title"] == question[: history_store.MAX_TITLE_LEN]
    assert len(saved["title"]) == history_store.MAX_TITLE_LEN
    # 额外字段（意图、档案）原样落盘，供前端渲染标签
    assert saved["messages"][1]["intent"] == "旅游规划"
    assert saved["messages"][1]["profiles"] == ["奶奶"]

    assert history_store.delete_session(session["id"]) is True
    assert history_store.get_session(session["id"]) is None
    assert history_store.delete_session(session["id"]) is False
    print("✓ 会话增删改查")


def test_invalid_session_id() -> None:
    """非法会话 id 必须被挡住：这是防路径穿越的关键校验。"""
    bad_ids = [
        "",
        "s_123",  # 长度不对
        "s_" + "Z" * 32,  # 非十六进制
        "../history",  # 路径穿越
        "s_../../etc/passwd",
        "s_" + "a" * 32 + "/../evil",
        None,
        123,
    ]
    before = set(history_store.HISTORY_DIR.glob("*"))
    for bad in bad_ids:
        assert not history_store.is_valid_session_id(bad), bad
        # 读取、写入、删除三条路径都要挡住
        assert history_store.get_session(bad) is None, bad
        assert history_store.append_message(bad, history_store.ROLE_USER, "越权") is None, bad
        assert history_store.delete_session(bad) is False, bad

    assert set(history_store.HISTORY_DIR.glob("*")) == before
    print("✓ 非法会话 id 被拒绝")


def test_get_or_create() -> None:
    """带 id 就复用，不带或 id 非法就新建。"""
    session = history_store.create_session("已有会话")
    assert history_store.get_or_create_session(session["id"])["id"] == session["id"]

    created = history_store.get_or_create_session(None, "新问题")
    assert created["id"] != session["id"] and created["title"] == "新问题"

    # 前端伪造的 id 同样不复用，直接新建
    forged = history_store.get_or_create_session("../../evil", "伪造")
    assert forged["id"] != "../../evil"
    assert history_store.is_valid_session_id(forged["id"])
    print("✓ 会话复用与新建")


def test_build_context() -> None:
    """历史上下文：条数上限、时间正序、角色名转换、空内容跳过。"""
    sid = history_store.create_session()["id"]
    for index in range(10):
        history_store.append_message(sid, history_store.ROLE_USER, f"第{index}问")
        history_store.append_message(sid, history_store.ROLE_AI, f"第{index}答")

    context = history_store.build_context(sid)
    assert len(context) == history_store.MAX_CONTEXT_MESSAGES
    assert [item["content"] for item in context[:2]] == ["第4问", "第4答"]
    # role 要转成模型认识的 user / assistant
    assert context[0]["role"] == "user" and context[-1]["role"] == "assistant"
    assert context[-1]["content"] == "第9答"
    # limit=0 表示不限长度
    assert len(history_store.build_context(sid, limit=0)) == 20

    # 空内容的消息不进上下文
    empty_sid = history_store.create_session()["id"]
    history_store.append_message(empty_sid, history_store.ROLE_USER, "有效提问")
    history_store.append_message(empty_sid, history_store.ROLE_AI, "")
    assert [item["content"] for item in history_store.build_context(empty_sid)] == ["有效提问"]

    # 首轮提问（前端不带 session_id）与不存在的会话都返回空列表
    assert history_store.build_context(None) == []
    assert history_store.build_context("s_" + "0" * 32) == []
    print("✓ 历史上下文拼装")


def test_last_profile_names() -> None:
    """沿用上一轮档案的依据：只看最近一条回答。"""
    sid = history_store.create_session()["id"]
    assert history_store.last_profile_names(sid) == []

    history_store.append_message(sid, history_store.ROLE_USER, "奶奶血压高吃什么好")
    history_store.append_message(sid, history_store.ROLE_AI, "……", profiles=["奶奶"])
    assert history_store.last_profile_names(sid) == ["奶奶"]

    # 最近一条回答没参考任何人，说明这轮本来就不带档案，不能继续往前翻
    history_store.append_message(sid, history_store.ROLE_USER, "讲个笑话")
    history_store.append_message(sid, history_store.ROLE_AI, "……", profiles=[])
    assert history_store.last_profile_names(sid) == []

    # 老数据没有 profiles 字段也不能抛异常
    history_store.append_message(sid, history_store.ROLE_AI, "……")
    assert history_store.last_profile_names(sid) == []
    print("✓ 上一轮档案沿用依据")


def test_list_sessions() -> None:
    """侧边栏列表：按更新时间倒序、带预览与条数、坏文件不影响其他会话。"""
    _clear_dir()
    original_now = history_store._now
    history_store._now = _fake_clock()
    try:
        older = history_store.create_session("较早")
        history_store.append_message(older["id"], history_store.ROLE_USER, "第一轮提问")

        newer = history_store.create_session("较晚")
        history_store.append_message(newer["id"], history_store.ROLE_USER, "第二轮提问")
        history_store.append_message(newer["id"], history_store.ROLE_AI, "回答")
    finally:
        history_store._now = original_now

    summaries = history_store.list_sessions()
    assert [s["id"] for s in summaries] == [newer["id"], older["id"]], summaries
    assert summaries[0]["count"] == 2
    assert summaries[0]["preview"] == "第二轮提问"

    # 损坏的文件跳过，其余照常列出
    (history_store.HISTORY_DIR / ("s_" + "f" * 32 + ".json")).write_text(
        "{ 坏掉的 json", encoding="utf-8"
    )
    assert len(history_store.list_sessions()) == 2
    print("✓ 历史列表与坏文件容错")


def test_atomic_write() -> None:
    """原子写：不留临时文件，内容可正常解析。"""
    sid = history_store.create_session()["id"]
    history_store.append_message(sid, history_store.ROLE_USER, "你好")

    path = history_store.HISTORY_DIR / f"{sid}.json"
    assert path.exists()
    assert list(history_store.HISTORY_DIR.glob("*.tmp")) == []
    assert json.loads(path.read_text(encoding="utf-8"))["messages"][0]["content"] == "你好"
    print("✓ 原子写不留临时文件")


def main() -> None:
    tmp = _use_temp_dir()
    try:
        test_session_crud()
        test_invalid_session_id()
        test_get_or_create()
        test_build_context()
        test_last_profile_names()
        test_list_sessions()
        test_atomic_write()
        print("\n对话历史（history_store）全部测试通过")
    finally:
        history_store.HISTORY_DIR = _ORIGINAL_DIR
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
