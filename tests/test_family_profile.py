"""家庭档案（family_profile）的离线测试。

不调大模型：需要模型判断的地方都换成假 LLM；
数据写在临时目录，真实档案在 familyagent/familydata/ 下，测试不会碰它。

运行：python -m tests.test_family_profile
"""

import json
import shutil
import tempfile
from pathlib import Path

from familyagent.func import family_profile

_ORIGINAL_DIR = family_profile.FAMILY_DIR
_ORIGINAL_FILE = family_profile.PROFILE_FILE
_ORIGINAL_LLM = family_profile.llm


class FakeLLM:
    """假大模型：返回预设文本，或按需抛异常。"""

    def __init__(self, reply: str = "", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error

        class _Reply:
            content = self.reply

        return _Reply()


def _use_temp_dir() -> Path:
    """把档案文件切到临时目录，返回该目录。"""
    tmp = Path(tempfile.mkdtemp(prefix="fa_family_"))
    family_profile.FAMILY_DIR = tmp
    family_profile.PROFILE_FILE = tmp / "profiles.json"
    return tmp


def _sample(name: str = "奶奶", **kwargs) -> dict:
    data = {
        "name": name,
        "relation": "祖母",
        "gender": "女",
        "age": "78",
        "conditions": "高血压",
        "allergies": "青霉素",
        "medications": "氨氯地平",
        "diet": "低盐",
        "notes": "行动不便",
    }
    data.update(kwargs)
    return data


def test_create_and_list() -> None:
    """新增档案：id 生成、年龄归一化成整数、时间戳落盘。"""
    profile = family_profile.create_profile(_sample())
    assert family_profile.is_valid_profile_id(profile["id"])
    # 前端传的是字符串，落库统一转成整数
    assert profile["age"] == 78
    assert profile["created_at"] and profile["created_at"] == profile["updated_at"]

    family_profile.create_profile(_sample(name="宝宝", relation="孙子", age="", gender="男"))
    profiles = family_profile.list_profiles()
    assert [p["name"] for p in profiles] == ["奶奶", "宝宝"]
    # 年龄留空存 None，不存空串
    assert profiles[1]["age"] is None

    # 文件结构固定为 {"profiles": [...]}，方便人工查看
    raw = json.loads(family_profile.PROFILE_FILE.read_text(encoding="utf-8"))
    assert isinstance(raw["profiles"], list) and len(raw["profiles"]) == 2
    print("✓ 新增与列表")


def test_validation() -> None:
    """校验失败必须抛 ValueError（路由据此回 400），并带上中文原因。"""
    cases = [
        (_sample(name=""), "姓名"),
        (_sample(name="   "), "姓名"),
        (_sample(age="七十八"), "整数"),
        (_sample(age="200"), "0~120"),
        (_sample(notes="长" * (family_profile.MAX_FIELD_LEN + 1)), "最多"),
        ("不是字典", "格式"),
    ]
    for data, keyword in cases:
        try:
            family_profile.create_profile(data)
        except ValueError as exc:
            assert keyword in str(exc), (keyword, str(exc))
        else:
            raise AssertionError(f"应当校验失败：{data!r}")

    # 边界值要能通过：0 岁、120 岁、刚好 200 字
    before = len(family_profile.list_profiles())
    family_profile.create_profile(_sample(name="婴儿", age=0))
    family_profile.create_profile(_sample(name="寿星", age=120))
    family_profile.create_profile(_sample(name="极限", notes="长" * family_profile.MAX_FIELD_LEN))
    assert len(family_profile.list_profiles()) == before + 3
    print("✓ 字段校验与边界值")


def test_update_and_delete() -> None:
    """编辑要能真的清空字段；id 非法或不存在返回 None / False。"""
    profile = family_profile.create_profile(_sample(name="爷爷"))
    updated = family_profile.update_profile(profile["id"], _sample(name="爷爷", allergies=""))
    assert updated["allergies"] == ""
    assert updated["created_at"] == profile["created_at"]
    assert updated["id"] == profile["id"]

    assert family_profile.get_profile(profile["id"])["name"] == "爷爷"

    # id 非法：路径穿越、长度不对、非字符串
    for bad in ["../../profiles", "p_123", None, 42]:
        assert family_profile.get_profile(bad) is None, bad
        assert family_profile.update_profile(bad, _sample()) is None, bad
        assert family_profile.delete_profile(bad) is False, bad

    # id 合法但不存在
    ghost = "p_" + "0" * 32
    assert family_profile.update_profile(ghost, _sample()) is None
    assert family_profile.delete_profile(ghost) is False

    assert family_profile.delete_profile(profile["id"]) is True
    assert family_profile.get_profile(profile["id"]) is None
    print("✓ 编辑与删除")


def test_match_by_text() -> None:
    """称呼直连：提问里直接写称呼就命中，不用调模型。"""
    profiles = family_profile.list_profiles()

    hit = family_profile.match_profiles_by_text("奶奶能吃海鲜吗", profiles)
    assert [p["name"] for p in hit] == ["奶奶"]

    multi = family_profile.match_profiles_by_text("寿星和极限一起去爬山可以吗", profiles)
    assert {p["name"] for p in multi} == {"寿星", "极限"}

    assert family_profile.match_profiles_by_text("北京三日游怎么安排", profiles) == []
    # 没有任何档案时不报错
    assert family_profile.match_profiles_by_text("奶奶", []) == []
    print("✓ 称呼直连匹配")


def test_select_profiles() -> None:
    """选择链路：空问题/无档案短路，命中直连时不调模型，未命中才问模型。"""
    profiles = family_profile.list_profiles()

    fake = FakeLLM(reply="奶奶")
    family_profile.llm = fake
    try:
        # 空问题与空档案列表都直接返回空，且不产生模型调用
        assert family_profile.select_profiles("   ", profiles) == []
        assert family_profile.select_profiles("奶奶血压高吗", []) == []
        assert fake.calls == []

        # 直连命中：不调模型
        direct = family_profile.select_profiles("奶奶血压高吗", profiles)
        assert [p["name"] for p in direct] == ["奶奶"] and fake.calls == []

        # 没点名：交给模型判断，名单里带上了全部成员
        selected = family_profile.select_profiles("家里老人出游要注意什么", profiles)
        assert [p["name"] for p in selected] == ["奶奶"]
        assert len(fake.calls) == 1
        roster_text = fake.calls[0][1].content
        assert all(p["name"] in roster_text for p in profiles)

        # 模型回「无」表示一个都不选
        family_profile.llm = FakeLLM(reply="无")
        assert family_profile.select_profiles("推荐几个北京景点", profiles) == []

        # 模型多说了几句，也应能认出被点名的成员
        family_profile.llm = FakeLLM(reply="需要参考：宝宝。因为涉及婴幼儿喂养。")
        picked = family_profile.select_profiles("孩子辅食怎么加", profiles)
        assert [p["name"] for p in picked] == ["宝宝"]

        # 模型崩了 / 未初始化：退化成不带档案，不抛异常
        family_profile.llm = FakeLLM(error=RuntimeError("模型不可用"))
        assert family_profile.select_profiles("家里老人出游要注意什么", profiles) == []
        family_profile.llm = None
        assert family_profile.select_profiles("家里老人出游要注意什么", profiles) == []
    finally:
        family_profile.llm = _ORIGINAL_LLM
    print("✓ 档案选择链路与容错")


def test_formatting() -> None:
    """注入文本与回答开头说明的格式。"""
    profiles = family_profile.list_profiles()
    granny = next(p for p in profiles if p["name"] == "奶奶")

    context = family_profile.format_profile_context([granny])
    assert context.startswith("【家庭成员档案】")
    assert "奶奶（祖母，女，78岁）" in context
    assert "- 慢病/既往史：高血压" in context
    assert "- 过敏史：青霉素" in context
    assert "不要编造" in context

    assert family_profile.format_profile_context([]) == ""

    note = family_profile.format_profile_note([granny])
    assert note == "> 📋 本次回答参考了家庭档案：奶奶"
    assert family_profile.format_profile_note([]) == ""
    # 多位家人用「、」连接
    two = family_profile.format_profile_note([granny, profiles[1]])
    assert two.endswith("奶奶、宝宝")

    # 档案块拼在用户问题前面，用分隔标记隔开，避免模型把档案当成问题
    merged = family_profile.merge_into_question("她能吃海鲜吗", context)
    assert merged.startswith("【家庭成员档案】")
    assert merged.endswith("【用户问题】\n她能吃海鲜吗")
    assert family_profile.merge_into_question("天气如何", "") == "天气如何"

    history = [{"role": "user", "content": "上一轮"}, {"role": "assistant", "content": "上一答"}]
    messages = family_profile.build_agent_messages("她能吃海鲜吗", context, history)
    assert len(messages) == 3
    assert messages[0]["content"] == "上一轮" and messages[1]["content"] == "上一答"
    assert messages[-1]["role"] == "user"
    assert "【家庭成员档案】" in messages[-1]["content"]
    assert "【用户问题】\n她能吃海鲜吗" in messages[-1]["content"]

    # 没有历史时就是一条当前问题
    assert len(family_profile.build_agent_messages("你好", "", None)) == 1
    print("✓ 注入文本与说明格式")


def test_broken_file() -> None:
    """档案文件损坏时返回空列表而不是抛异常，问答主流程不受影响。"""
    family_profile.PROFILE_FILE.write_text("{ 坏掉的 json", encoding="utf-8")
    assert family_profile.list_profiles() == []
    assert family_profile.select_profiles("奶奶血压高吗") == []

    # 结构不对（顶层是列表）同样按空处理
    family_profile.PROFILE_FILE.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert family_profile.list_profiles() == []

    # 原子写不留临时文件
    family_profile.create_profile(_sample(name="重建"))
    assert list(family_profile.FAMILY_DIR.glob("*.tmp")) == []
    print("✓ 坏文件容错与原子写")


def main() -> None:
    tmp = _use_temp_dir()
    try:
        test_create_and_list()
        test_validation()
        test_update_and_delete()
        test_match_by_text()
        test_select_profiles()
        test_formatting()
        test_broken_file()
        print("\n家庭档案（family_profile）全部测试通过")
    finally:
        family_profile.FAMILY_DIR = _ORIGINAL_DIR
        family_profile.PROFILE_FILE = _ORIGINAL_FILE
        family_profile.llm = _ORIGINAL_LLM
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
