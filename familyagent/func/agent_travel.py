"""家庭旅游规划智能体：基于 LangChain create_agent，面向「老人 + 儿童」出行的 3 天行程规划。

强制先调用 get_weather 拿到未来 3 天真实天气（含天气指数与空气质量），
再结合这些数据逐日生成舒缓友好、适配天气的家庭旅游路线。

测试：python -m familyagent.test.test_agent_travel
"""

import logging
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, NamedTuple

import httpx
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk
from langchain_core.tools import tool

from familyagent.config import config
from familyagent.func.family_profile import build_agent_messages
from familyagent.func.llm_config import llm

logger = logging.getLogger(__name__)

SERVICE_UNAVAILABLE = "旅游规划服务暂时不可用，请稍后再试。"

WEATHER_NOT_CONFIGURED = (
    "天气服务未配置：请在 .env 中填写 WEATHER_API_KEY 与 WEATHER_API_HOST"
    "（和风天气控制台 →「项目」页可查看专属 API Host）。"
)

# 和风天气路径与状态码
QWEATHER_GEO_PATH = "/geo/v2/city/lookup"
QWEATHER_FORECAST_PATH = "/v7/weather/3d"
QWEATHER_INDICES_PATH = "/v7/indices/3d"
# 空气质量接口把经纬度放在路径里，且返回城市级当前值，不是逐日预报
QWEATHER_AIR_PATH = "/airquality/v1/current/{lat}/{lon}"
# 天气预警接口同样用经纬度，返回当前生效的预警列表
QWEATHER_ALERT_PATH = "/weatheralert/v1/current/{lat}/{lon}"
QWEATHER_OK = "200"

# 预警 messageType 为 cancel 表示预警已解除，不属于生效中的预警
WEATHER_ALERT_CANCEL = "cancel"

ALERT_COLOR_NAMES = {"blue": "蓝色", "yellow": "黄色", "orange": "橙色", "red": "红色"}
ALERT_SEVERITY_NAMES = {
    "minor": "一般",
    "moderate": "较重",
    "severe": "严重",
    "extreme": "特别严重",
}

# 和风返回的时间是 UTC，统一转成北京时间展示
BEIJING_TZ = timezone(timedelta(hours=8))

# 天气指数类型：穿衣、紫外线、感冒。改这里即可增减，详见和风 indices 文档
QWEATHER_INDEX_TYPES = "3,5,9"

QWEATHER_ERRORS = {
    "204": "没有查询到匹配的城市，请确认城市名称。",
    "400": "天气服务请求参数有误。",
    "401": "天气服务认证失败，请检查 WEATHER_API_KEY 是否正确。",
    "402": "天气服务调用额度不足或已欠费。",
    "403": "天气服务无访问权限，请在和风控制台确认该项目已勾选对应 API（如 GeoAPI 城市搜索）。",
    "404": "天气服务未找到该城市。",
    "500": "天气服务内部错误，请稍后重试。",
}

TRAVEL_SYSTEM_PROMPT = """你是家庭旅游规划助手，专门为「老人 + 儿童」同行的家庭设计 3 天旅游路线。

【强制流程】
1. 收到需求后，必须先调用 get_weather 工具查询目的地未来 3 天的天气，禁止凭记忆猜测或编造天气。
2. 只有拿到工具返回的天气数据后，才能开始规划行程。
3. 如果工具返回查询失败，必须如实告知用户失败原因并停止规划，禁止在缺少天气数据的情况下继续输出行程。

【行程要求】
- 天气预警优先级最高。工具返回了生效中的预警时，必须在回答开头先提示预警内容，
  并按预警的防御指南调整行程；橙色或红色预警要明确建议改期或改为全程室内活动。
- 严格按工具返回的 3 天逐日安排，每天的行程必须与当天天气匹配：
  雨天、高温天优先安排室内或室内外结合的活动；空气质量差时减少户外停留。
- 参考工具返回的穿衣指数、紫外线指数和感冒指数来安排当天的活动强度和着装，不要与之矛盾。
- 节奏舒缓：每天核心景点不超过 2 个，景点之间预留交通和休息时间，避免长时间步行和爬坡。
- 每隔 2 小时左右安排一次休息点（公园长椅、茶室、母婴室、商场休息区等），并说明为什么适合老人或儿童。
- 每天给出穿衣建议，结合当天的天气指数、温度区间、昼夜温差和降水。
- 每天推荐午餐和晚餐，优先清淡、少油、易咀嚼、可提供儿童餐或软食的餐厅类型，并说明大致所在的区域。

【输出格式】
按「第 1 天 / 第 2 天 / 第 3 天」分段，每天包含：天气小结、当日行程、休息点、餐饮推荐、穿衣与随身物品提醒。
最后补充一段针对老人和儿童的通用注意事项。
语言简洁专业，只保留核心信息，不要客套话。"""


class WeatherError(Exception):
    """天气查询失败，异常信息可直接展示给模型和用户。"""


class Location(NamedTuple):
    """地理编码结果：LocationID 供预报/指数接口用，经纬度供空气质量接口用。"""

    id: str
    label: str
    lat: str
    lon: str


def _to_float(value: object) -> float | None:
    """把接口返回的数值字符串转成 float，缺失或异常时返回 None。"""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _weekday_label(fx_date: str) -> str:
    """把 fxDate（YYYY-MM-DD）转成「周三」形式，解析失败返回空串。"""
    try:
        return "周" + "一二三四五六日"[date.fromisoformat(fx_date).weekday()]
    except ValueError:
        return ""


def _travel_notes(temp_min: float | None, temp_max: float | None, precip: float | None) -> str:
    """给出天气指数覆盖不到的出行提醒：昼夜温差、低温保暖、降水防滑。"""
    notes = []
    if temp_min is not None and temp_max is not None and temp_max - temp_min >= 8:
        notes.append("昼夜温差大，务必给老人和孩子各备一件外套")
    if temp_min is not None and temp_min <= 5:
        notes.append("早晚温度低，老人和儿童需额外注意保暖")
    if precip is not None and precip > 0:
        notes.append("有降水，需带伞并给老人和孩子准备防滑鞋")
    return "；".join(notes)


def _clothing_fallback(temp_min: float | None, temp_max: float | None) -> str:
    """穿衣指数取不到时的兜底建议，仅按气温粗略判断。"""
    if temp_min is None or temp_max is None:
        return "气温数据缺失，请按当地当季常规着装准备。"
    if temp_max >= 30:
        return "炎热，建议短袖短裤、透气鞋，注意遮阳防晒"
    if temp_max >= 25:
        return "偏热，建议短袖或薄长袖"
    if temp_max >= 20:
        return "舒适，建议长袖衬衫或薄外套"
    if temp_max >= 15:
        return "微凉，建议加穿外套或针织衫"
    if temp_max >= 10:
        return "较冷，建议毛衣加厚外套"
    return "寒冷，建议羽绒服或厚棉衣，注意头颈部保暖"


def _qweather_get(
    path: str, params: dict[str, str] | None = None, *, check_code: bool = True
) -> dict:
    """调用和风天气接口，失败统一抛 WeatherError。

    check_code=False 用于新版 REST 风格接口（如空气质量 v1），
    这类接口响应里没有 code 信封，成功与否只看 HTTP 状态码。
    """
    if not config.weather_api_key or not config.weather_api_host:
        raise WeatherError(WEATHER_NOT_CONFIGURED)

    url = f"https://{config.weather_api_host}{path}"
    try:
        response = httpx.get(
            url,
            params={**(params or {}), "key": config.weather_api_key},
            timeout=config.weather_timeout,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as exc:
        logger.warning("和风天气请求失败：path=%s, params=%s, error=%s", path, params, exc)
        raise WeatherError(f"网络请求失败（{exc}）。") from exc
    except ValueError as exc:
        raise WeatherError("天气服务返回了无法解析的数据。") from exc

    if not check_code:
        return data

    code = str(data.get("code", ""))
    if code != QWEATHER_OK:
        logger.warning("和风天气返回错误码：path=%s, code=%s", path, code)
        raise WeatherError(QWEATHER_ERRORS.get(code, f"天气服务返回错误码 {code}。"))
    return data


def _lookup_location(city: str) -> Location:
    """城市名 → LocationID + 经纬度。依赖 GeoAPI，需在控制台勾选该 API。"""
    data = _qweather_get(QWEATHER_GEO_PATH, {"location": city, "number": "1"})
    locations = data.get("location") or []
    if not locations:
        raise WeatherError(f"没有找到城市「{city}」，请确认城市名称。")

    item = locations[0]
    # 国家/省/市/区可能重复（如「中国 北京市 北京 北京」），按序去重
    parts = [item.get(key) for key in ("country", "adm1", "adm2", "name")]
    label = " ".join(dict.fromkeys(part for part in parts if part))
    return Location(
        id=str(item["id"]),
        label=label or city,
        lat=str(item.get("lat", "")),
        lon=str(item.get("lon", "")),
    )


def _fetch_indices(location_id: str) -> dict[str, list[dict]]:
    """取天气指数，按日期分组。失败不致命，返回空字典，行程照常生成。"""
    try:
        data = _qweather_get(
            QWEATHER_INDICES_PATH,
            {"location": location_id, "type": QWEATHER_INDEX_TYPES},
        )
    except WeatherError as exc:
        logger.warning("天气指数获取失败，本次跳过：%s", exc)
        return {}

    grouped: dict[str, list[dict]] = {}
    for item in data.get("daily") or []:
        grouped.setdefault(str(item.get("date", "")), []).append(item)
    return grouped


def _fetch_air(lat: str, lon: str) -> dict | None:
    """取空气质量当前值。失败不致命，返回 None。"""
    if not lat or not lon:
        return None
    try:
        data = _qweather_get(
            QWEATHER_AIR_PATH.format(lat=lat, lon=lon), check_code=False
        )
    except WeatherError as exc:
        logger.warning("空气质量获取失败，本次跳过：%s", exc)
        return None

    indexes = data.get("indexes") or []
    if not indexes:
        return None
    # 优先国标 AQI，否则取第一个
    return next((i for i in indexes if i.get("code") == "cn-mee"), indexes[0])


def _format_air(air: dict) -> str:
    """把空气质量当前值整理成一段说明。"""
    pollutant = (air.get("primaryPollutant") or {}).get("fullName") or "无"
    health = air.get("health") or {}
    advice = health.get("advice") or {}
    # 接口返回的建议原文已自带「一般人群 / 敏感人群」主语，直接拼接即可
    advice_text = " ".join(
        part
        for part in (advice.get("generalPopulation"), advice.get("sensitivePopulation"))
        if part
    )
    return "\n".join(
        [
            "【空气质量（当前值，城市级，非逐日）】",
            f"  AQI {air.get('aqiDisplay') or air.get('aqi')}（{air.get('category') or '未知'}）"
            f" ｜ 首要污染物：{pollutant}",
            f"  健康影响：{health.get('effect') or '暂无'}",
            f"  建议：{advice_text or '暂无'}",
        ]
    )


def _local_time(value: object) -> str:
    """把接口返回的 ISO8601 UTC 时间转成北京时间，解析失败原样返回。"""
    if not value:
        return "未知"
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(
            BEIJING_TZ
        ).strftime("%m-%d %H:%M")
    except ValueError:
        return str(value)


def _fetch_alerts(lat: str, lon: str) -> list[dict]:
    """取当前生效的天气预警，已解除（messageType=cancel）的会被剔除。失败不致命。"""
    if not lat or not lon:
        return []
    try:
        data = _qweather_get(
            QWEATHER_ALERT_PATH.format(lat=lat, lon=lon), check_code=False
        )
    except WeatherError as exc:
        logger.warning("天气预警获取失败，本次跳过：%s", exc)
        return []

    return [
        item
        for item in data.get("alerts") or []
        if (item.get("messageType") or {}).get("code") != WEATHER_ALERT_CANCEL
    ]


def _format_alerts(alerts: list[dict]) -> str:
    """把生效中的预警整理成一段说明。防御指南对老人儿童出行最有用，完整保留。"""
    if not alerts:
        return "【天气预警】当前无生效中的天气预警。"

    lines = [f"【天气预警】当前有 {len(alerts)} 条生效预警，行程必须据此调整"]
    for item in alerts:
        event = (item.get("eventType") or {}).get("name") or "未知"
        color = ALERT_COLOR_NAMES.get((item.get("color") or {}).get("code"), "")
        severity = ALERT_SEVERITY_NAMES.get(str(item.get("severity")), "")
        instruction = " ".join(str(item.get("instruction") or "").split())

        lines.append(f"  · {item.get('headline') or event}")
        lines.append(
            f"    类型：{event}{color}预警 ｜ 等级：{severity or '未知'}"
            f" ｜ 发布：{item.get('senderName') or '未知'}"
            f" ｜ 时间：{_local_time(item.get('issuedTime'))}"
        )
        lines.append(f"    生效至：{_local_time(item.get('expireTime'))}")
        description = " ".join(str(item.get("description") or "").split())
        if description:
            lines.append(f"    说明：{description}")
        if instruction:
            lines.append(f"    防御指南：{instruction}")

    return "\n".join(lines)


def _format_forecast(
    location: Location,
    update_time: str,
    daily: list[dict],
    indices: dict[str, list[dict]],
    air: dict | None,
    alerts: list[dict],
) -> str:
    """把天气预警、逐日预报、天气指数、空气质量整理成模型易读的文本。"""
    blocks = [
        # 预警放在最前面，安全信息优先进入模型视野
        _format_alerts(alerts),
        f"【{location.label} 未来 {len(daily)} 天天气】（数据更新时间 {update_time}）",
    ]

    for index, day in enumerate(daily, start=1):
        fx_date = str(day.get("fxDate", ""))
        temp_min = _to_float(day.get("tempMin"))
        temp_max = _to_float(day.get("tempMax"))
        precip = _to_float(day.get("precip"))

        if temp_min is None or temp_max is None:
            temp_text, diff_text = "未知", ""
        else:
            temp_text = f"{temp_min:.0f} ~ {temp_max:.0f}℃"
            diff_text = f"（昼夜温差 {temp_max - temp_min:.0f}℃）"

        lines = [
            f"第 {index} 天  {fx_date} {_weekday_label(fx_date)}".rstrip(),
            f"  天气：白天 {day.get('textDay') or '未知'} / 夜间 {day.get('textNight') or '未知'}",
            f"  气温：{temp_text}{diff_text}",
            f"  风力：白天 {day.get('windDirDay') or '未知'} {day.get('windScaleDay') or '?'} 级"
            f" / 夜间 {day.get('windDirNight') or '未知'} {day.get('windScaleNight') or '?'} 级",
            f"  湿度 {day.get('humidity') or '?'}%（降水 {day.get('precip') or '?'}mm）"
            f" ｜ 紫外线指数 {day.get('uvIndex') or '?'}",
            f"  日出 {day.get('sunrise') or '?'} ｜ 日落 {day.get('sunset') or '?'}",
        ]

        day_indices = indices.get(fx_date) or []
        for item in day_indices:
            lines.append(f"  {item.get('name')}：{item.get('category')} —— {item.get('text')}")
        if not any(item.get("type") == "3" for item in day_indices):
            # 穿衣指数缺失时兜底，避免行程里完全没有穿衣建议
            lines.append(f"  穿衣参考（兜底）：{_clothing_fallback(temp_min, temp_max)}")

        notes = _travel_notes(temp_min, temp_max, precip)
        if notes:
            lines.append(f"  出行提醒：{notes}")

        blocks.append("\n".join(lines))

    if air:
        blocks.append(_format_air(air))

    return "\n\n".join(blocks)


@tool
def get_weather(city: Annotated[str, "中文城市名，例如：北京、杭州"]) -> str:
    """查询指定城市未来 3 天的详细天气，供家庭旅游规划使用。

    返回内容包含：当前生效的天气预警（含防御指南）、逐日天气现象、气温区间与
    昼夜温差、风力、湿度、降水量、紫外线强度、日出日落；以及穿衣指数、紫外线指数、
    感冒指数等天气指数，和该城市当前的空气质量（AQI、首要污染物、健康建议）。

    任何需要按天气安排行程的规划任务，都必须先调用本工具获取真实天气。
    """
    try:
        location = _lookup_location(city)

        # 四个数据接口相互独立，并发拉取；只有预报是必需的，其余失败即跳过
        with ThreadPoolExecutor(max_workers=4) as pool:
            forecast_future = pool.submit(
                _qweather_get, QWEATHER_FORECAST_PATH, {"location": location.id}
            )
            indices_future = pool.submit(_fetch_indices, location.id)
            air_future = pool.submit(_fetch_air, location.lat, location.lon)
            alerts_future = pool.submit(_fetch_alerts, location.lat, location.lon)

            data = forecast_future.result()
            indices = indices_future.result()
            air = air_future.result()
            alerts = alerts_future.result()

        daily = data.get("daily") or []
        if not daily:
            raise WeatherError(f"「{location.label}」未来 3 天暂无可用预报数据。")

        return _format_forecast(
            location, str(data.get("updateTime", "未知")), daily, indices, air, alerts
        )
    except WeatherError as exc:
        logger.warning("天气查询失败：city=%s, error=%s", city, exc)
        return f"天气查询失败：{exc}"


def create_travel_agent():
    """创建旅游规划智能体，失败时记日志并返回 None。"""
    if llm is None:
        logger.error("旅游规划智能体创建失败：LLM 未初始化")
        return None

    try:
        agent = create_agent(
            model=llm,
            tools=[get_weather],
            system_prompt=TRAVEL_SYSTEM_PROMPT,
        )
    except Exception:
        logger.exception("旅游规划智能体创建失败：model=%s", llm.model_name)
        return None

    logger.info("旅游规划智能体创建成功：model=%s, tools=%s", llm.model_name, [get_weather.name])
    return agent


def _chunk_text(message: object) -> str:
    """取出消息正文，兼容 content 为字符串或分块列表两种形式（同 agent_health）。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


async def stream_travel_plan(
    question: str,
    profile_context: str = "",
    history: list[dict] | None = None,
) -> AsyncIterator[str]:
    """流式返回行程规划片段，逐段吐出模型最终回答（不含工具调用过程）。

    profile_context 为路由挑好的家庭档案文本块，history 为该会话的历史消息。
    """
    if travel_agent is None:
        yield SERVICE_UNAVAILABLE
        return

    async for chunk, _metadata in travel_agent.astream(
        {"messages": build_agent_messages(question, profile_context, history)},
        stream_mode="messages",
    ):
        # 跳过工具调用产生的片段，只保留最终答案
        if isinstance(chunk, AIMessageChunk) and chunk.content and not chunk.tool_call_chunks:
            yield _chunk_text(chunk)


travel_agent = create_travel_agent()
