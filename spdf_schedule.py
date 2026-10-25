"""Daily DeepThink admission policy; independent of server local timezone."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


BEIJING_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")


class SpdfAccessDenied(RuntimeError):
    """An expected access-policy rejection, not a model/compilation failure."""


@dataclass(frozen=True)
class SpdfAdmission:
    # Internal, request-bound ticket: keep admitted jobs running past closing.
    owner: object
    event: object


def _minute(value):
    text = str(value).strip()
    if not re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", text):
        raise ValueError("time must be HH:MM")
    hour, minute = map(int, text.split(":"))
    return hour * 60 + minute


def spdf_access_denial(cfg, uid, now=None):
    """Return a user-facing reason, or an empty string if admission is allowed."""
    if not cfg("enable_spdf_output", True):
        return "SPDF 输出未启用"
    if not cfg("spdf_time_limit_enabled", False):
        return ""

    exempt = cfg("spdf_time_limit_exempt_qq", []) or []
    if isinstance(exempt, str):
        exempt = re.split(r"[,，;；\s]+", exempt.strip())
    if not isinstance(exempt, (list, tuple, set)):
        exempt = []
    uid = str(uid).strip()
    if re.fullmatch(r"[0-9]+", uid) and uid in {str(x).strip() for x in exempt}:
        return ""

    start_text = str(cfg("spdf_allowed_start", "08:00")).strip()
    end_text = str(cfg("spdf_allowed_end", "23:00")).strip()
    try:
        start, end = _minute(start_text), _minute(end_text)
    except ValueError:
        return "⚠️ /spdf 开放时段配置有误，请管理员在插件 WebUI 中填写 24 小时制 HH:MM（例如 08:00、23:00）。/pdf 不受此限制。"

    if now is None:
        now = datetime.now(BEIJING_TZ)
    elif now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    else:
        now = now.astimezone(BEIJING_TZ)
    current = now.hour * 60 + now.minute
    allowed = (start == end or
               (start <= current < end if start < end else current >= start or current < end))
    if allowed:
        return ""
    overnight = "（跨午夜）" if start > end else ""
    return (f"⏰ /spdf 当前不在开放时段。\n"
            f"每天开放：{start_text}–{end_text}{overnight}。")
