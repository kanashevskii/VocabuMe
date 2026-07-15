import re
import string
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

PUNCTUATION = string.punctuation.replace("'", "")
_punct_regex = re.compile(f"[{re.escape(PUNCTUATION)}]")


def clean_word(word: str) -> str:
    """Return word lowercased with punctuation removed (except apostrophes)."""
    if not isinstance(word, str):
        return ""
    cleaned = _punct_regex.sub("", word)
    cleaned = cleaned.strip().lower()
    return cleaned


def translate_to_ru(text: str) -> str:
    """Translate text through Google Translate's public endpoint.

    This is a best-effort enrichment path: callers must tolerate an empty value.
    """
    value = text.strip()
    if not value:
        return ""
    try:
        response = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": "auto", "tl": "ru", "dt": "t", "q": value},
            timeout=5,
        )
        response.raise_for_status()
        payload = response.json()
        segments = payload[0] if isinstance(payload, list) and payload else []
        translation = "".join(
            segment[0]
            for segment in segments
            if isinstance(segment, list) and segment and isinstance(segment[0], str)
        )
        return translation.strip()
    except (requests.RequestException, TypeError, ValueError):
        return ""


def normalize_timezone_value(raw: str) -> str:
    """
    Приводит ввод пользователя к строке таймзоны.
    Поддерживает IANA-имена (Europe/Moscow) и смещения UTC±HH[:MM].
    """
    value = (raw or "").strip()
    if not value:
        raise ValueError("Empty timezone")

    value = value.replace(" ", "")
    upper = value.upper()

    # Форматы вида "+3", "-05", "UTC+3", "GMT-3", "UTC+03:30"
    offset_match = re.fullmatch(r"(?:UTC|GMT)?([+-]?\d{1,2})(?::?(\d{2}))?", upper)
    if offset_match:
        hours = int(offset_match.group(1))
        minutes = int(offset_match.group(2) or 0)
        if not (-14 <= hours <= 14) or not (0 <= minutes < 60):
            raise ValueError("Invalid offset range")
        sign = 1 if hours >= 0 else -1
        hours = abs(hours)
        prefix = "UTC+" if sign >= 0 else "UTC-"
        return f"{prefix}{hours:02d}:{minutes:02d}"

    # Имена зон, например Europe/Moscow
    try:
        ZoneInfo(value)
        return value
    except Exception as exc:
        raise ValueError(f"Unknown timezone: {value}") from exc


def timezone_from_name(name: str):
    """Возвращает tzinfo по сохранённой строке."""
    if not name:
        return timezone.utc

    upper = name.upper()
    if upper.startswith("UTC") or upper.startswith("GMT"):
        match = re.fullmatch(r"(?:UTC|GMT)([+-])(\d{2}):(\d{2})", upper)
        if match:
            sign = 1 if match.group(1) == "+" else -1
            hours = int(match.group(2))
            minutes = int(match.group(3))
            return timezone(sign * timedelta(hours=hours, minutes=minutes))
        # fallback to plain UTC
        return timezone.utc

    try:
        return ZoneInfo(name)
    except Exception:
        return timezone.utc


def format_timezone_short(name: str) -> str:
    """Возвращает короткое текстовое представление таймзоны/смещения."""
    if not name:
        return "UTC"
    if name.upper().startswith(("UTC", "GMT")):
        return name.upper()
    try:
        tz = ZoneInfo(name)
        now_local = datetime.now(tz)
        offset = now_local.utcoffset() or timedelta(0)
        total_minutes = int(offset.total_seconds() // 60)
        sign = "+" if total_minutes >= 0 else "-"
        total_minutes = abs(total_minutes)
        hours, minutes = divmod(total_minutes, 60)
        return f"{name} (UTC{sign}{hours:02d}:{minutes:02d})"
    except Exception:
        return name
