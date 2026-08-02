"""Local-media adapter for profile avatar storage."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from uuid import uuid4

from django.utils import timezone

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - Pillow may be absent in some envs
    Image = None
    ImageOps = None

from vocab.models import TelegramUser

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_AVATAR_DIR = PROJECT_ROOT / "media" / "profile_avatars"
MAX_AVATAR_BYTES = 5 * 1024 * 1024
ALLOWED_AVATAR_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _preferred_served_image(source: Path) -> Path:
    if source.suffix.lower() == ".webp":
        return source
    webp = source.with_suffix(".webp")
    if webp.exists() and webp.stat().st_size > 0:
        return webp
    if webp.exists():
        try:
            webp.unlink()
        except OSError:
            pass
    return source


def get_profile_avatar_file(user: TelegramUser) -> Path | None:
    """Return a verified avatar path or ``None`` for a stale or unsafe value."""
    if not user.avatar_path:
        return None
    raw_path = Path(user.avatar_path)
    candidate = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None
    if not resolved.is_relative_to(PROFILE_AVATAR_DIR.resolve()):
        return None
    return _preferred_served_image(resolved)


def _remove_user_avatar_file(user: TelegramUser) -> None:
    avatar_file = get_profile_avatar_file(user)
    if avatar_file is None:
        return
    try:
        avatar_file.unlink(missing_ok=True)
    except OSError:
        logger.warning("Failed to delete avatar file %s", avatar_file)


def save_user_avatar(user: TelegramUser, uploaded_file) -> TelegramUser:
    content_type = (getattr(uploaded_file, "content_type", "") or "").lower()
    if content_type not in ALLOWED_AVATAR_CONTENT_TYPES:
        raise ValueError("Разрешены только JPG, PNG или WEBP.")
    if getattr(uploaded_file, "size", 0) > MAX_AVATAR_BYTES:
        raise ValueError("Файл слишком большой. Максимум 5 MB.")
    if Image is None:
        raise ValueError("Обработка изображений временно недоступна.")

    PROFILE_AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    previous_avatar_file = get_profile_avatar_file(user)
    output_path = PROFILE_AVATAR_DIR / f"user_{user.id}.webp"
    temporary_path = PROFILE_AVATAR_DIR / f".{output_path.name}.{uuid4().hex}.tmp"
    try:
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            image.thumbnail((512, 512))
            image.save(temporary_path, format="WEBP", quality=84, method=6)
        os.replace(temporary_path, output_path)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise ValueError("Не удалось обработать изображение.") from exc

    user.avatar_path = str(output_path.relative_to(PROJECT_ROOT))
    user.avatar_updated_at = timezone.now()
    user.custom_avatar_url = ""
    user.save(update_fields=["avatar_path", "avatar_updated_at", "custom_avatar_url"])
    if previous_avatar_file and previous_avatar_file != output_path:
        try:
            previous_avatar_file.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "Failed to delete replaced avatar file %s", previous_avatar_file
            )
    return user


def delete_user_avatar(user: TelegramUser) -> TelegramUser:
    _remove_user_avatar_file(user)
    user.avatar_path = ""
    user.avatar_updated_at = None
    user.save(update_fields=["avatar_path", "avatar_updated_at"])
    return user
