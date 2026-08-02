"""Filesystem-backed card-media adapter behind the ``media_storage`` boundary."""

from __future__ import annotations

from pathlib import Path
import logging
import re

from vocab.jobs import enqueue_job
from vocab.media_storage import media_storage
from vocab.models import AddWordDraft, VocabularyItem

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow may be absent in some envs
    Image = None

logger = logging.getLogger(__name__)

IMAGE_CACHE_DIR = media_storage.directory("card_images")
USER_IMAGE_DIR = media_storage.directory("user_images")
DRAFT_IMAGE_DIR = media_storage.directory("draft_images")
CARD_MEDIA_KINDS = {"card_images", "user_images", "draft_images"}


def _webp_variant_path(source: Path) -> Path:
    return source.with_suffix(".webp")


def optimize_image_to_webp(source: Path) -> Path | None:
    """Create an up-to-date WebP sibling without modifying the source image."""
    if Image is None or source.suffix.lower() == ".webp":
        return source if source.exists() else None

    target = _webp_variant_path(source)
    try:
        if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
            return target
    except OSError:
        pass

    try:
        with Image.open(source) as image:
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            image.save(target, format="WEBP", quality=82, method=6)
        return target
    except Exception:
        return None


def schedule_image_optimization(source: Path) -> None:
    if Image is None or source.suffix.lower() == ".webp" or not source.exists():
        return
    enqueue_job(
        kind="image_optimize",
        deduplication_key=f"image-optimize:{source}:{int(source.stat().st_mtime)}",
        payload={"source_path": str(source)},
        priority=200,
    )


def preferred_served_image(source: Path) -> Path:
    if source.suffix.lower() == ".webp":
        return source
    webp = _webp_variant_path(source)
    if webp.exists() and webp.stat().st_size > 0:
        return webp
    if webp.exists():
        try:
            webp.unlink()
        except OSError:
            pass
    return source


def _resolve_card_media(reference: str | Path) -> Path | None:
    return media_storage.resolve_existing(reference, allowed_kinds=CARD_MEDIA_KINDS)


def get_word_image_file(item: VocabularyItem) -> Path | None:
    if item.image_path:
        resolved = _resolve_card_media(item.image_path)
        if resolved is not None:
            return preferred_served_image(resolved)

    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", item.word or "").strip("_") or "word"
    for candidate in (
        IMAGE_CACHE_DIR / f"{item.id}_{slug}.jpg",
        IMAGE_CACHE_DIR / f"_{slug}.jpg",
    ):
        resolved = _resolve_card_media(candidate)
        if resolved is not None:
            return preferred_served_image(resolved)
    return None


def get_draft_image_file(draft: AddWordDraft) -> Path | None:
    if not draft.image_path:
        return None
    resolved = _resolve_card_media(draft.image_path)
    return preferred_served_image(resolved) if resolved is not None else None
