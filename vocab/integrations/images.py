"""Safe image-provider and local-cache integration for bot card images."""

import asyncio
import hashlib
import logging
import re
from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from vocab.media_storage import media_storage

IMAGE_CACHE_DIR = media_storage.directory("card_images")
USER_IMAGE_DIR = media_storage.directory("user_images")
logger = logging.getLogger(__name__)


def to_project_relative(path: Path) -> str:
    return media_storage.to_project_relative(path)


def _stable_seed(key: str) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % 1_000_000


def compute_image_cache_path(word_obj: object) -> Path:
    cache_key = f"{getattr(word_obj, 'id', '')}_{getattr(word_obj, 'word', '')}"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", cache_key) or "word"
    return IMAGE_CACHE_DIR / f"{slug}.jpg"


def get_image_queries(word_obj: object) -> list[str]:
    word = getattr(word_obj, "word", "") or ""
    translation = getattr(word_obj, "translation", "") or ""
    example = getattr(word_obj, "example", "") or ""
    part = (getattr(word_obj, "part_of_speech", "") or "").lower()

    queries: list[str] = []
    seen: set[str] = set()

    def add(query: str) -> None:
        query = query.strip()
        if query and query not in seen:
            seen.add(query)
            queries.append(query)

    add(word)
    if translation and translation != word:
        add(translation)
    if word and translation:
        add(f"{word} {translation}")
    if part.startswith("verb"):
        add(f"{word} verb action")
        add(f"{translation} глагол действие")
    elif part.startswith("adjective"):
        add(f"{word} adjective")
        add(f"{translation} прилагательное")
    if example:
        add(" ".join(example.split()[:6]))
    return queries


def get_image_urls(word_obj: object, seed: int = 0) -> list[str]:
    urls: list[str] = []
    for index, query in enumerate(get_image_queries(word_obj)):
        signature = (seed + index * 137) % 1_000_000
        escaped_query = quote_plus(query)
        urls.extend(
            [
                f"https://source.unsplash.com/random/?{escaped_query}&sig={signature}",
                f"https://loremflickr.com/1280/720/{escaped_query}?lock={signature}",
            ]
        )
    urls.append(f"https://picsum.photos/seed/{seed}/1280/720")
    return urls


async def fetch_image_bytes(word_obj: object) -> bytes | None:
    """Read a safe local image or fetch/cache one from the provider fallback list."""
    media_storage.directory("card_images", create=True)
    manual_path = getattr(word_obj, "image_path", "") or ""
    if manual_path:
        resolved = media_storage.resolve_existing(
            manual_path,
            allowed_kinds={"card_images", "user_images"},
        )
        if resolved is not None:
            try:
                return resolved.read_bytes()
            except OSError:
                logger.warning("Failed to read manual image %s", resolved)
        else:
            logger.warning("Rejected unsafe or missing image_path: %s", manual_path)

    cache_key = f"{getattr(word_obj, 'id', '')}_{getattr(word_obj, 'word', '')}"
    seed = _stable_seed(cache_key or (getattr(word_obj, "translation", "") or ""))
    cache_path = compute_image_cache_path(word_obj)
    if cache_path.exists():
        try:
            return cache_path.read_bytes()
        except OSError:
            logger.warning("Failed to read image cache %s, will refetch", cache_path)

    def download(url: str) -> bytes | None:
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(
                request, timeout=8
            ) as response:  # noqa: S310 - fixed providers
                content_type = response.headers.get("Content-Type", "")
                if "image" not in content_type:
                    raise ValueError(f"Unexpected content-type: {content_type}")
                return response.read()
        except Exception as exc:  # noqa: BLE001 - fall through to the next provider
            logger.warning("Image fetch failed for %s: %s", url, exc)
            return None

    for url in get_image_urls(word_obj, seed):
        data = await asyncio.to_thread(download, url)
        if data:
            try:
                cache_path.write_bytes(data)
            except OSError:
                logger.warning("Failed to write image cache %s", cache_path)
            return data
    return None
