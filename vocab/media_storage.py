"""Media storage boundary with a backwards-compatible local filesystem adapter.

Database records currently contain project-relative paths such as
``media/card_images/example.jpg``.  The adapter preserves that representation
so a future object-storage migration can use dual-read/copy/cutover without
rewriting user records in place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEDIA_ROOT = PROJECT_ROOT / "media"
MEDIA_KINDS = frozenset(
    {"card_images", "user_images", "draft_images", "profile_avatars"}
)


class MediaStorage(Protocol):
    def directory(self, kind: str, *, create: bool = False) -> Path: ...

    def resolve_existing(
        self, reference: str | Path, *, allowed_kinds: set[str] | frozenset[str]
    ) -> Path | None: ...

    def to_project_relative(self, path: Path) -> str: ...


class LocalMediaStorage:
    """Filesystem implementation that rejects paths outside approved media roots."""

    def directory(self, kind: str, *, create: bool = False) -> Path:
        if kind not in MEDIA_KINDS:
            raise ValueError(f"Unsupported media kind: {kind}")
        directory = MEDIA_ROOT / kind
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    def resolve_existing(
        self, reference: str | Path, *, allowed_kinds: set[str] | frozenset[str]
    ) -> Path | None:
        if not allowed_kinds or not set(allowed_kinds).issubset(MEDIA_KINDS):
            raise ValueError("Invalid allowed media roots.")
        candidate = Path(reference)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            return None
        allowed_roots = tuple(
            self.directory(kind).resolve() for kind in sorted(allowed_kinds)
        )
        return (
            resolved
            if any(resolved.is_relative_to(root) for root in allowed_roots)
            else None
        )

    def to_project_relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)


media_storage: MediaStorage = LocalMediaStorage()
