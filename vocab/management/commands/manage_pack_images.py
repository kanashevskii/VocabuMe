from __future__ import annotations

from pathlib import Path
import json
import re
import time

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Count, Q

from vocab.models import PackPreparedWord, VocabularyItem
from vocab.openai_utils import CARD_IMAGE_STYLE_PROMPT, generate_image_file
from vocab.services import PROJECT_ROOT

try:
    from PIL import Image
except ImportError:  # pragma: no cover - deployment should include Pillow
    Image = None


CARD_IMAGE_DIR = PROJECT_ROOT / "media" / "card_images"
PACK_ITEM_IMAGE_DIR = CARD_IMAGE_DIR / "pack_items"
SHEET_IMAGE_DIR = CARD_IMAGE_DIR / "pack_contact_sheets"
OLD_MANUAL_IMAGE_PREFIX = "media/card_images/manual_pack_images/"
CONTACT_SHEET_STYLE_PROMPT = """
VocabuMe contact-sheet style:
- One realistic or lightly editorial semi-realistic scene per panel.
- Warm natural light, calm neutral background, and a consistent soft blue/teal and warm-neutral palette.
- Keep the visual subject fully inside its own panel with clear gutters between panels.
- No written text, captions, letters, numbers, UI, logos, watermarks, official seals, or flags as decoration.
- Each panel must be understandable after being cropped to a square mobile flashcard.
""".strip()


def _existing_media_path(image_path: str) -> Path | None:
    if not image_path:
        return None
    resolved = (PROJECT_ROOT / image_path).resolve()
    try:
        resolved.relative_to((PROJECT_ROOT / "media").resolve())
    except ValueError:
        return None
    if not resolved.exists() or resolved.stat().st_size == 0:
        return None
    return resolved


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug[:48] or "item"


def _item_filename(item: PackPreparedWord) -> str:
    slug = _safe_slug(item.normalized_word or item.word)
    return f"{item.id:05d}-{slug}.jpg"


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _duplicate_image_paths(queryset):
    return {
        row["image_path"]
        for row in queryset.exclude(image_path="")
        .order_by()
        .values("image_path")
        .annotate(count=Count("id"))
        .filter(count__gt=1)
    }


def _build_single_prompt(item: PackPreparedWord) -> str:
    return f"""
Create one square image for a VocabuMe vocabulary flashcard.

Word or phrase: {item.word}
Russian meaning: {item.translation}
Part of speech: {item.part_of_speech or "unknown"}
Example context: {item.example or "-"}

Visual goal:
- Make the image specific to this exact word or phrase.
- Show one clear real-world scene, object, action, or symbolic situation.
- Do not include written words, letters, numbers, captions, UI, logos, watermarks, stamps, or document text.

{CARD_IMAGE_STYLE_PROMPT}
""".strip()


def _build_sheet_prompt(items: list[PackPreparedWord], cols: int, rows: int) -> str:
    panel_lines = []
    for index, item in enumerate(items, start=1):
        panel_lines.append(
            f"Panel {index}: {item.word} = {item.translation}. "
            f"Use context: {item.example or item.part_of_speech or 'everyday scene'}."
        )
    empty_count = cols * rows - len(items)
    if empty_count:
        panel_lines.append(
            f"Leave the remaining {empty_count} panel(s) as simple neutral background."
        )

    return f"""
Create a {cols} by {rows} contact sheet for VocabuMe vocabulary flashcards.
Each panel must be visually different and specific to its assigned word or phrase.
Use clean invisible gutters between panels so they can be cropped into individual flashcard images.

Panels:
{chr(10).join(panel_lines)}

Do not put panel numbers or any written text inside the image.
Do not use captions, letters, numbers, UI, logos, watermarks, stamps, document text, collages inside a panel, or split-screen scenes inside a panel.

{CONTACT_SHEET_STYLE_PROMPT}
""".strip()


def _sheet_layout(count: int) -> tuple[int, int, str]:
    if count <= 1:
        return 1, 1, "1024x1024"
    if count <= 4:
        return 2, 2, "1024x1024"
    return 3, 2, "1536x1024"


def _trim_light_border(image):
    if Image is None:
        return image

    source = image.convert("RGB")
    pixels = source.load()
    width, height = source.size
    min_x, min_y = width, height
    max_x, max_y = -1, -1
    for y in range(height):
        for x in range(width):
            red, green, blue = pixels[x, y]
            if min(red, green, blue) < 246:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

    if max_x < min_x or max_y < min_y:
        return source

    pad = max(4, min(width, height) // 80)
    left = max(0, min_x - pad)
    top = max(0, min_y - pad)
    right = min(width, max_x + pad + 1)
    bottom = min(height, max_y + pad + 1)
    if left == 0 and top == 0 and right == width and bottom == height:
        return source

    cropped_area = (right - left) * (bottom - top)
    if cropped_area < width * height * 0.35:
        return source

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return source.crop((left, top, right, bottom)).resize(source.size, resampling)


def _crop_contact_sheet(
    sheet_path: Path, items: list[PackPreparedWord], cols: int, rows: int
) -> list[Path]:
    if Image is None:
        raise CommandError("Pillow is required to crop generated contact sheets.")

    with Image.open(sheet_path) as sheet:
        sheet = sheet.convert("RGB")
        width, height = sheet.size
        cell_width = width // cols
        cell_height = height // rows
        outputs: list[Path] = []
        for index, item in enumerate(items):
            row = index // cols
            col = index % cols
            left = col * cell_width
            top = row * cell_height
            cell = sheet.crop((left, top, left + cell_width, top + cell_height))
            side = min(cell.size)
            crop_left = (cell.width - side) // 2
            crop_top = (cell.height - side) // 2
            square = cell.crop((crop_left, crop_top, crop_left + side, crop_top + side))
            square = _trim_light_border(square)
            destination = (
                PACK_ITEM_IMAGE_DIR
                / item.course_code
                / item.pack_id
                / item.level_id
                / _item_filename(item)
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            square.save(destination, format="JPEG", quality=88, optimize=True)
            outputs.append(destination)
        return outputs


class Command(BaseCommand):
    help = "Audit, generate, and backfill prepared pack images."

    def add_arguments(self, parser):
        parser.add_argument(
            "--course", default="", help="Limit by course code, e.g. en or ka."
        )
        parser.add_argument("--pack", default="", help="Limit by pack_id.")
        parser.add_argument("--level", default="", help="Limit by level_id.")
        parser.add_argument(
            "--limit-items", type=int, default=0, help="Limit generated prepared items."
        )
        parser.add_argument(
            "--generate-unique",
            action="store_true",
            help="Generate unique item-level images for duplicate/missing/broken prepared images.",
        )
        parser.add_argument(
            "--force-all",
            action="store_true",
            help="With --generate-unique, regenerate every prepared item in the selected scope.",
        )
        parser.add_argument(
            "--normalize-existing",
            action="store_true",
            help="Trim light borders from existing generated pack item images.",
        )
        parser.add_argument(
            "--sync-user-images",
            action="store_true",
            help="Sync matching user words to prepared item images when current images are empty, broken, or old manual pack images.",
        )
        parser.add_argument(
            "--export-manifest",
            default="",
            help="Write prepared image paths for the selected scope to this JSON file.",
        )
        parser.add_argument(
            "--import-manifest",
            default="",
            help="Read prepared image paths from this JSON file and apply them by course/pack/level/normalized_word.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print actions without writing DB or generated files.",
        )

    def handle(self, *args, **options):
        queryset = PackPreparedWord.objects.all().order_by(
            "course_code", "pack_id", "level_id", "id"
        )
        if options["course"]:
            queryset = queryset.filter(course_code=options["course"])
        if options["pack"]:
            queryset = queryset.filter(pack_id=options["pack"])
        if options["level"]:
            queryset = queryset.filter(level_id=options["level"])

        items = list(queryset)
        duplicate_paths = _duplicate_image_paths(queryset)
        missing_items = [item for item in items if not item.image_path]
        broken_items = [
            item
            for item in items
            if item.image_path and _existing_media_path(item.image_path) is None
        ]
        duplicate_items = [item for item in items if item.image_path in duplicate_paths]

        self.stdout.write(f"Prepared items: {len(items)}")
        self.stdout.write(f"Missing prepared images: {len(missing_items)}")
        self.stdout.write(f"Broken prepared images: {len(broken_items)}")
        self.stdout.write(f"Duplicate prepared image paths: {len(duplicate_paths)}")
        for row in (
            queryset.exclude(image_path="")
            .order_by()
            .values("image_path")
            .annotate(count=Count("id"))
            .filter(count__gt=1)
            .order_by("-count", "image_path")[:20]
        ):
            self.stdout.write(f"  {row['count']:>3}x {row['image_path']}")

        generated_count = 0
        if options["generate_unique"]:
            candidates = []
            seen_ids = set()
            source_items = (
                items
                if options["force_all"]
                else [*missing_items, *broken_items, *duplicate_items]
            )
            for item in source_items:
                if item.id in seen_ids:
                    continue
                seen_ids.add(item.id)
                candidates.append(item)
            if options["limit_items"]:
                candidates = candidates[: options["limit_items"]]
            self.stdout.write(
                f"Prepared images selected for unique generation: {len(candidates)}"
            )
            if not options["dry_run"]:
                generated_count = self._generate_unique_images(candidates)

        imported_count = 0
        if options["import_manifest"]:
            imported_count = self._import_manifest(
                Path(options["import_manifest"]), dry_run=options["dry_run"]
            )

        synced_count = 0
        if options["sync_user_images"]:
            synced_count = self._sync_user_images(queryset, dry_run=options["dry_run"])

        normalized_count = 0
        if options["normalize_existing"]:
            normalized_count = self._normalize_existing_images(
                queryset, dry_run=options["dry_run"]
            )

        if options["export_manifest"]:
            self._export_manifest(queryset, Path(options["export_manifest"]))

        user_missing = VocabularyItem.objects.filter(
            Q(image_path="") | Q(image_path__isnull=True)
        ).count()
        self.stdout.write(f"User words without image_path: {user_missing}")
        self.stdout.write(
            self.style.SUCCESS(
                f"Done. generated={generated_count}, imported={imported_count}, synced_user_images={synced_count}, normalized={normalized_count}"
            )
        )

    def _generate_unique_images(self, candidates: list[PackPreparedWord]) -> int:
        generated = 0
        for start in range(0, len(candidates), 6):
            chunk = candidates[start : start + 6]
            cols, rows, size = _sheet_layout(len(chunk))
            timestamp = int(time.time())
            sheet_path = (
                SHEET_IMAGE_DIR
                / f"{chunk[0].course_code}_{chunk[0].pack_id}_{chunk[0].level_id}_{start + 1}_{timestamp}.png"
            )

            if len(chunk) == 1:
                prompt = _build_single_prompt(chunk[0])
                destination = (
                    PACK_ITEM_IMAGE_DIR
                    / chunk[0].course_code
                    / chunk[0].pack_id
                    / chunk[0].level_id
                    / _item_filename(chunk[0])
                )
                generate_image_file(
                    prompt, destination, size="1024x1024", quality="low"
                )
                output_paths = [destination]
            else:
                prompt = _build_sheet_prompt(chunk, cols, rows)
                generate_image_file(prompt, sheet_path, size=size, quality="low")
                output_paths = _crop_contact_sheet(sheet_path, chunk, cols, rows)

            for item, output_path in zip(chunk, output_paths, strict=True):
                item.image_path = _relative(output_path)
                item.image_generation_in_progress = False
                item.last_failure_at = None
                item.save(
                    update_fields=[
                        "image_path",
                        "image_generation_in_progress",
                        "last_failure_at",
                        "prepared_at",
                    ]
                )
                generated += 1
                self.stdout.write(
                    f"Generated {item.course_code}/{item.pack_id}/{item.level_id}/{item.word}: {item.image_path}"
                )
        return generated

    def _sync_user_images(self, queryset, *, dry_run: bool) -> int:
        prepared_items = list(
            queryset.exclude(image_path="").order_by(
                "course_code", "pack_id", "level_id", "id"
            )
        )
        synced = 0
        for prepared in prepared_items:
            if _existing_media_path(prepared.image_path) is None:
                continue
            matches = VocabularyItem.objects.filter(
                course_code=prepared.course_code,
                normalized_word=prepared.normalized_word,
            ).filter(Q(translation__iexact=prepared.translation) | Q(image_path=""))
            for item in matches:
                current_is_old_manual = item.image_path.startswith(
                    OLD_MANUAL_IMAGE_PREFIX
                )
                current_is_missing_or_broken = (
                    _existing_media_path(item.image_path) is None
                )
                if item.image_path == prepared.image_path:
                    continue
                if not current_is_missing_or_broken and not current_is_old_manual:
                    continue
                synced += 1
                if not dry_run:
                    item.image_path = prepared.image_path
                    item.image_generation_in_progress = False
                    item.save(
                        update_fields=[
                            "image_path",
                            "image_generation_in_progress",
                            "updated_at",
                        ]
                    )
        return synced

    def _export_manifest(self, queryset, path: Path) -> None:
        payload = [
            {
                "course_code": item.course_code,
                "pack_id": item.pack_id,
                "level_id": item.level_id,
                "normalized_word": item.normalized_word,
                "word": item.word,
                "translation": item.translation,
                "image_path": item.image_path,
            }
            for item in queryset.exclude(image_path="").order_by(
                "course_code", "pack_id", "level_id", "normalized_word"
            )
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.stdout.write(f"Exported manifest: {path} ({len(payload)} rows)")

    def _import_manifest(self, path: Path, *, dry_run: bool) -> int:
        if not path.exists():
            raise CommandError(f"Manifest not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        imported = 0
        for row in payload:
            image_path = row.get("image_path") or ""
            if not image_path or _existing_media_path(image_path) is None:
                continue
            updated = PackPreparedWord.objects.filter(
                course_code=row.get("course_code"),
                pack_id=row.get("pack_id"),
                level_id=row.get("level_id"),
                normalized_word=row.get("normalized_word"),
            ).exclude(image_path=image_path)
            count = updated.count()
            if count and not dry_run:
                updated.update(
                    image_path=image_path,
                    image_generation_in_progress=False,
                    last_failure_at=None,
                )
            imported += count
        return imported

    def _normalize_existing_images(self, queryset, *, dry_run: bool) -> int:
        if Image is None:
            raise CommandError("Pillow is required to normalize generated images.")

        normalized = 0
        for item in queryset.exclude(image_path=""):
            if not item.image_path.startswith("media/card_images/pack_items/"):
                continue
            path = _existing_media_path(item.image_path)
            if path is None:
                continue
            with Image.open(path) as image:
                original = image.convert("RGB")
                trimmed = _trim_light_border(original)
                if (
                    trimmed.size == original.size
                    and trimmed.tobytes() == original.tobytes()
                ):
                    continue
                normalized += 1
                if not dry_run:
                    trimmed.save(path, format="JPEG", quality=88, optimize=True)
        return normalized
