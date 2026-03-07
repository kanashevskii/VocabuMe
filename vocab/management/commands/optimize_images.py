from pathlib import Path

from django.core.management.base import BaseCommand

from vocab.services import DRAFT_IMAGE_DIR, IMAGE_CACHE_DIR, USER_IMAGE_DIR, _optimize_image_to_webp


class Command(BaseCommand):
    help = "Pre-generate optimized WebP variants for existing card images."

    def handle(self, *args, **options):
        roots = [IMAGE_CACHE_DIR, USER_IMAGE_DIR, DRAFT_IMAGE_DIR]
        source_exts = {".jpg", ".jpeg", ".png"}
        total = 0
        created = 0
        skipped = 0

        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in source_exts:
                    continue
                total += 1
                target = path.with_suffix(".webp")
                if target.exists() and target.stat().st_mtime >= path.stat().st_mtime:
                    skipped += 1
                    continue
                result = _optimize_image_to_webp(path)
                if isinstance(result, Path) and result.exists() and result.suffix.lower() == ".webp":
                    created += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Scanned {total} images, created/updated {created} WebP variants, skipped {skipped}."
            )
        )
