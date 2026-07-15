from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from vocab.models import AppErrorLog


class Command(BaseCommand):
    help = "Delete application error logs older than the configured retention period."

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=30,
            help="Keep this many days of logs (default: 30).",
        )

    def handle(self, *args, **options):
        days = options["days"]
        if days < 1:
            raise CommandError("--days must be at least 1.")
        cutoff = timezone.now() - timedelta(days=days)
        deleted_count, _ = AppErrorLog.objects.filter(created_at__lt=cutoff).delete()
        self.stdout.write(
            self.style.SUCCESS(
                f"Deleted {deleted_count} application error log(s) older than {cutoff.isoformat()}."
            )
        )
