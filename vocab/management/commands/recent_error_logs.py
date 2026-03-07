from django.core.management.base import BaseCommand

from vocab.models import AppErrorLog


class Command(BaseCommand):
    help = "Show recent app error logs."

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=50)

    def handle(self, *args, **options):
        limit = max(1, options["limit"])
        logs = AppErrorLog.objects.select_related("user")[:limit]
        if not logs:
            self.stdout.write("No app error logs.")
            return

        for item in logs:
            user_label = item.user.username if item.user and item.user.username else (item.user.chat_id if item.user else "-")
            self.stdout.write(
                f"[{item.created_at.isoformat()}] {item.category}/{item.level} "
                f"{item.status_code or '-'} {item.method} {item.path} user={user_label}\n"
                f"{item.message}\n"
            )
