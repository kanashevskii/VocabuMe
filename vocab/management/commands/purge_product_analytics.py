from django.core.management.base import BaseCommand

from vocab.analytics import EVENT_RETENTION_DAYS, purge_expired_product_events


class Command(BaseCommand):
    help = "Delete first-party product analytics events past the retention window."

    def handle(self, *args, **options):
        deleted = purge_expired_product_events()
        self.stdout.write(
            self.style.SUCCESS(
                f"Deleted {deleted} product analytics events older than {EVENT_RETENTION_DAYS} days."
            )
        )
