from django.core.management.base import BaseCommand, CommandError

from vocab.analytics import build_funnel_report


class Command(BaseCommand):
    help = "Print the first-party product analytics funnel for the requested period."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30)

    def handle(self, *args, **options):
        try:
            report = build_funnel_report(days=options["days"])
        except ValueError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(f"Product funnel, last {report['days']} days since {report['since']}")
        for name, metrics in report["events"].items():
            self.stdout.write(
                f"{name}: {metrics['users']} users, {metrics['events']} events"
            )
