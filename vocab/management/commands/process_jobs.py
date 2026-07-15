from __future__ import annotations

import time

from django.core.management.base import BaseCommand

from vocab.jobs import run_one_job


class Command(BaseCommand):
    help = "Process durable VocabuMe background jobs outside the web process."

    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true")
        parser.add_argument("--poll-seconds", type=float, default=1.0)

    def handle(self, *args, **options):
        while True:
            processed = run_one_job()
            if options["once"]:
                return
            if not processed:
                time.sleep(max(0.1, options["poll_seconds"]))
