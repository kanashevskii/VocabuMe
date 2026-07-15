from datetime import timedelta

import pytest
from django.core.management import call_command
from django.utils import timezone

from vocab.models import AppErrorLog


@pytest.mark.django_db
def test_purge_error_logs_only_deletes_records_past_the_retention_cutoff():
    stale = AppErrorLog.objects.create(message="stale")
    fresh = AppErrorLog.objects.create(message="fresh")
    AppErrorLog.objects.filter(pk=stale.pk).update(
        created_at=timezone.now() - timedelta(days=31)
    )

    call_command("purge_error_logs", days=30)

    assert AppErrorLog.objects.filter(pk=stale.pk).exists() is False
    assert AppErrorLog.objects.filter(pk=fresh.pk).exists() is True
