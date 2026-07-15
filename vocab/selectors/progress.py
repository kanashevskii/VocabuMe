"""Database-backed selectors for progress read models."""

from django.db.models import Count, Q

from vocab.models import TelegramUser


def get_rank_percent(*, course_code: str, learned_count: int) -> int | None:
    """Return a user's percentile without materializing every user in Python."""
    users_with_progress = TelegramUser.objects.annotate(
        learned_count=Count(
            "vocabularyitem",
            filter=Q(
                vocabularyitem__course_code=course_code,
                vocabularyitem__is_learned=True,
            ),
        )
    )
    total_users = users_with_progress.count()
    if not total_users:
        return None

    better_than = users_with_progress.filter(learned_count__lt=learned_count).count()
    return round(100 * (1 - better_than / total_users))
