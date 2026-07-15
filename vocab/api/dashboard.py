"""Read-only dashboard projection for an authenticated learner."""

from __future__ import annotations

from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_GET

from vocab.api.common import require_user
from vocab.services import (
    build_user_progress,
    get_billing_payload,
    get_ordered_unlearned_words,
    list_words,
    serialize_user,
    serialize_word,
)


@require_GET
def dashboard(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    stats = build_user_progress(user)
    recent_words = [serialize_word(item) for item in list_words(user, limit=6)]
    next_cards = [
        serialize_word(item) for item in get_ordered_unlearned_words(user, count=4)
    ]
    return JsonResponse(
        {
            "ok": True,
            "user": serialize_user(user),
            "progress": stats,
            "billing": get_billing_payload(user),
            "recent_words": recent_words,
            "next_cards": next_cards,
        }
    )
