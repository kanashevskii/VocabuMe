"""Word-pack catalog, preparation, and add-to-dictionary endpoints."""

from __future__ import annotations

from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_GET, require_POST

from vocab.api.common import enforce_request_limit, json_body, json_error, require_user
from vocab.services import (
    EntitlementError,
    add_pack_words_to_user,
    build_user_progress,
    ensure_pack_preparation,
    get_active_course_code,
    get_billing_payload,
    list_word_packs,
)


def _json_entitlement_error(
    user, exc: EntitlementError, status: int = 402
) -> JsonResponse:
    return json_error(
        exc.message,
        status=status,
        code=exc.code,
        paywall_trigger=exc.paywall_trigger,
        billing=get_billing_payload(user),
    )


@require_GET
def packs_view(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    return JsonResponse({"ok": True, "packs": list_word_packs(user)})


@require_POST
def packs_prepare(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := enforce_request_limit(
        request, scope="pack-preparation", limit=2, window=60, user=user
    ):
        return limited
    active_course = get_active_course_code(user)
    for pack in list_word_packs(user):
        for level in pack["levels"]:
            ensure_pack_preparation(pack["id"], level["id"], course_code=active_course)
    return JsonResponse({"ok": True})


@require_POST
def packs_add(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := enforce_request_limit(
        request, scope="pack-add", limit=5, window=60, user=user
    ):
        return limited
    try:
        payload = json_body(request)
    except ValueError as exc:
        return json_error(str(exc))

    pack_id = str(payload.get("pack_id", "")).strip()
    level_id = str(payload.get("level_id", "")).strip()
    selected_words = payload.get("selected_words") or []
    if not isinstance(selected_words, list):
        return json_error("selected_words must be a list.")

    try:
        result = add_pack_words_to_user(
            user, pack_id, level_id, [str(word) for word in selected_words]
        )
    except EntitlementError as exc:
        return _json_entitlement_error(user, exc)
    except ValueError as exc:
        return json_error(str(exc))

    return JsonResponse(
        {
            "ok": True,
            **result,
            "progress": build_user_progress(user),
            "packs": list_word_packs(user),
        }
    )
