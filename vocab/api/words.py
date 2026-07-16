"""Personal dictionary and word-draft API endpoints."""

from __future__ import annotations

import logging
import traceback

from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_http_methods, require_POST

from vocab.analytics import record_product_event
from vocab.api.common import enforce_request_limit, json_body, json_error, require_user
from vocab.api.errors import log_app_error
from vocab.models import AddWordDraft, VocabularyItem
from vocab.services import (
    EntitlementError,
    add_words_from_text,
    build_user_progress,
    create_word_drafts_from_text,
    delete_user_draft,
    delete_word,
    finalize_word_draft,
    get_billing_payload,
    get_user_draft,
    get_user_word,
    list_words,
    refresh_draft_language_data,
    request_draft_image_generation,
    request_word_image_generation,
    serialize_draft,
    serialize_word,
    update_word_translation,
    word_already_exists,
)

logger = logging.getLogger(__name__)
MAX_ADD_BATCH_WORDS = 10


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


def _get_draft_for_user(user, draft_id: int) -> AddWordDraft | JsonResponse:
    draft = get_user_draft(user, draft_id)
    return draft if draft is not None else json_error("Draft not found.", status=404)


@require_http_methods(["GET", "POST"])
def words(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user

    if request.method == "GET":
        search = request.GET.get("search", "").strip()
        status = request.GET.get("status", "all")
        items = [
            serialize_word(item)
            for item in list_words(user, search=search, status=status, limit=150)
        ]
        return JsonResponse({"ok": True, "items": items})

    if limited := enforce_request_limit(
        request, scope="word-generation", limit=10, window=60, user=user
    ):
        return limited
    try:
        payload = json_body(request)
    except ValueError as exc:
        return json_error(str(exc))

    try:
        result = add_words_from_text(
            user, payload.get("text", ""), max_batch_words=MAX_ADD_BATCH_WORDS
        )
    except EntitlementError as exc:
        return _json_entitlement_error(user, exc)
    except ValueError as exc:
        return json_error(str(exc), status=400)

    if result["created"]:
        record_product_event(
            user,
            "words_created",
            properties={"count": len(result["created"]), "source": "direct_add"},
        )

    return JsonResponse(
        {
            "ok": True,
            "created": [serialize_word(item) for item in result["created"]],
            "skipped": result["skipped"],
            "failed": result["failed"],
        }
    )


@require_POST
def word_draft_create(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := enforce_request_limit(
        request, scope="word-draft-generation", limit=10, window=60, user=user
    ):
        return limited
    try:
        payload = json_body(request)
    except ValueError as exc:
        return json_error(str(exc))

    try:
        result = create_word_drafts_from_text(
            user, payload.get("text", ""), max_batch_words=MAX_ADD_BATCH_WORDS
        )
        if result["mode"] == "batch_review":
            return JsonResponse(
                {
                    "ok": True,
                    "batch_review": True,
                    "drafts": [serialize_draft(draft) for draft in result["drafts"]],
                    "skipped": result["skipped"],
                    "failed": result["failed"],
                    "progress": build_user_progress(user),
                }
            )
        if result["mode"] == "auto_saved":
            return JsonResponse(
                {
                    "ok": True,
                    "auto_saved": True,
                    "item": serialize_word(result["item"]),
                    "progress": build_user_progress(user),
                }
            )
        return JsonResponse(
            {
                "ok": True,
                "draft": serialize_draft(result["draft"]),
                "step": result["step"],
            }
        )
    except EntitlementError as exc:
        return _json_entitlement_error(user, exc)
    except ValueError as exc:
        return json_error(str(exc), status=400)
    except Exception as exc:
        logger.exception("word_draft_create failed")
        log_app_error(
            request,
            user=user,
            category="add_word",
            status_code=500,
            message=f"word_draft_create failed: {exc}",
            context={
                "text": (payload.get("text", "") or "")[:500],
                "traceback": traceback.format_exc()[-4000:],
            },
        )
        return json_error("Не удалось подготовить слово. Попробуй ещё раз.", status=500)


@require_POST
def word_draft_confirm_translation(request: HttpRequest, draft_id: int) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    draft = _get_draft_for_user(user, draft_id)
    if isinstance(draft, JsonResponse):
        return draft
    try:
        payload = json_body(request)
    except ValueError as exc:
        return json_error(str(exc))
    translation = (payload.get("translation") or "").strip()
    if not translation:
        return json_error("Translation is required.", status=400)
    draft = refresh_draft_language_data(draft, translation)
    draft = request_draft_image_generation(draft)
    return JsonResponse(
        {"ok": True, "draft": serialize_draft(draft), "step": "confirm_image"}
    )


@require_POST
def word_draft_regenerate_image(request: HttpRequest, draft_id: int) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := enforce_request_limit(
        request, scope="image-regeneration", limit=5, window=60, user=user
    ):
        return limited
    draft = _get_draft_for_user(user, draft_id)
    if isinstance(draft, JsonResponse):
        return draft
    if not draft.translation_confirmed:
        return json_error("Confirm translation first.", status=400)
    try:
        draft = request_draft_image_generation(draft, force_regenerate=True)
    except EntitlementError as exc:
        return _json_entitlement_error(user, exc)
    return JsonResponse(
        {"ok": True, "draft": serialize_draft(draft), "step": "confirm_image"}
    )


@require_POST
def word_draft_save(request: HttpRequest, draft_id: int) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    draft = _get_draft_for_user(user, draft_id)
    if isinstance(draft, JsonResponse):
        return draft
    if not draft.translation_confirmed:
        return json_error("Confirm translation first.", status=400)
    if word_already_exists(user, draft.word):
        delete_user_draft(user, draft.id)
        return json_error("This word already exists.", status=400)
    try:
        payload = json_body(request)
    except ValueError as exc:
        return json_error(str(exc))
    try:
        item = finalize_word_draft(
            draft, use_image=bool(payload.get("use_image", True))
        )
    except EntitlementError as exc:
        return _json_entitlement_error(user, exc)
    return JsonResponse(
        {
            "ok": True,
            "item": serialize_word(item),
            "progress": build_user_progress(user),
        }
    )


@require_http_methods(["GET", "DELETE"])
def word_draft_delete(request: HttpRequest, draft_id: int) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    draft = _get_draft_for_user(user, draft_id)
    if isinstance(draft, JsonResponse):
        return draft
    if request.method == "GET":
        return JsonResponse({"ok": True, "draft": serialize_draft(draft)})
    delete_user_draft(user, draft.id)
    return JsonResponse({"ok": True})


@require_http_methods(["PATCH", "DELETE"])
def word_detail(request: HttpRequest, word_id: int) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    try:
        item = get_user_word(user, word_id)
        if item is None:
            raise VocabularyItem.DoesNotExist
    except VocabularyItem.DoesNotExist:
        return json_error("Word not found.", status=404)
    if request.method == "DELETE":
        delete_word(user, word_id)
        return JsonResponse({"ok": True})
    try:
        payload = json_body(request)
    except ValueError as exc:
        return json_error(str(exc))
    translation = (payload.get("translation") or "").strip()
    if not translation:
        return json_error("Translation is required.")
    item = update_word_translation(user, word_id, translation)
    return JsonResponse({"ok": True, "item": serialize_word(item)})


@require_POST
def word_image_regenerate(request: HttpRequest, word_id: int) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := enforce_request_limit(
        request, scope="image-regeneration", limit=5, window=60, user=user
    ):
        return limited
    item = get_user_word(user, word_id)
    if item is None:
        return json_error("Word not found.", status=404)
    try:
        item = request_word_image_generation(item, force_regenerate=True)
    except EntitlementError as exc:
        return _json_entitlement_error(user, exc)
    return JsonResponse({"ok": True, "item": serialize_word(item)})
