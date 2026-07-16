"""HTTP endpoints for server-authoritative learning sessions."""

from __future__ import annotations

from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_GET, require_POST

from vocab.analytics import record_product_event
from vocab.api.common import json_body, json_error, require_user
from vocab.services import (
    get_ordered_unlearned_words,
    get_word_image_file,
    issue_learning_question,
    list_words,
    serialize_word,
    submit_issued_learning_answer,
)


@require_POST
def learn_question(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user

    try:
        payload = json_body(request)
    except ValueError as exc:
        return json_error(str(exc))
    raw_exclude = payload.get("exclude_ids", [])
    if not isinstance(raw_exclude, list):
        return json_error("exclude_ids must be an array.")
    exclude_ids: list[int] = []
    for chunk in raw_exclude[:50]:
        try:
            exclude_ids.append(int(chunk))
        except (TypeError, ValueError):
            continue

    question = issue_learning_question(user, exclude_ids=exclude_ids)
    if question is None:
        return JsonResponse(
            {
                "ok": True,
                "empty": True,
                "session_limit": max(1, min(user.session_question_limit, 50)),
            }
        )
    return JsonResponse(
        {
            "ok": True,
            "question": question,
            "session_limit": max(1, min(user.session_question_limit, 50)),
        }
    )


@require_POST
def learn_answer(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user

    try:
        payload = json_body(request)
        question_id = str(payload.get("question_id", ""))
        answer = str(payload.get("answer", ""))
    except (TypeError, ValueError):
        return json_error("Invalid learning answer payload.")

    try:
        result = submit_issued_learning_answer(user, question_id, answer)
    except ValueError as exc:
        return json_error(str(exc))
    record_product_event(
        user,
        "practice_completed",
        properties={"correct": bool(result.get("correct"))},
    )
    return JsonResponse({"ok": True, **result})


@require_GET
def study_cards(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user

    scope = request.GET.get("scope", "").strip().lower()
    if scope == "all":
        items = list_words(user, status="learning", limit=500)
        with_images = [item for item in items if get_word_image_file(item) is not None]
        without_images = [item for item in items if get_word_image_file(item) is None]
        cards = [serialize_word(item) for item in with_images + without_images]
    else:
        count = max(1, min(int(request.GET.get("count", 10)), 20))
        cards = [
            serialize_word(item)
            for item in get_ordered_unlearned_words(user, count=count)
        ]
    return JsonResponse({"ok": True, "items": cards})


def _retired_learning_endpoint(message: str) -> JsonResponse:
    return json_error(message, status=410)


@require_GET
def practice_question(request: HttpRequest) -> JsonResponse:
    return _retired_learning_endpoint(
        "This endpoint is retired. Use /api/learn/question instead."
    )


@require_POST
def practice_answer(request: HttpRequest) -> JsonResponse:
    return _retired_learning_endpoint(
        "This endpoint is retired. Submit a server-issued learning question instead."
    )


@require_GET
def listening_question(request: HttpRequest) -> JsonResponse:
    return _retired_learning_endpoint(
        "This endpoint is retired. Use /api/learn/question instead."
    )


@require_POST
def listening_answer(request: HttpRequest) -> JsonResponse:
    return _retired_learning_endpoint(
        "This endpoint is retired. Submit a server-issued learning question instead."
    )


@require_POST
def study_answer(request: HttpRequest) -> JsonResponse:
    return _retired_learning_endpoint(
        "This endpoint is retired. Submit a server-issued learning question instead."
    )
