"""Server-authoritative irregular-verb practice API."""

from __future__ import annotations

from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_GET, require_POST

from vocab.api.common import enforce_request_limit, json_body, json_error, require_user
from vocab.application.irregular_questions import (
    issue_irregular_question,
    submit_issued_irregular_answer,
)
from vocab.services import list_irregular_page


@require_GET
def irregular_list(request: HttpRequest) -> JsonResponse:
    page = max(0, int(request.GET.get("page", 0)))
    return JsonResponse({"ok": True, **list_irregular_page(page)})


@require_GET
def irregular_question(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    return JsonResponse({"ok": True, "question": issue_irregular_question(user)})


@require_POST
def irregular_answer(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := enforce_request_limit(
        request, scope="irregular-answer", limit=30, window=60, user=user
    ):
        return limited
    try:
        payload = json_body(request)
        question_id = str(payload.get("question_id", ""))
        answer = str(payload.get("answer", "")).strip()
    except (TypeError, ValueError):
        return json_error("Invalid irregular payload.")

    try:
        result = submit_issued_irregular_answer(user, question_id, answer)
    except ValueError as exc:
        return json_error(str(exc), status=400)
    return JsonResponse({"ok": True, **result})
