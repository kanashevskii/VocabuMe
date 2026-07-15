"""Alphabet practice API with signed, course-scoped question tokens."""

from __future__ import annotations

from django.core.signing import BadSignature, SignatureExpired, TimestampSigner
from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_GET, require_POST

from vocab.api.common import enforce_request_limit, json_body, json_error, require_user
from vocab.services import (
    build_alphabet_question,
    get_active_course_code,
    list_alphabet_page,
    submit_alphabet_answer,
)

QUESTION_SIGNER = TimestampSigner(salt="vocab.alphabet-question")


@require_GET
def alphabet_list(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user

    page = max(0, int(request.GET.get("page", 0)))
    return JsonResponse({"ok": True, **list_alphabet_page(user, page)})


@require_GET
def alphabet_question(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user

    question = build_alphabet_question(user)
    question["letter"] = dict(question["letter"])
    symbol = question.pop("correct_symbol")
    question["letter"].pop("symbol", None)
    question["question_token"] = QUESTION_SIGNER.sign(
        f"{user.id}:{question['course_code']}:{symbol}"
    )
    return JsonResponse({"ok": True, "question": question})


@require_POST
def alphabet_answer(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := enforce_request_limit(
        request, scope="alphabet-answer", limit=30, window=60, user=user
    ):
        return limited

    try:
        payload = json_body(request)
        question_token = str(payload.get("question_token", ""))
        answer = str(payload.get("answer", ""))
    except (TypeError, ValueError):
        return json_error("Invalid alphabet payload.")

    try:
        signed_user_id, course_code, symbol = QUESTION_SIGNER.unsign(
            question_token, max_age=15 * 60
        ).split(":", 2)
        if int(signed_user_id) != user.id or course_code != get_active_course_code(
            user
        ):
            return json_error(
                "Alphabet question does not belong to this user.", status=403
            )
        result = submit_alphabet_answer(user, symbol, answer)
    except (BadSignature, SignatureExpired, ValueError) as exc:
        return json_error(str(exc))
    return JsonResponse({"ok": True, **result})
