"""Authenticated first-party analytics ingestion endpoint."""

from __future__ import annotations

from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_POST

from vocab.analytics import CLIENT_EVENT_NAMES, record_product_event
from vocab.api.common import enforce_request_limit, json_body, json_error, require_user


@require_POST
def analytics_event(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := enforce_request_limit(
        request, scope="analytics-event", limit=30, window=60, user=user
    ):
        return limited
    try:
        payload = json_body(request)
    except ValueError as exc:
        return json_error(str(exc))

    name = payload.get("name")
    if not isinstance(name, str) or name not in CLIENT_EVENT_NAMES:
        return json_error("Unsupported analytics event.", status=400)
    properties = payload.get("properties")
    if properties is not None and not isinstance(properties, dict):
        return json_error("Analytics properties must be an object.", status=400)
    record_product_event(user, name, properties=properties)
    return JsonResponse({"ok": True}, status=202)
