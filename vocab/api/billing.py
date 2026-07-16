"""Billing status and checkout transport layer."""

from __future__ import annotations

import logging
import traceback

from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_GET, require_POST

from vocab.analytics import record_product_event
from vocab.api.common import enforce_request_limit, json_body, json_error, require_user
from vocab.api.errors import log_app_error
from vocab.services import create_checkout_session, get_billing_payload

logger = logging.getLogger(__name__)


@require_GET
def billing_status(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    return JsonResponse({"ok": True, "billing": get_billing_payload(user)})


@require_POST
def billing_checkout(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := enforce_request_limit(
        request, scope="billing-checkout", limit=5, window=60, user=user
    ):
        return limited
    try:
        payload = json_body(request)
    except ValueError as exc:
        return json_error(str(exc))
    try:
        checkout = create_checkout_session(
            user,
            plan_code=(payload.get("plan_code") or "premium").strip().lower(),
            billing_period=(payload.get("billing_period") or "monthly").strip().lower(),
            return_source=(payload.get("source") or "miniapp").strip().lower(),
        )
    except ValueError as exc:
        return json_error(str(exc), status=400)
    except Exception as exc:
        logger.exception("billing checkout failed")
        log_app_error(
            request,
            user=user,
            category="billing",
            status_code=500,
            message=f"billing checkout failed: {exc}",
            context={"traceback": traceback.format_exc()[-4000:]},
        )
        return json_error("Не удалось начать оплату. Попробуй ещё раз.", status=500)
    record_product_event(
        user,
        "checkout_started",
        properties={
            "billing_period": checkout["plan"]["billing_period"],
            "source": (payload.get("source") or "miniapp").strip().lower(),
        },
    )
    return JsonResponse({"ok": True, **checkout, "billing": get_billing_payload(user)})
