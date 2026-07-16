from __future__ import annotations

import logging

from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET

from core.env import get_telegram_bot_username, get_webapp_url
from .models import TelegramUser
from .services import (
    get_billing_payload,
    EntitlementError,
)
from .telegram_auth import verify_webapp_init_data
from .api.common import (
    current_user as _resolve_current_user,
    json_error as _json_error,
)
from .api.docs import api_docs, openapi_schema  # noqa: F401 - URL compatibility exports
from .api.errors import client_error_log  # noqa: F401 - URL compatibility export
from .api.analytics import analytics_event  # noqa: F401 - URL compatibility export
from .api.media import (  # noqa: F401 - URL compatibility exports
    alphabet_audio,
    alphabet_audio_prepare,
    word_audio,
    word_audio_prepare,
)
from .api.images import (  # noqa: F401 - URL compatibility exports
    draft_image,
    word_image,
)
from .api.irregular import (  # noqa: F401 - URL compatibility exports
    irregular_answer,
    irregular_list,
    irregular_question,
)
from .api.learning import (  # noqa: F401 - URL compatibility exports
    learn_answer,
    learn_question,
    listening_answer,
    listening_question,
    practice_answer,
    practice_question,
    study_answer,
    study_cards,
)
from .api.speaking import (  # noqa: F401 - URL compatibility exports
    speaking_answer,
    speaking_question,
)
from .api.packs import (  # noqa: F401 - URL compatibility exports
    packs_add,
    packs_prepare,
    packs_view,
)
from .api.words import (  # noqa: F401 - URL compatibility exports
    word_detail,
    word_draft_confirm_translation,
    word_draft_create,
    word_draft_delete,
    word_draft_regenerate_image,
    word_draft_save,
    word_image_regenerate,
    words,
)
from .api.billing import (  # noqa: F401 - URL compatibility exports
    billing_checkout,
    billing_status,
)
from .api.profile import (  # noqa: F401 - URL compatibility exports
    profile_avatar,
    settings_view,
)
from .api.alphabet import (  # noqa: F401 - URL compatibility exports
    alphabet_answer,
    alphabet_list,
    alphabet_question,
)
from .api.dashboard import dashboard  # noqa: F401 - URL compatibility export

logger = logging.getLogger(__name__)
MAX_IMAGE_REGENERATIONS = 3
MAX_ADD_BATCH_WORDS = 10


def _current_user(request: HttpRequest) -> TelegramUser | None:
    """Compatibility facade for legacy callers that patch the auth verifier."""
    return _resolve_current_user(
        request,
        verify_init_data=verify_webapp_init_data,
    )


def _require_user(request: HttpRequest) -> TelegramUser | JsonResponse:
    user = _current_user(request)
    return (
        user
        if user is not None
        else _json_error("Authentication required.", status=401)
    )


def _json_entitlement_error(
    user: TelegramUser, exc: EntitlementError, status: int = 402
) -> JsonResponse:
    return _json_error(
        exc.message,
        status=status,
        code=exc.code,
        paywall_trigger=exc.paywall_trigger,
        billing=get_billing_payload(user),
    )


@ensure_csrf_cookie
@require_GET
def spa_index(request: HttpRequest):
    return render(request, "index.html")


@ensure_csrf_cookie
@require_GET
def app_config(request: HttpRequest) -> JsonResponse:
    return JsonResponse(
        {
            "ok": True,
            "bot_username": get_telegram_bot_username(),
            "webapp_url": get_webapp_url(),
        }
    )


# Auth endpoints are implemented in ``vocab.api.auth`` while integrations still
# import ``vocab.views``.
from .api.auth import (  # noqa: E402,F401
    auth_logout,
    auth_me,
    auth_poll_link,
    auth_request_link,
    auth_telegram_webapp,
    auth_telegram_widget,
    auth_web_login,
    auth_web_register,
)
