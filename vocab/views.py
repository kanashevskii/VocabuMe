from __future__ import annotations

import logging
import mimetypes
import traceback

from django.conf import settings
from django.http import FileResponse, HttpRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from core.env import get_telegram_bot_username, get_telegram_token, get_webapp_url
from .models import AddWordDraft, TelegramUser, VocabularyItem
from .services import (
    add_pack_words_to_user,
    add_words_from_text,
    apply_user_settings,
    build_user_progress,
    consume_web_login_token,
    create_web_login_token,
    create_word_drafts_from_text,
    delete_user_avatar,
    delete_word,
    delete_user_draft,
    finalize_word_draft,
    create_checkout_session,
    get_billing_payload,
    get_active_course_code,
    get_profile_avatar_file,
    get_user_draft,
    get_user_settings_payload,
    get_user_word,
    get_ordered_unlearned_words,
    list_words,
    list_word_packs,
    request_draft_image_generation,
    refresh_draft_language_data,
    request_word_image_generation,
    save_user_avatar,
    serialize_user,
    serialize_draft,
    serialize_word,
    update_word_translation,
    upsert_telegram_user,
    word_already_exists,
    ensure_pack_preparation,
    EntitlementError,
)
from .telegram_auth import (
    TelegramAuthError,
    verify_login_widget,
    verify_webapp_init_data,
)
from .api.common import (
    SESSION_USER_KEY,
    current_user as _resolve_current_user,
    enforce_request_limit as _enforce_request_limit,
    json_body as _json_body,
    json_error as _json_error,
    login as _login,
)
from .api.docs import api_docs, openapi_schema  # noqa: F401 - URL compatibility exports
from .api.errors import (
    client_error_log,  # noqa: F401 - URL compatibility export
    log_app_error as _log_app_error,
)
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
from .api.alphabet import (  # noqa: F401 - URL compatibility exports
    alphabet_answer,
    alphabet_list,
    alphabet_question,
)

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


@csrf_exempt
@require_POST
def auth_telegram_widget(request: HttpRequest) -> JsonResponse:
    if limited := _enforce_request_limit(
        request, scope="auth-widget", limit=20, window=60
    ):
        return limited
    try:
        payload = _json_body(request)
        verified = verify_login_widget(
            payload, get_telegram_token(), settings.TELEGRAM_AUTH_MAX_AGE_SECONDS
        )
    except (ValueError, TelegramAuthError) as exc:
        return _json_error(str(exc), status=400)

    telegram_id = int(verified["id"])
    username = verified.get("username")
    user = upsert_telegram_user(chat_id=telegram_id, username=username)
    return _login(request, user)


@csrf_exempt
@require_POST
def auth_telegram_webapp(request: HttpRequest) -> JsonResponse:
    if limited := _enforce_request_limit(
        request, scope="auth-webapp", limit=20, window=60
    ):
        return limited
    try:
        payload = _json_body(request)
        init_data = payload.get("init_data", "")
        verified = verify_webapp_init_data(
            init_data, get_telegram_token(), settings.TELEGRAM_AUTH_MAX_AGE_SECONDS
        )
    except (ValueError, TelegramAuthError) as exc:
        return _json_error(str(exc), status=400)

    telegram_user = verified.get("user")
    if not isinstance(telegram_user, dict):
        return _json_error("Telegram user was not provided.", status=400)
    telegram_id = telegram_user.get("id")
    if isinstance(telegram_id, bool) or not isinstance(telegram_id, int):
        return _json_error("Telegram user was not provided.", status=400)

    username = telegram_user.get("username")
    user = upsert_telegram_user(
        chat_id=telegram_id,
        username=username if isinstance(username, str) else None,
    )
    return _login(request, user)


@require_POST
def auth_logout(request: HttpRequest) -> JsonResponse:
    request.session.flush()
    return JsonResponse({"ok": True})


@require_GET
def auth_me(request: HttpRequest) -> JsonResponse:
    user = _current_user(request)
    return JsonResponse(
        {
            "ok": True,
            "authenticated": user is not None,
            "user": serialize_user(user) if user else None,
            "progress": build_user_progress(user) if user else None,
        }
    )


@require_POST
def auth_request_link(request: HttpRequest) -> JsonResponse:
    if limited := _enforce_request_limit(
        request, scope="auth-link", limit=5, window=60
    ):
        return limited
    login_token = create_web_login_token()
    deep_link = (
        f"https://t.me/{get_telegram_bot_username()}"
        f"?start=login_{login_token.token}"
    )
    return JsonResponse(
        {
            "ok": True,
            "token": login_token.token,
            "deep_link": deep_link,
            "expires_at": login_token.expires_at.isoformat(),
        }
    )


@require_POST
def auth_web_register(request: HttpRequest) -> JsonResponse:
    return _json_error(
        "Email/password registration is no longer supported. Use Telegram login.",
        status=410,
    )


@require_POST
def auth_web_login(request: HttpRequest) -> JsonResponse:
    return _json_error(
        "Email/password login is no longer supported. Use Telegram login.", status=410
    )


@require_POST
def auth_poll_link(request: HttpRequest, token: str) -> JsonResponse:
    user = consume_web_login_token(token)
    if user is None:
        return JsonResponse({"ok": True, "authenticated": False})
    request.session.cycle_key()
    request.session[SESSION_USER_KEY] = user.id
    return JsonResponse(
        {
            "ok": True,
            "authenticated": True,
            "user": serialize_user(user),
            "progress": build_user_progress(user),
        }
    )


def _get_draft_for_user(
    user: TelegramUser, draft_id: int
) -> AddWordDraft | JsonResponse:
    draft = get_user_draft(user, draft_id)
    if draft is None:
        return _json_error("Draft not found.", status=404)
    return draft


@require_GET
def dashboard(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
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


@require_GET
def billing_status(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user
    return JsonResponse({"ok": True, "billing": get_billing_payload(user)})


@require_POST
def billing_checkout(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := _enforce_request_limit(
        request, scope="billing-checkout", limit=5, window=60, user=user
    ):
        return limited

    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json_error(str(exc))

    try:
        checkout = create_checkout_session(
            user,
            plan_code=(payload.get("plan_code") or "premium").strip().lower(),
            billing_period=(payload.get("billing_period") or "monthly").strip().lower(),
            return_source=(payload.get("source") or "miniapp").strip().lower(),
        )
    except ValueError as exc:
        return _json_error(str(exc), status=400)
    except Exception as exc:
        logger.exception("billing checkout failed")
        _log_app_error(
            request,
            user=user,
            category="billing",
            status_code=500,
            message=f"billing checkout failed: {exc}",
            context={"traceback": traceback.format_exc()[-4000:]},
        )
        return _json_error("Не удалось начать оплату. Попробуй ещё раз.", status=500)

    return JsonResponse({"ok": True, **checkout, "billing": get_billing_payload(user)})


@require_http_methods(["GET", "POST"])
def words(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
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

    if limited := _enforce_request_limit(
        request, scope="word-generation", limit=10, window=60, user=user
    ):
        return limited

    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json_error(str(exc))

    try:
        result = add_words_from_text(
            user, payload.get("text", ""), max_batch_words=MAX_ADD_BATCH_WORDS
        )
    except EntitlementError as exc:
        return _json_entitlement_error(user, exc)
    except ValueError as exc:
        return _json_error(str(exc), status=400)

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
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := _enforce_request_limit(
        request, scope="word-draft-generation", limit=10, window=60, user=user
    ):
        return limited

    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json_error(str(exc))

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
        return _json_error(str(exc), status=400)
    except Exception as exc:
        logger.exception("word_draft_create failed")
        _log_app_error(
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
        return _json_error(
            "Не удалось подготовить слово. Попробуй ещё раз.", status=500
        )


@require_POST
def word_draft_confirm_translation(request: HttpRequest, draft_id: int) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    draft = _get_draft_for_user(user, draft_id)
    if isinstance(draft, JsonResponse):
        return draft

    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json_error(str(exc))

    translation = (payload.get("translation") or "").strip()
    if not translation:
        return _json_error("Translation is required.", status=400)

    draft = refresh_draft_language_data(draft, translation)
    draft = request_draft_image_generation(draft)
    return JsonResponse(
        {"ok": True, "draft": serialize_draft(draft), "step": "confirm_image"}
    )


@require_POST
def word_draft_regenerate_image(request: HttpRequest, draft_id: int) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := _enforce_request_limit(
        request, scope="image-regeneration", limit=5, window=60, user=user
    ):
        return limited

    draft = _get_draft_for_user(user, draft_id)
    if isinstance(draft, JsonResponse):
        return draft
    if not draft.translation_confirmed:
        return _json_error("Confirm translation first.", status=400)
    try:
        draft = request_draft_image_generation(draft, force_regenerate=True)
    except EntitlementError as exc:
        return _json_entitlement_error(user, exc)
    return JsonResponse(
        {"ok": True, "draft": serialize_draft(draft), "step": "confirm_image"}
    )


@require_POST
def word_draft_save(request: HttpRequest, draft_id: int) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    draft = _get_draft_for_user(user, draft_id)
    if isinstance(draft, JsonResponse):
        return draft
    if not draft.translation_confirmed:
        return _json_error("Confirm translation first.", status=400)
    if word_already_exists(user, draft.word):
        delete_user_draft(user, draft.id)
        return _json_error("This word already exists.", status=400)

    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json_error(str(exc))

    use_image = bool(payload.get("use_image", True))
    try:
        item = finalize_word_draft(draft, use_image=use_image)
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
    user = _require_user(request)
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
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    try:
        item = get_user_word(user, word_id)
        if item is None:
            raise VocabularyItem.DoesNotExist
    except VocabularyItem.DoesNotExist:
        return _json_error("Word not found.", status=404)

    if request.method == "DELETE":
        delete_word(user, word_id)
        return JsonResponse({"ok": True})

    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json_error(str(exc))

    translation = (payload.get("translation") or "").strip()
    if not translation:
        return _json_error("Translation is required.")

    item = update_word_translation(user, word_id, translation)
    return JsonResponse({"ok": True, "item": serialize_word(item)})


@require_POST
def word_image_regenerate(request: HttpRequest, word_id: int) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := _enforce_request_limit(
        request, scope="image-regeneration", limit=5, window=60, user=user
    ):
        return limited

    item = get_user_word(user, word_id)
    if item is None:
        return _json_error("Word not found.", status=404)
    try:
        item = request_word_image_generation(item, force_regenerate=True)
    except EntitlementError as exc:
        return _json_entitlement_error(user, exc)
    return JsonResponse({"ok": True, "item": serialize_word(item)})


@require_http_methods(["GET", "POST"])
def settings_view(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    if request.method == "GET":
        return JsonResponse({"ok": True, "settings": get_user_settings_payload(user)})

    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json_error(str(exc))

    try:
        apply_user_settings(user, payload)
    except (TypeError, ValueError) as exc:
        return _json_error(f"Invalid settings payload: {exc}")
    return JsonResponse({"ok": True})


@require_http_methods(["GET", "POST", "DELETE"])
def profile_avatar(request: HttpRequest) -> JsonResponse | FileResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    if request.method == "GET":
        avatar_path = get_profile_avatar_file(user)
        if avatar_path is None:
            return _json_error("Avatar not found.", status=404)
        content_type, _ = mimetypes.guess_type(avatar_path.name)
        try:
            return FileResponse(
                open(avatar_path, "rb"), content_type=content_type or "image/webp"
            )
        except OSError:
            logger.exception(
                "Avatar file open failed for user=%s path=%s", user.id, avatar_path
            )
            return _json_error("Avatar is temporarily unavailable.", status=503)

    if request.method == "DELETE":
        delete_user_avatar(user)
        return JsonResponse(
            {
                "ok": True,
                "user": serialize_user(user),
                "settings": get_user_settings_payload(user),
            }
        )

    uploaded_file = request.FILES.get("avatar")
    if uploaded_file is None:
        return _json_error("Avatar file is required.")

    try:
        save_user_avatar(user, uploaded_file)
    except ValueError as exc:
        return _json_error(str(exc))

    return JsonResponse(
        {
            "ok": True,
            "user": serialize_user(user),
            "settings": get_user_settings_payload(user),
        }
    )


@require_GET
def packs_view(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user
    return JsonResponse({"ok": True, "packs": list_word_packs(user)})


@require_POST
def packs_prepare(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := _enforce_request_limit(
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
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := _enforce_request_limit(
        request, scope="pack-add", limit=5, window=60, user=user
    ):
        return limited
    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json_error(str(exc))

    pack_id = str(payload.get("pack_id", "")).strip()
    level_id = str(payload.get("level_id", "")).strip()
    selected_words = payload.get("selected_words") or []
    if not isinstance(selected_words, list):
        return _json_error("selected_words must be a list.")

    try:
        result = add_pack_words_to_user(
            user, pack_id, level_id, [str(word) for word in selected_words]
        )
    except EntitlementError as exc:
        return _json_entitlement_error(user, exc)
    except ValueError as exc:
        return _json_error(str(exc))

    return JsonResponse(
        {
            "ok": True,
            **result,
            "progress": build_user_progress(user),
            "packs": list_word_packs(user),
        }
    )
