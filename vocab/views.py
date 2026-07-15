from __future__ import annotations

import logging
import mimetypes
import os
import tempfile
import traceback
from hashlib import sha256

from django.conf import settings
from django.core.signing import BadSignature, SignatureExpired, TimestampSigner
from django.http import FileResponse, HttpRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from core.env import get_telegram_bot_username, get_telegram_token, get_webapp_url
from .models import AddWordDraft, TelegramUser, VocabularyItem
from .alphabets import get_alphabet_letter
from .irregular_verbs import IRREGULAR_VERBS
from .openai_utils import transcribe_speech_file
from .openai_limits import openai_user_scope
from .services import (
    add_pack_words_to_user,
    add_words_from_text,
    apply_user_settings,
    issue_learning_question,
    build_user_progress,
    build_irregular_question,
    build_alphabet_question,
    build_speaking_question,
    consume_web_login_token,
    create_web_login_token,
    create_word_drafts_from_text,
    delete_user_avatar,
    delete_word,
    delete_user_draft,
    finalize_word_draft,
    create_checkout_session,
    get_billing_payload,
    get_draft_image_file,
    get_profile_avatar_file,
    get_active_course_code,
    get_user_draft,
    get_user_settings_payload,
    get_user_word,
    get_word_image_file,
    get_ordered_unlearned_words,
    list_words,
    list_irregular_page,
    list_alphabet_page,
    list_word_packs,
    request_draft_image_generation,
    refresh_draft_language_data,
    request_word_image_generation,
    save_user_avatar,
    serialize_user,
    serialize_draft,
    serialize_word,
    submit_issued_learning_answer,
    get_issued_speaking_question,
    submit_issued_speaking_answer,
    submit_alphabet_answer,
    update_learning_streak,
    update_irregular_progress,
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
from .jobs import enqueue_job
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

logger = logging.getLogger(__name__)
MAX_IMAGE_REGENERATIONS = 3
MAX_ADD_BATCH_WORDS = 10
QUESTION_SIGNER = TimestampSigner(salt="vocab.alphabet-question")


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


def _safe_unlink(path: str) -> None:
    if not path or not os.path.exists(path):
        return
    try:
        os.remove(path)
    except OSError:
        logger.warning("Failed to remove temp file %s", path)


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


@require_POST
def learn_question(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json_error(str(exc))
    raw_exclude = payload.get("exclude_ids", [])
    if not isinstance(raw_exclude, list):
        return _json_error("exclude_ids must be an array.")
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
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    try:
        payload = _json_body(request)
        question_id = str(payload.get("question_id", ""))
        answer = str(payload.get("answer", ""))
    except (TypeError, ValueError):
        return _json_error("Invalid learning answer payload.")

    try:
        result = submit_issued_learning_answer(user, question_id, answer)
    except ValueError as exc:
        return _json_error(str(exc))
    return JsonResponse({"ok": True, **result})


@require_GET
def practice_question(request: HttpRequest) -> JsonResponse:
    return _json_error(
        "This endpoint is retired. Use /api/learn/question instead.", status=410
    )


@require_POST
def practice_answer(request: HttpRequest) -> JsonResponse:
    return _json_error(
        "This endpoint is retired. Submit a server-issued learning question instead.",
        status=410,
    )


@require_GET
def listening_question(request: HttpRequest) -> JsonResponse:
    return _json_error(
        "This endpoint is retired. Use /api/learn/question instead.", status=410
    )


@require_POST
def listening_answer(request: HttpRequest) -> JsonResponse:
    return _json_error(
        "This endpoint is retired. Submit a server-issued learning question instead.",
        status=410,
    )


@require_GET
def speaking_question(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    question = build_speaking_question(user)
    if question is None:
        return JsonResponse({"ok": True, "empty": True})
    return JsonResponse({"ok": True, "question": question})


@require_POST
def speaking_answer(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    if limited := _enforce_request_limit(
        request, scope="speech-transcription", limit=10, window=60, user=user
    ):
        return limited
    question_id = str(request.POST.get("question_id", ""))
    try:
        get_issued_speaking_question(user, question_id)
    except ValueError as exc:
        return _json_error(str(exc), status=400)

    audio_file = request.FILES.get("audio")
    if audio_file is None:
        return _json_error("Audio file is required.")
    if audio_file.size > 10 * 1024 * 1024:
        return _json_error("Audio file is too large.", status=413)
    allowed_audio_content_types = {
        "audio/webm",
        "audio/ogg",
        "audio/mpeg",
        "audio/wav",
        "audio/mp4",
    }
    if audio_file.content_type not in allowed_audio_content_types:
        return _json_error("Unsupported audio format.", status=415)

    suffix = os.path.splitext(audio_file.name or "")[1].lower() or ".webm"
    if suffix not in {".webm", ".ogg", ".mp3", ".wav", ".m4a", ".mp4"}:
        return _json_error("Unsupported audio format.", status=415)
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            for chunk in audio_file.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name

        with openai_user_scope(user.id):
            transcript = transcribe_speech_file(temp_path)
        return JsonResponse(
            {"ok": True, **submit_issued_speaking_answer(user, question_id, transcript)}
        )
    except ValueError as exc:
        return _json_error(str(exc), status=400)
    except Exception:
        logger.exception(
            "Speaking recognition failed for user=%s question_id=%s",
            user.id,
            question_id,
        )
        return _json_error("Speech recognition is temporarily unavailable.", status=503)
    finally:
        _safe_unlink(temp_path)


def _serve_cached_audio(text: str, language_code: str) -> JsonResponse | FileResponse:
    """Serve a cached audio asset; request handlers must never generate it on GET."""
    from .tts import get_audio_path, is_audio_ready

    audio_path = get_audio_path(text, language_code=language_code)
    if not is_audio_ready(text, language_code=language_code):
        return _json_error(
            "Audio is being prepared.", status=404, code="audio_not_ready"
        )
    try:
        return FileResponse(open(audio_path, "rb"), content_type="audio/mpeg")
    except OSError:
        logger.exception("Cached audio file disappeared: %s", audio_path)
        return _json_error("Audio is temporarily unavailable.", status=503)


def _enqueue_audio_generation(text: str, language_code: str) -> tuple[bool, int | None]:
    from .tts import is_audio_ready

    if is_audio_ready(text, language_code=language_code):
        return True, None
    fingerprint = sha256(f"{language_code}:{text}".encode("utf-8")).hexdigest()
    job = enqueue_job(
        kind="tts_audio",
        deduplication_key=f"tts:{fingerprint}",
        payload={"text": text, "language_code": language_code},
        priority=10,
    )
    return False, job.id


@require_GET
def word_audio(request: HttpRequest, word_id: int) -> JsonResponse | FileResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user
    item = get_user_word(user, word_id)
    if item is None:
        return _json_error("Word not found.", status=404)
    return _serve_cached_audio(item.word, item.course_code)


@require_POST
def word_audio_prepare(request: HttpRequest, word_id: int) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := _enforce_request_limit(
        request, scope="tts-prepare", limit=20, window=60, user=user
    ):
        return limited
    item = get_user_word(user, word_id)
    if item is None:
        return _json_error("Word not found.", status=404)
    ready, job_id = _enqueue_audio_generation(item.word, item.course_code)
    return JsonResponse(
        {"ok": True, "ready": ready, "job_id": job_id},
        status=200 if ready else 202,
    )


def _get_alphabet_audio_text(
    user: TelegramUser, symbol: str
) -> tuple[str, str] | JsonResponse:
    if not symbol:
        return _json_error("Alphabet symbol is required.", status=400)
    active_course = get_active_course_code(user)
    letter = get_alphabet_letter(active_course, symbol)
    if letter is None:
        return _json_error("Alphabet letter not found.", status=404)
    return str(letter.get("name") or letter["symbol"]).strip(), active_course


@require_GET
def alphabet_audio(request: HttpRequest) -> JsonResponse | FileResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user
    audio_spec = _get_alphabet_audio_text(user, request.GET.get("symbol", "").strip())
    if isinstance(audio_spec, JsonResponse):
        return audio_spec
    return _serve_cached_audio(*audio_spec)


@require_POST
def alphabet_audio_prepare(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := _enforce_request_limit(
        request, scope="tts-prepare", limit=20, window=60, user=user
    ):
        return limited
    try:
        payload = _json_body(request)
        symbol = str(payload.get("symbol", "")).strip()
    except (TypeError, ValueError):
        return _json_error("Invalid alphabet audio payload.")
    audio_spec = _get_alphabet_audio_text(user, symbol)
    if isinstance(audio_spec, JsonResponse):
        return audio_spec
    ready, job_id = _enqueue_audio_generation(*audio_spec)
    return JsonResponse(
        {"ok": True, "ready": ready, "job_id": job_id},
        status=200 if ready else 202,
    )


@require_GET
def word_image(request: HttpRequest, word_id: int) -> JsonResponse | FileResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    item = get_user_word(user, word_id)
    if item is None:
        return _json_error("Word not found.", status=404)

    image_path = get_word_image_file(item)
    if image_path is None:
        return _json_error("Image not found.", status=404)

    content_type, _ = mimetypes.guess_type(image_path.name)
    try:
        return FileResponse(
            open(image_path, "rb"), content_type=content_type or "image/jpeg"
        )
    except OSError:
        logger.exception(
            "Image file open failed for user=%s word_id=%s path=%s",
            user.id,
            word_id,
            image_path,
        )
        return _json_error("Image is temporarily unavailable.", status=503)


@require_GET
def draft_image(request: HttpRequest, draft_id: int) -> JsonResponse | FileResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    draft = _get_draft_for_user(user, draft_id)
    if isinstance(draft, JsonResponse):
        return draft

    image_path = get_draft_image_file(draft)
    if image_path is None:
        return _json_error("Image not found.", status=404)

    content_type, _ = mimetypes.guess_type(image_path.name)
    try:
        return FileResponse(
            open(image_path, "rb"), content_type=content_type or "image/png"
        )
    except OSError:
        logger.exception(
            "Draft image open failed for user=%s draft_id=%s path=%s",
            user.id,
            draft_id,
            image_path,
        )
        return _json_error("Image is temporarily unavailable.", status=503)


@require_GET
def irregular_list(request: HttpRequest) -> JsonResponse:
    page = max(0, int(request.GET.get("page", 0)))
    return JsonResponse({"ok": True, **list_irregular_page(page)})


@require_GET
def irregular_question(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user
    return JsonResponse({"ok": True, "question": build_irregular_question()})


@require_POST
def irregular_answer(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := _enforce_request_limit(
        request, scope="irregular-answer", limit=30, window=60, user=user
    ):
        return limited

    try:
        payload = _json_body(request)
        base = str(payload.get("base", "")).strip()
        answer = str(payload.get("answer", "")).strip()
    except (TypeError, ValueError):
        return _json_error("Invalid irregular payload.")

    verb = next((item for item in IRREGULAR_VERBS if item["base"] == base), None)
    if verb is None:
        return _json_error("Irregular verb not found.", status=404)
    correct_pair = f"{verb['past']} {verb['participle']}"
    correct = answer == correct_pair
    if correct:
        update_irregular_progress(user, base, True)
        update_learning_streak(user)
    return JsonResponse(
        {
            "ok": True,
            "correct": correct,
            "correct_answer": correct_pair,
            "points_earned": 1 if correct else 0,
            "progress": build_user_progress(user),
        }
    )


@require_GET
def alphabet_list(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    page = max(0, int(request.GET.get("page", 0)))
    return JsonResponse({"ok": True, **list_alphabet_page(user, page)})


@require_GET
def alphabet_question(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
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
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := _enforce_request_limit(
        request, scope="alphabet-answer", limit=30, window=60, user=user
    ):
        return limited

    try:
        payload = _json_body(request)
        question_token = str(payload.get("question_token", ""))
        answer = str(payload.get("answer", ""))
    except (TypeError, ValueError):
        return _json_error("Invalid alphabet payload.")

    try:
        signed_user_id, course_code, symbol = QUESTION_SIGNER.unsign(
            question_token, max_age=15 * 60
        ).split(":", 2)
        if int(signed_user_id) != user.id or course_code != get_active_course_code(
            user
        ):
            return _json_error(
                "Alphabet question does not belong to this user.", status=403
            )
        result = submit_alphabet_answer(user, symbol, answer)
    except (BadSignature, SignatureExpired, ValueError) as exc:
        return _json_error(str(exc))
    return JsonResponse({"ok": True, **result})


@require_GET
def study_cards(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
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


@require_POST
def study_answer(request: HttpRequest) -> JsonResponse:
    return _json_error(
        "This endpoint is retired. Submit a server-issued learning question instead.",
        status=410,
    )
