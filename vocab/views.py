from __future__ import annotations

import json
import logging
import mimetypes
import os
import tempfile
import traceback
from datetime import time as dt_time

from decouple import config
from django.http import FileResponse, HttpRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from .models import AddWordDraft, AppErrorLog, TelegramUser, VocabularyItem
from .openai_utils import generate_word_data, generate_word_data_batch, transcribe_speech_file
from .services import (
    add_pack_words_to_user,
    build_learning_question,
    build_user_progress,
    build_choice_question,
    build_irregular_question,
    build_listening_question,
    build_speaking_question,
    consume_web_login_token,
    create_word_draft,
    create_web_login_token,
    create_word,
    ensure_draft_image,
    finalize_word_draft,
    get_draft_image_file,
    get_word_image_file,
    get_ordered_unlearned_words,
    list_words,
    list_irregular_page,
    list_word_packs,
    parse_word_batch,
    request_draft_image_generation,
    request_word_image_generation,
    refresh_draft_language_data,
    recalculate_user_word_progress,
    resolve_shared_image_path,
    request_word_image_generation,
    serialize_user,
    serialize_draft,
    serialize_word,
    submit_learning_text_answer,
    submit_choice_answer,
    submit_listening_answer,
    evaluate_speaking_answer,
    update_learning_streak,
    update_irregular_progress,
    update_word_progress,
    upsert_telegram_user,
    word_already_exists,
    ensure_pack_preparation,
)
from .telegram_auth import TelegramAuthError, verify_login_widget, verify_webapp_init_data
from .utils import normalize_timezone_value

SESSION_USER_KEY = "telegram_user_id"
logger = logging.getLogger(__name__)
MAX_IMAGE_REGENERATIONS = 3
MAX_ADD_BATCH_WORDS = 10


def _json_body(request: HttpRequest) -> dict:
    if not request.body:
        return {}
    try:
        return json.loads(request.body)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON body.") from exc


def _json_error(message: str, status: int = 400) -> JsonResponse:
    return JsonResponse({"ok": False, "error": message}, status=status)


def _safe_context(payload: dict | None) -> dict:
    if not payload:
        return {}
    safe = {}
    for key, value in payload.items():
        if key.lower() in {"password", "token", "init_data", "audio"}:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def _log_app_error(
    request: HttpRequest,
    *,
    message: str,
    category: str = "server",
    level: str = "error",
    status_code: int | None = None,
    user: TelegramUser | None = None,
    context: dict | None = None,
) -> None:
    try:
        AppErrorLog.objects.create(
            user=user or _current_user(request),
            category=category,
            level=level,
            message=message[:4000],
            path=request.path[:255],
            method=request.method[:10],
            status_code=status_code,
            context=_safe_context(context),
        )
    except Exception:
        logger.exception("Failed to persist AppErrorLog for %s", request.path)


def _current_user(request: HttpRequest) -> TelegramUser | None:
    user_id = request.session.get(SESSION_USER_KEY)
    if user_id:
        try:
            return TelegramUser.objects.get(id=user_id)
        except TelegramUser.DoesNotExist:
            request.session.pop(SESSION_USER_KEY, None)

    init_data = request.headers.get("X-Telegram-Init-Data", "").strip()
    if not init_data:
        return None

    try:
        verified = verify_webapp_init_data(init_data, config("TELEGRAM_TOKEN"))
    except TelegramAuthError:
        return None

    telegram_user = verified.get("user") or {}
    telegram_id = telegram_user.get("id")
    if not telegram_id:
        return None

    user = upsert_telegram_user(chat_id=int(telegram_id), username=telegram_user.get("username"))
    request.session[SESSION_USER_KEY] = user.id
    return user


def _require_user(request: HttpRequest) -> TelegramUser | JsonResponse:
    user = _current_user(request)
    if user is None:
        return _json_error("Authentication required.", status=401)
    return user


def _login(request: HttpRequest, user: TelegramUser) -> JsonResponse:
    request.session.cycle_key()
    request.session[SESSION_USER_KEY] = user.id
    return JsonResponse({"ok": True, "user": serialize_user(user), "progress": build_user_progress(user)})


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
            "bot_username": config("TELEGRAM_BOT_USERNAME", default=""),
            "webapp_url": config("WEBAPP_URL", default=""),
        }
    )


@csrf_exempt
@require_POST
def auth_telegram_widget(request: HttpRequest) -> JsonResponse:
    try:
        payload = _json_body(request)
        verified = verify_login_widget(payload, config("TELEGRAM_TOKEN"))
    except (ValueError, TelegramAuthError) as exc:
        return _json_error(str(exc), status=400)

    telegram_id = int(verified["id"])
    username = verified.get("username")
    user = upsert_telegram_user(chat_id=telegram_id, username=username)
    return _login(request, user)


@csrf_exempt
@require_POST
def auth_telegram_webapp(request: HttpRequest) -> JsonResponse:
    try:
        payload = _json_body(request)
        init_data = payload.get("init_data", "")
        verified = verify_webapp_init_data(init_data, config("TELEGRAM_TOKEN"))
    except (ValueError, TelegramAuthError) as exc:
        return _json_error(str(exc), status=400)

    telegram_user = verified.get("user") or {}
    telegram_id = telegram_user.get("id")
    if not telegram_id:
        return _json_error("Telegram user was not provided.", status=400)

    username = telegram_user.get("username")
    user = upsert_telegram_user(chat_id=int(telegram_id), username=username)
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
    login_token = create_web_login_token()
    deep_link = f"https://t.me/{config('TELEGRAM_BOT_USERNAME', default='')}" f"?start=login_{login_token.token}"
    return JsonResponse(
        {
            "ok": True,
            "token": login_token.token,
            "deep_link": deep_link,
            "expires_at": login_token.expires_at.isoformat(),
        }
    )


@require_GET
def auth_poll_link(request: HttpRequest, token: str) -> JsonResponse:
    user = consume_web_login_token(token)
    if user is None:
        return JsonResponse({"ok": True, "authenticated": False})
    request.session.cycle_key()
    request.session[SESSION_USER_KEY] = user.id
    return JsonResponse({"ok": True, "authenticated": True, "user": serialize_user(user), "progress": build_user_progress(user)})


@csrf_exempt
@require_POST
def client_error_log(request: HttpRequest) -> JsonResponse:
    user = _current_user(request)
    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json_error(str(exc))

    _log_app_error(
        request,
        user=user,
        category=(payload.get("category") or "client")[:50],
        level=(payload.get("level") or "error")[:20],
        status_code=payload.get("status_code"),
        message=(payload.get("message") or "Client-side error")[:4000],
        context={
            "url": payload.get("url", ""),
            "detail": payload.get("detail", ""),
            "meta": payload.get("meta", {}),
        },
    )
    return JsonResponse({"ok": True})


def _get_draft_for_user(user: TelegramUser, draft_id: int) -> AddWordDraft | JsonResponse:
    try:
        return AddWordDraft.objects.get(id=draft_id, user=user)
    except AddWordDraft.DoesNotExist:
        return _json_error("Draft not found.", status=404)


@require_GET
def dashboard(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    stats = build_user_progress(user)
    recent_words = [serialize_word(item) for item in list_words(user, limit=6)]
    next_cards = [serialize_word(item) for item in get_ordered_unlearned_words(user, count=4)]
    return JsonResponse(
        {
            "ok": True,
            "user": serialize_user(user),
            "progress": stats,
            "recent_words": recent_words,
            "next_cards": next_cards,
        }
    )


@require_http_methods(["GET", "POST"])
def words(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    if request.method == "GET":
        search = request.GET.get("search", "").strip()
        status = request.GET.get("status", "all")
        items = [serialize_word(item) for item in list_words(user, search=search, status=status, limit=150)]
        return JsonResponse({"ok": True, "items": items})

    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json_error(str(exc))

    entries = parse_word_batch(payload.get("text", ""))
    if not entries:
        return _json_error("Add at least one word.", status=400)
    if len(entries) > MAX_ADD_BATCH_WORDS:
        return _json_error(f"You can add at most {MAX_ADD_BATCH_WORDS} words at once.", status=400)

    created_items = []
    skipped = []
    failed = []

    for entry in entries:
        if word_already_exists(user, entry.word):
            skipped.append({"word": entry.word, "reason": "duplicate"})
            continue

        word_data = generate_word_data(entry.word, translation_hint=entry.translation_hint)
        if not word_data:
            failed.append({"word": entry.word, "reason": "generation_failed"})
            continue

        try:
            item = create_word(user, word_data)
            created_items.append(serialize_word(item))
        except Exception:
            failed.append({"word": entry.word, "reason": "save_failed"})

    return JsonResponse(
        {
            "ok": True,
            "created": created_items,
            "skipped": skipped,
            "failed": failed,
        }
    )


@require_POST
def word_draft_create(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json_error(str(exc))

    try:
        entries = parse_word_batch(payload.get("text", ""))
        if not entries:
            return _json_error("Add one word or phrase.", status=400)
        if len(entries) > MAX_ADD_BATCH_WORDS:
            return _json_error(f"За один раз можно добавить максимум {MAX_ADD_BATCH_WORDS} слов или фраз.", status=400)
        if len(entries) > 1:
            missing_translation = [entry.word for entry in entries if not entry.translation_hint]
            if missing_translation:
                return _json_error(
                    "For multiple lines, provide a translation on every line. Add these one by one: "
                    + ", ".join(missing_translation[:5]),
                    status=400,
                )

            drafts = []
            skipped = []
            failed = []

            batch_generated = generate_word_data_batch(
                [{"word": entry.word, "translation_hint": entry.translation_hint} for entry in entries]
            )

            for entry, generated in zip(entries, batch_generated, strict=False):
                if word_already_exists(user, entry.word):
                    skipped.append({"word": entry.word, "reason": "duplicate"})
                    continue

                if not generated:
                    failed.append({"word": entry.word, "reason": "generation_failed"})
                    continue

                try:
                    draft = create_word_draft(user, entry.word, generated, translation_hint=entry.translation_hint)
                    shared_image_path = resolve_shared_image_path(draft.word, draft.translation, "")
                    if shared_image_path:
                        draft.image_path = shared_image_path
                        draft.save(update_fields=["image_path", "updated_at"])
                    else:
                        draft = request_draft_image_generation(draft)
                    drafts.append(serialize_draft(draft))
                except Exception:
                    logger.exception("Batch draft save failed for %s", entry.word)
                    failed.append({"word": entry.word, "reason": "save_failed"})

            return JsonResponse(
                {
                    "ok": True,
                    "batch_review": True,
                    "drafts": drafts,
                    "skipped": skipped,
                    "failed": failed,
                    "progress": build_user_progress(user),
                }
            )

        entry = entries[0]
        if word_already_exists(user, entry.word):
            return _json_error("This word already exists.", status=400)

        generated = generate_word_data(entry.word, translation_hint=entry.translation_hint)
        if not generated:
            return _json_error("Could not prepare the word data.", status=400)

        shared_image_path = resolve_shared_image_path(generated["word"], generated.get("translation", ""), "")
        if generated.get("translation") and shared_image_path:
            item = create_word(user, {**generated, "image_path": shared_image_path})
            return JsonResponse(
                {
                    "ok": True,
                    "auto_saved": True,
                    "item": serialize_word(item),
                    "progress": build_user_progress(user),
                }
            )

        draft = create_word_draft(user, entry.word, generated, translation_hint=entry.translation_hint)
        step = "confirm_translation"
        if draft.translation_confirmed:
            draft = request_draft_image_generation(draft)
            step = "confirm_image"
        return JsonResponse({"ok": True, "draft": serialize_draft(draft), "step": step})
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
        return _json_error("Не удалось подготовить слово. Попробуй ещё раз.", status=500)


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
    return JsonResponse({"ok": True, "draft": serialize_draft(draft), "step": "confirm_image"})


@require_POST
def word_draft_regenerate_image(request: HttpRequest, draft_id: int) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    draft = _get_draft_for_user(user, draft_id)
    if isinstance(draft, JsonResponse):
        return draft
    if not draft.translation_confirmed:
        return _json_error("Confirm translation first.", status=400)
    if draft.image_regeneration_count >= MAX_IMAGE_REGENERATIONS:
        return _json_error("Лимит перегенерации фото исчерпан.", status=400)

    draft = request_draft_image_generation(draft, force_regenerate=True)
    return JsonResponse({"ok": True, "draft": serialize_draft(draft), "step": "confirm_image"})


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
        draft.delete()
        return _json_error("This word already exists.", status=400)

    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json_error(str(exc))

    use_image = bool(payload.get("use_image", True))
    item = finalize_word_draft(draft, use_image=use_image)
    return JsonResponse({"ok": True, "item": serialize_word(item), "progress": build_user_progress(user)})


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
    draft.delete()
    return JsonResponse({"ok": True})


@require_http_methods(["PATCH", "DELETE"])
def word_detail(request: HttpRequest, word_id: int) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    try:
        item = VocabularyItem.objects.get(id=word_id, user=user)
    except VocabularyItem.DoesNotExist:
        return _json_error("Word not found.", status=404)

    if request.method == "DELETE":
        item.delete()
        return JsonResponse({"ok": True})

    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json_error(str(exc))

    translation = (payload.get("translation") or "").strip()
    if not translation:
        return _json_error("Translation is required.")

    item.translation = translation
    item.save(update_fields=["translation", "updated_at"])
    return JsonResponse({"ok": True, "item": serialize_word(item)})


@require_POST
def word_image_regenerate(request: HttpRequest, word_id: int) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    try:
        item = VocabularyItem.objects.get(id=word_id, user=user)
    except VocabularyItem.DoesNotExist:
        return _json_error("Word not found.", status=404)
    if item.image_regeneration_count >= MAX_IMAGE_REGENERATIONS:
        return _json_error("Лимит перегенерации фото исчерпан.", status=400)

    item = request_word_image_generation(item, force_regenerate=True)
    return JsonResponse({"ok": True, "item": serialize_word(item)})


@require_http_methods(["GET", "POST"])
def settings_view(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    if request.method == "GET":
        return JsonResponse(
            {
                "ok": True,
                "settings": {
                    "exercise_goal": max(2, min(user.repeat_threshold, 5)),
                    "session_question_limit": max(1, min(user.session_question_limit, 50)),
                    "enable_review_old_words": user.enable_review_old_words,
                    "days_before_review": user.days_before_review,
                    "reminder_enabled": user.reminder_enabled,
                    "reminder_time": user.reminder_time.strftime("%H:%M"),
                    "reminder_interval_days": user.reminder_interval_days,
                    "reminder_timezone": user.reminder_timezone,
                },
            }
        )

    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json_error(str(exc))

    try:
        previous_goal = user.repeat_threshold
        exercise_goal = payload.get("exercise_goal", payload.get("repeat_threshold", user.repeat_threshold))
        user.repeat_threshold = max(2, min(int(exercise_goal), 5))
        user.session_question_limit = max(1, min(int(payload.get("session_question_limit", user.session_question_limit)), 50))
        user.enable_review_old_words = bool(payload.get("enable_review_old_words", user.enable_review_old_words))
        user.days_before_review = max(1, min(int(payload.get("days_before_review", user.days_before_review)), 365))
        user.reminder_enabled = bool(payload.get("reminder_enabled", user.reminder_enabled))
        user.reminder_interval_days = max(1, min(int(payload.get("reminder_interval_days", user.reminder_interval_days)), 30))
        reminder_time = payload.get("reminder_time", user.reminder_time.strftime("%H:%M"))
        user.reminder_time = dt_time.fromisoformat(reminder_time)
        user.reminder_timezone = normalize_timezone_value(payload.get("reminder_timezone", user.reminder_timezone))
    except (TypeError, ValueError) as exc:
        return _json_error(f"Invalid settings payload: {exc}")

    user.save()
    if user.repeat_threshold != previous_goal:
        recalculate_user_word_progress(user)
    return JsonResponse({"ok": True})


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
    for pack in list_word_packs():
        for level in pack["levels"]:
            ensure_pack_preparation(pack["id"], level["id"])
    return JsonResponse({"ok": True})


@require_POST
def packs_add(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user
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
        result = add_pack_words_to_user(user, pack_id, level_id, [str(word) for word in selected_words])
    except ValueError as exc:
        return _json_error(str(exc))

    return JsonResponse({"ok": True, **result, "progress": build_user_progress(user), "packs": list_word_packs(user)})


@require_GET
def learn_question(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    raw_exclude = [chunk.strip() for chunk in request.GET.get("exclude_ids", "").split(",") if chunk.strip()]
    exclude_ids: list[int] = []
    for chunk in raw_exclude:
        try:
            exclude_ids.append(int(chunk))
        except ValueError:
            continue

    question = build_learning_question(user, exclude_ids=exclude_ids)
    if question is None:
        return JsonResponse({"ok": True, "empty": True, "session_limit": max(1, min(user.session_question_limit, 50))})
    return JsonResponse({"ok": True, "question": question, "session_limit": max(1, min(user.session_question_limit, 50))})


@require_POST
def learn_answer(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    try:
        payload = _json_body(request)
        word_id = int(payload.get("word_id"))
        answer = str(payload.get("answer", ""))
        exercise_type = str(payload.get("exercise_type", ""))
    except (TypeError, ValueError):
        return _json_error("Invalid learning answer payload.")

    if exercise_type not in {"practice_en_ru", "practice_ru_en", "listening_word", "listening_translate"}:
        return _json_error("Unknown exercise type.")

    try:
        result = submit_learning_text_answer(user, word_id, answer, exercise_type)
    except ValueError as exc:
        return _json_error(str(exc))
    return JsonResponse({"ok": True, **result})


@require_GET
def practice_question(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    mode = request.GET.get("mode", "classic")
    if mode not in {"classic", "reverse", "review"}:
        return _json_error("Unknown practice mode.")
    question = build_choice_question(user, mode)
    if question is None:
        return JsonResponse({"ok": True, "empty": True})
    return JsonResponse({"ok": True, "question": question})


@require_POST
def practice_answer(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    try:
        payload = _json_body(request)
        word_id = int(payload.get("word_id"))
        answer = str(payload.get("answer", ""))
        mode = str(payload.get("mode", "classic"))
    except (TypeError, ValueError):
        return _json_error("Invalid practice answer payload.")

    return JsonResponse({"ok": True, **submit_choice_answer(user, word_id, answer, mode)})


@require_GET
def listening_question(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    mode = request.GET.get("mode", "word")
    if mode not in {"word", "translate"}:
        return _json_error("Unknown listening mode.")
    question = build_listening_question(user, mode)
    if question is None:
        return JsonResponse({"ok": True, "empty": True})
    return JsonResponse({"ok": True, "question": question})


@require_POST
def listening_answer(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    try:
        payload = _json_body(request)
        word_id = int(payload.get("word_id"))
        answer = str(payload.get("answer", ""))
        mode = str(payload.get("mode", "word"))
    except (TypeError, ValueError):
        return _json_error("Invalid listening answer payload.")

    return JsonResponse({"ok": True, **submit_listening_answer(user, word_id, answer, mode)})


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

    try:
        word_id = int(request.POST.get("word_id"))
    except (TypeError, ValueError):
        return _json_error("Invalid speaking payload.")

    audio_file = request.FILES.get("audio")
    if audio_file is None:
        return _json_error("Audio file is required.")

    suffix = os.path.splitext(audio_file.name or "")[1] or ".webm"
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            for chunk in audio_file.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name

        transcript = transcribe_speech_file(temp_path)
        return JsonResponse({"ok": True, **evaluate_speaking_answer(user, word_id, transcript)})
    except Exception as exc:
        return _json_error(f"Speech recognition failed: {exc}", status=400)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


@require_GET
def word_audio(request: HttpRequest, word_id: int) -> JsonResponse | FileResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    try:
        item = VocabularyItem.objects.get(id=word_id, user=user)
    except VocabularyItem.DoesNotExist:
        return _json_error("Word not found.", status=404)

    import asyncio
    import os
    from .tts import get_audio_path, generate_tts_audio

    try:
        audio_path = get_audio_path(item.word)
        if not audio_path or not os.path.exists(audio_path):
            audio_path = asyncio.run(generate_tts_audio(item.word))
    except Exception:
        return _json_error("Audio is temporarily unavailable.", status=503)
    return FileResponse(open(audio_path, "rb"), content_type="audio/mpeg")


@require_GET
def word_image(request: HttpRequest, word_id: int) -> JsonResponse | FileResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    try:
        item = VocabularyItem.objects.get(id=word_id, user=user)
    except VocabularyItem.DoesNotExist:
        return _json_error("Word not found.", status=404)

    image_path = get_word_image_file(item)
    if image_path is None:
        return _json_error("Image not found.", status=404)

    content_type, _ = mimetypes.guess_type(image_path.name)
    return FileResponse(open(image_path, "rb"), content_type=content_type or "image/jpeg")


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
    return FileResponse(open(image_path, "rb"), content_type=content_type or "image/png")


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

    try:
        payload = _json_body(request)
        base = str(payload.get("base"))
        answer = str(payload.get("answer"))
        correct_pair = str(payload.get("correct_pair"))
    except (TypeError, ValueError):
        return _json_error("Invalid irregular payload.")

    correct = answer == correct_pair
    if correct:
        update_irregular_progress(user, base, True)
    update_learning_streak(user)
    return JsonResponse({"ok": True, "correct": correct, "correct_answer": correct_pair, "progress": build_user_progress(user)})


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
        cards = [serialize_word(item) for item in get_ordered_unlearned_words(user, count=count)]
    return JsonResponse({"ok": True, "items": cards})


@require_POST
def study_answer(request: HttpRequest) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    try:
        payload = _json_body(request)
        word_id = int(payload.get("word_id"))
        correct = bool(payload.get("correct"))
    except (TypeError, ValueError):
        return _json_error("Invalid answer payload.")

    try:
        item = VocabularyItem.objects.get(id=word_id, user=user)
    except VocabularyItem.DoesNotExist:
        return _json_error("Word not found.", status=404)

    updated = update_word_progress(item.id, correct=correct)
    update_learning_streak(user)
    return JsonResponse({"ok": True, "item": serialize_word(updated), "progress": build_user_progress(user)})
