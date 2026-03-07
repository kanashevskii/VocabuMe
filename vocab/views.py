from __future__ import annotations

import json
import mimetypes
import os
import tempfile
from datetime import time as dt_time

from decouple import config
from django.http import FileResponse, HttpRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from .models import AddWordDraft, TelegramUser, VocabularyItem
from .openai_utils import generate_word_data, transcribe_speech_file
from .services import (
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
    parse_word_batch,
    refresh_draft_language_data,
    regenerate_word_image,
    resolve_shared_image_path,
    serialize_user,
    serialize_draft,
    serialize_word,
    submit_choice_answer,
    submit_listening_answer,
    evaluate_speaking_answer,
    update_learning_streak,
    update_irregular_progress,
    update_word_progress,
    upsert_telegram_user,
    word_already_exists,
)
from .telegram_auth import TelegramAuthError, verify_login_widget, verify_webapp_init_data
from .utils import normalize_timezone_value

SESSION_USER_KEY = "telegram_user_id"


def _json_body(request: HttpRequest) -> dict:
    if not request.body:
        return {}
    try:
        return json.loads(request.body)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON body.") from exc


def _json_error(message: str, status: int = 400) -> JsonResponse:
    return JsonResponse({"ok": False, "error": message}, status=status)


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

    entries = parse_word_batch(payload.get("text", ""))
    if not entries:
        return _json_error("Add one word or phrase.", status=400)
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

        for entry in entries:
            if word_already_exists(user, entry.word):
                skipped.append({"word": entry.word, "reason": "duplicate"})
                continue

            generated = generate_word_data(entry.word, translation_hint=entry.translation_hint)
            if not generated:
                failed.append({"word": entry.word, "reason": "generation_failed"})
                continue

            try:
                draft = create_word_draft(user, entry.word, generated, translation_hint=entry.translation_hint)
                if draft.translation_confirmed:
                    draft = ensure_draft_image(draft)
                drafts.append(serialize_draft(draft))
            except Exception:
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
        draft = ensure_draft_image(draft)
        step = "confirm_image"
    return JsonResponse({"ok": True, "draft": serialize_draft(draft), "step": step})


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
    draft = ensure_draft_image(draft)
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

    draft = ensure_draft_image(draft, force_regenerate=True)
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


@require_http_methods(["DELETE"])
def word_draft_delete(request: HttpRequest, draft_id: int) -> JsonResponse:
    user = _require_user(request)
    if isinstance(user, JsonResponse):
        return user

    draft = _get_draft_for_user(user, draft_id)
    if isinstance(draft, JsonResponse):
        return draft
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

    item = regenerate_word_image(item)
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
                    "repeat_threshold": user.repeat_threshold,
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
        user.repeat_threshold = max(1, min(int(payload.get("repeat_threshold", user.repeat_threshold)), 10))
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
    return JsonResponse({"ok": True})


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
