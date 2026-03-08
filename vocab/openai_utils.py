from openai import OpenAI, RateLimitError
import logging
import json
import ast
import re
import base64
import time
from pathlib import Path
from contextlib import contextmanager

from django.db import close_old_connections, connection

from core.env import env, get_openai_api_key

client = OpenAI(api_key=get_openai_api_key())
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRAFT_IMAGE_DIR = PROJECT_ROOT / "media" / "draft_images"
TEXT_MODEL = "gpt-5-mini"
IMAGE_MODEL = "gpt-image-1.5"
TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"
OPENAI_QUEUE_LOCK_KEY = env("OPENAI_QUEUE_LOCK_KEY", cast=int, default=841725)
OPENAI_QUEUE_WAIT_SECONDS = env("OPENAI_QUEUE_WAIT_SECONDS", cast=int, default=180)


def detect_language(text):
    return "ru" if any("а" <= c <= "я" or "А" <= c <= "Я" for c in text) else "en"


def _strip_code_fences(text: str) -> str:
    """Remove ``` and language hint if the model wrapped JSON in a fence."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    end = stripped.rfind("```")
    if end != -1:
        body = stripped[3:end].strip()
        if body.lower().startswith("json"):
            body = body[4:].strip()
        return body
    return stripped


@contextmanager
def openai_request_slot(label: str):
    close_old_connections()
    started_at = time.monotonic()
    acquired = False

    try:
        with connection.cursor() as cursor:
            while True:
                cursor.execute(
                    "SELECT pg_try_advisory_lock(%s)", [OPENAI_QUEUE_LOCK_KEY]
                )
                acquired = bool(cursor.fetchone()[0])
                if acquired:
                    waited = time.monotonic() - started_at
                    if waited >= 0.25:
                        logging.info(
                            "OpenAI queue acquired for %s after %.2fs", label, waited
                        )
                    break
                if time.monotonic() - started_at >= OPENAI_QUEUE_WAIT_SECONDS:
                    raise TimeoutError(f"OpenAI queue wait timed out for {label}")
                time.sleep(0.2)

        yield
    finally:
        if acquired:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT pg_advisory_unlock(%s)", [OPENAI_QUEUE_LOCK_KEY]
                    )
            except Exception:
                logging.exception(
                    "Failed to release OpenAI advisory lock for %s", label
                )
        close_old_connections()


def _parse_model_json(payload: str) -> dict:
    cleaned = _strip_code_fences(payload)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(cleaned)
        except Exception:
            logging.exception("Failed to parse model JSON: %s", cleaned)
            raise


def _infer_part_from_translation(translation_hint: str | None) -> str | None:
    if not translation_hint:
        return None
    t = translation_hint.strip().lower()
    if re.match(r"^[а-яё-]+(?:\s*/\s*[а-яё-]+)*$", t):
        variants = [part.strip() for part in t.split("/") if part.strip()]
        if variants and all(variant.endswith(("о", "е")) for variant in variants):
            return "adverb"
    if t.endswith(("ть", "ться")):
        return "verb"
    if t.endswith(("о", "е")):
        return "adverb"
    if t.endswith(("ый", "ий", "ая", "ое", "ее", "ие", "ые", "ой", "ей", "ых", "их")):
        return "adjective"
    if " " in t:
        return "phrase"
    return "noun"


def _looks_like_verb_usage(word: str, example: str) -> bool:
    base = word.strip().lower()
    text = example.strip().lower()
    pronouns = ["i", "you", "he", "she", "we", "they", "it"]
    if re.search(rf"\bto\s+{re.escape(base)}\b", text):
        return True
    if re.search(rf"\b({'|'.join(pronouns)})\s+{re.escape(base)}(s|ed|ing)?\b", text):
        return True
    if re.search(rf"\b{re.escape(base)}(s|ed|ing)\b", text):
        return True
    return False


def _is_valid_example(word: str, expected_part: str | None, example: str) -> bool:
    if not expected_part:
        return True
    part = expected_part.lower()
    if part == "noun":
        return not _looks_like_verb_usage(word, example)
    if part == "verb":
        return (
            _looks_like_verb_usage(word, example)
            or f"{word.lower()} " in example.lower()
        )
    return True


def _example_matches_course(course_code: str, example: str) -> bool:
    text = (example or "").strip()
    if not text:
        return False
    if course_code == "ka":
        return bool(re.search(r"[\u10A0-\u10FF]", text))
    return bool(re.search(r"[A-Za-z]", text))


def _is_valid_example_for_course(
    course_code: str, word: str, expected_part: str | None, example: str
) -> bool:
    return _example_matches_course(course_code, example) and _is_valid_example(
        word, expected_part, example
    )


def _build_prompt(
    word: str,
    effective_part: str | None,
    translation_hint: str | None,
    course_code: str = "en",
    extra_note: str = "",
) -> str:
    hint_text = ""
    if effective_part:
        sense_rules = {
            "noun": (
                "Use it strictly as a noun; avoid verb or adjective usages. "
                "Example must use the noun form only (no conjugated verbs like 'prides')."
            ),
            "verb": (
                "Use it strictly as a verb in the expected sense; avoid noun or adjective usages. "
                "Example should use a proper verb form."
            ),
            "adjective": "Use it strictly as an adjective; avoid noun or verb usages.",
        }
        rule = sense_rules.get(
            effective_part,
            f"Use it strictly as a {effective_part}. Avoid other parts of speech.",
        )
        hint_text += (
            f"The user expects the {word!r} sense as a {effective_part}. " f"{rule}\n"
        )
    if translation_hint:
        hint_text += (
            f"Match the meaning of the Russian translation '{translation_hint}'. "
            "The example and translation must reflect this sense.\n"
        )
    if extra_note:
        hint_text += extra_note + "\n"

    if course_code == "ka":
        prompt = f"""
You are a Georgian language assistant.

{hint_text}For the Georgian word or phrase "{word}", return the following in JSON format:
1. "translation": most common Russian translation (matching the same sense and part of speech)
2. "transcription": IPA transcription of the Georgian word
3. "example": short sentence in Georgian that uses the same sense and part of speech
4. "part_of_speech": word type — one of: noun, verb, adjective, adverb, pronoun, preposition, conjunction, interjection, phrase

Example format:
{{
  "translation": "перевод",
  "transcription": "madlɔba",
  "example": "დიდი მადლობა დახმარებისთვის.",
  "part_of_speech": "interjection"
}}

Only return the JSON object. No extra text.
"""
    else:
        prompt = f"""
You are an English language assistant.

{hint_text}For the English word or phrase "{word}", return the following in JSON format:
1. "translation": most common Russian translation (matching the same sense and part of speech)
2. "transcription": IPA transcription of the English word
3. "example": short sentence in English that uses the same sense and part of speech
4. "part_of_speech": word type — one of: noun, verb, adjective, adverb, pronoun, preposition, conjunction, interjection, phrase

Example format:
{{
  "translation": "перевод",
  "transcription": "həʊˈevə",
  "example": "However, it was too late.",
  "part_of_speech": "adverb"
}}

Only return the JSON object. No extra text.
"""
    return prompt


def generate_word_data(
    word: str,
    part_hint: str | None = None,
    translation_hint: str | None = None,
    course_code: str = "en",
) -> dict:
    lang = detect_language(word)
    if course_code == "en" and lang == "ru":
        prompt_translate = f'Translate the Russian word or phrase "{word}" to English. Just give the main English equivalent.'
        with openai_request_slot("translate-ru-to-en"):
            translation_resp = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[{"role": "user", "content": prompt_translate}],
            )
        word = _strip_code_fences(translation_resp.choices[0].message.content)

    effective_part = part_hint or _infer_part_from_translation(translation_hint)

    prompt_note = ""
    last_error = None
    for _ in range(3):
        prompt = _build_prompt(
            word, effective_part, translation_hint, course_code=course_code, extra_note=prompt_note
        )
        try:
            with openai_request_slot("generate-word-data"):
                response = client.chat.completions.create(
                    model=TEXT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
            content = response.choices[0].message.content.strip()
            parsed = _parse_model_json(content)
            if effective_part:
                parsed["part_of_speech"] = effective_part

            example_ok = _is_valid_example_for_course(
                course_code, word, effective_part, parsed.get("example", "")
            )
            pos_ok = (
                not effective_part
                or parsed.get("part_of_speech", "").lower() == effective_part
            )
            if example_ok and pos_ok:
                return parsed | {"word": word, "course_code": course_code}

            reason = (
                "example doesn't match expected part"
                if not example_ok
                else "part_of_speech mismatch"
            )
            last_error = reason
            prompt_note = (
                f"Previous attempt invalid ({reason}). Rewrite strictly as a {effective_part or 'given'} sense; "
                "avoid verb conjugations if noun."
            )
        except Exception as e:
            logging.exception("OpenAI error: %s", e)
            last_error = str(e)
            prompt_note = "Previous attempt failed to parse. Return valid JSON only."

    logging.warning("Falling back after validation failures: %s", last_error)
    return None


def generate_word_data_batch(entries: list[dict]) -> list[dict | None]:
    if not entries:
        return []

    course_code = (entries[0].get("course_code") or "en").strip().lower() or "en"

    payload = [
        {
            "word": entry["word"],
            "translation_hint": entry.get("translation_hint") or "",
            "part_hint": entry.get("part_hint") or "",
        }
        for entry in entries
    ]
    if course_code == "ka":
        prompt = f"""
You are a Georgian language assistant.

For each entry in the JSON array below, return one object in the same order.
Each result object must contain:
1. "word": original Georgian word or phrase
2. "translation": the matching Russian translation for the intended sense
3. "transcription": IPA transcription
4. "example": short Georgian sentence using the same sense
5. "part_of_speech": one of noun, verb, adjective, adverb, pronoun, preposition, conjunction, interjection, phrase

Important:
- Respect the provided translation_hint if present.
- Respect the provided part_hint if present.
- Keep phrase meanings intact.
- Return Georgian examples, not English examples.
- Return JSON array only, no prose.

Input:
{json.dumps(payload, ensure_ascii=False)}
"""
    else:
        prompt = f"""
You are an English language assistant.

For each entry in the JSON array below, return one object in the same order.
Each result object must contain:
1. "word": original English word or phrase
2. "translation": the matching Russian translation for the intended sense
3. "transcription": IPA transcription
4. "example": short English sentence using the same sense
5. "part_of_speech": one of noun, verb, adjective, adverb, pronoun, preposition, conjunction, interjection, phrase

Important:
- Respect the provided translation_hint if present.
- Respect the provided part_hint if present.
- Keep phrase meanings intact.
- Return JSON array only, no prose.

Input:
{json.dumps(payload, ensure_ascii=False)}
"""
    try:
        with openai_request_slot("generate-word-data-batch"):
            response = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
        content = response.choices[0].message.content.strip()
        parsed = _parse_model_json(content)
        if not isinstance(parsed, list) or len(parsed) != len(entries):
            raise ValueError("Batch response shape mismatch")

        normalized_results: list[dict | None] = []
        for source, item in zip(entries, parsed, strict=False):
            if not isinstance(item, dict):
                normalized_results.append(None)
                continue
            effective_part = source.get("part_hint") or _infer_part_from_translation(
                source.get("translation_hint")
            )
            if effective_part:
                item["part_of_speech"] = effective_part

            example_ok = _is_valid_example_for_course(
                course_code, source["word"], effective_part, item.get("example", "")
            )
            pos_ok = (
                not effective_part
                or item.get("part_of_speech", "").lower() == effective_part
            )
            if not example_ok or not pos_ok:
                normalized_results.append(None)
                continue
            normalized_results.append(
                item | {"word": source["word"], "course_code": course_code}
            )
        return normalized_results
    except Exception as exc:
        logging.exception("Batch OpenAI generation failed: %s", exc)
        return [
            generate_word_data(
                entry["word"],
                part_hint=entry.get("part_hint"),
                translation_hint=entry.get("translation_hint"),
                course_code=entry.get("course_code", course_code),
            )
            for entry in entries
        ]


def build_visual_prompt(
    word: str, translation: str, part_of_speech: str, example: str = ""
) -> str | None:
    prompt = f"""
You create short visual briefs for vocabulary learning images.

Task:
- English word or phrase: "{word}"
- Russian translation: "{translation}"
- Part of speech: "{part_of_speech}"
- Example context: "{example}"

Return JSON only:
{{
  "prompt": "A concise image-generation prompt in English"
}}

Rules:
- Show the actual meaning of the word or phrase, not a random literal association.
- If the word is abstract, generate a clear symbolic but understandable scene.
- Mention the action/object/context explicitly.
- The image must help a learner remember the exact translation.
- Avoid text, captions, watermarks, UI, collages, split screens.
- Keep it suitable for a clean educational flashcard.
"""
    try:
        with openai_request_slot("build-visual-prompt"):
            response = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
        content = response.choices[0].message.content.strip()
        parsed = _parse_model_json(content)
        return parsed.get("prompt", "").strip() or None
    except Exception as exc:
        logging.exception("Failed to build visual prompt: %s", exc)
        return None


def generate_card_image(prompt: str, slug: str) -> str:
    DRAFT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{slug}.png"
    destination = DRAFT_IMAGE_DIR / filename

    response = None
    for attempt in range(6):
        try:
            with openai_request_slot("generate-card-image"):
                response = client.images.generate(
                    model=IMAGE_MODEL,
                    prompt=prompt,
                    size="1024x1024",
                    quality="low",
                )
            break
        except RateLimitError as exc:
            if attempt >= 5:
                raise
            message = str(exc)
            match = re.search(r"try again in (\d+)s", message, re.IGNORECASE)
            delay = int(match.group(1)) + 1 if match else 15
            time.sleep(delay)
    if response is None:
        raise RuntimeError("Image generation did not return a response.")
    image_b64 = response.data[0].b64_json
    destination.write_bytes(base64.b64decode(image_b64))
    return str(destination.relative_to(PROJECT_ROOT))


def transcribe_speech_file(audio_path: str) -> str:
    with open(audio_path, "rb") as audio_file:
        with openai_request_slot("transcribe-speech"):
            response = client.audio.transcriptions.create(
                model=TRANSCRIBE_MODEL,
                file=audio_file,
                language="en",
            )
    text = getattr(response, "text", "") or ""
    return text.strip()
