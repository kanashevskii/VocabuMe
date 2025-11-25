from openai import OpenAI
from decouple import config
import logging
import json
import ast
import re

client = OpenAI(api_key=config("OPENAI_API_KEY"))


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
    if t.endswith(("ть", "ться")):
        return "verb"
    if t.endswith(("ый", "ий", "ая", "ое", "ее", "ие", "ые", "ой", "ей", "ых", "их")):
        return "adjective"
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
        return _looks_like_verb_usage(word, example) or f"{word.lower()} " in example.lower()
    return True


def _build_prompt(word: str, effective_part: str | None, translation_hint: str | None, extra_note: str = "") -> str:
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
        rule = sense_rules.get(effective_part, f"Use it strictly as a {effective_part}. Avoid other parts of speech.")
        hint_text += (
            f"The user expects the {word!r} sense as a {effective_part}. "
            f"{rule}\n"
        )
    if translation_hint:
        hint_text += (
            f"Match the meaning of the Russian translation '{translation_hint}'. "
            "The example and translation must reflect this sense.\n"
        )
    if extra_note:
        hint_text += extra_note + "\n"

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


def generate_word_data(word: str, part_hint: str | None = None, translation_hint: str | None = None) -> dict:
    lang = detect_language(word)
    if lang == "ru":
        prompt_translate = f"Translate the Russian word or phrase \"{word}\" to English. Just give the main English equivalent."
        translation_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_translate}],
            temperature=0.2,
        )
        word = _strip_code_fences(translation_resp.choices[0].message.content)

    effective_part = part_hint or _infer_part_from_translation(translation_hint)

    prompt_note = ""
    last_error = None
    for _ in range(3):
        prompt = _build_prompt(word, effective_part, translation_hint, prompt_note)
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.15,
            )
            content = response.choices[0].message.content.strip()
            parsed = _parse_model_json(content)
            if effective_part:
                parsed["part_of_speech"] = effective_part

            example_ok = _is_valid_example(word, effective_part, parsed.get("example", ""))
            pos_ok = not effective_part or parsed.get("part_of_speech", "").lower() == effective_part
            if example_ok and pos_ok:
                return parsed | {"word": word}

            reason = "example doesn't match expected part" if not example_ok else "part_of_speech mismatch"
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
