import os
import django
import json
import ast
import re

# Подключаем настройки Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

# Теперь всё доступно
from vocab.models import VocabularyItem
from vocab.openai_utils import client
from time import sleep

BATCH_SIZE = 20

def _strip_code_fences(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    end = stripped.rfind("```")
    if end == -1:
        return stripped
    body = stripped[3:end].strip()
    if body.lower().startswith("json"):
        body = body[4:].strip()
    return body


def _parse_json_dict(payload: str) -> dict:
    cleaned = _strip_code_fences(payload)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Model returned non-dict payload")
    return parsed

while True:
    batch = list(
        VocabularyItem.objects
        .filter(part_of_speech="unknown")
        .exclude(word__icontains=" ")
        .order_by("id")[:BATCH_SIZE]
    )

    if not batch:
        print("🎉 Готово: все слова размечены.")
        break

    words = [item.word for item in batch]
    word_str = ", ".join(f'"{w}"' for w in words)

    prompt = f"""
You are a professional linguist.

For the following English words: {word_str}

Return a JSON dictionary where each word maps to its part of speech.

Only use: noun, verb, adjective, adverb, pronoun, preposition, conjunction, interjection, phrase

Example:
{{
  "go": "verb",
  "apple": "noun"
}}

Return only JSON, no extra text.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        raw = resp.choices[0].message.content.strip()
        parsed = _parse_json_dict(raw)

        for item in batch:
            pos = parsed.get(item.word)
            if pos:
                item.part_of_speech = re.sub(r"[^a-z]+", "", str(pos).lower())[:50] or "unknown"
                item.save()
                print(f"✅ {item.word} → {pos}")
            else:
                print(f"⚠️  Не найдено: {item.word}")

    except Exception as e:
        print("❌ Ошибка:", e)
        sleep(5)
