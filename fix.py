import os
import django

# Подключаем настройки Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

# Теперь всё доступно
from vocab.models import VocabularyItem
from vocab.openai_utils import client
from time import sleep

BATCH_SIZE = 20

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

        if raw.startswith("```json"):
            raw = raw.strip("```json").strip("` \n")

        parsed = eval(raw)

        for item in batch:
            pos = parsed.get(item.word)
            if pos:
                item.part_of_speech = pos
                item.save()
                print(f"✅ {item.word} → {pos}")
            else:
                print(f"⚠️  Не найдено: {item.word}")

    except Exception as e:
        print("❌ Ошибка:", e)
        sleep(5)
