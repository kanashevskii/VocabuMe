from openai import OpenAI
from decouple import config

client = OpenAI(api_key=config("OPENAI_API_KEY"))

def detect_language(text):
    return "ru" if any("а" <= c <= "я" or "А" <= c <= "Я" for c in text) else "en"

def generate_word_data(word: str) -> dict:
    lang = detect_language(word)
    if lang == "ru":
        # Получаем английский перевод
        prompt_translate = f"Translate the Russian word or phrase \"{word}\" to English. Just give the main English equivalent."
        translation_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_translate}],
            temperature=0.2,
        )
        word = translation_resp.choices[0].message.content.strip()

    # Основной prompt
    prompt = f"""
You are an English language assistant.

For the English word or phrase "{word}", return the following in JSON format:
1. "translation": most common Russian translation
2. "transcription": IPA transcription of the English word
3. "example": short sentence in English
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

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.strip("```json").strip("` \n")

        return eval(content) | {"word": word}
    except Exception as e:
        print("OpenAI error:", e)
        return None
