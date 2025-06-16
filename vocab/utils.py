import re
import string
from deep_translator import GoogleTranslator

PUNCTUATION = string.punctuation.replace("'", "")
_punct_regex = re.compile(f"[{re.escape(PUNCTUATION)}]")

def clean_word(word: str) -> str:
    """Return word lowercased with punctuation removed (except apostrophes)."""
    if not isinstance(word, str):
        return ""
    cleaned = _punct_regex.sub("", word)
    cleaned = cleaned.strip().lower()
    return cleaned


def translate_to_ru(text: str) -> str:
    """Translate the given text to Russian using GoogleTranslator."""
    try:
        return GoogleTranslator(source="auto", target="ru").translate(text)
    except Exception as e:
        print("Translation error:", e)
        return ""
