import re
import string

PUNCTUATION = string.punctuation.replace("'", "")
_punct_regex = re.compile(f"[{re.escape(PUNCTUATION)}]")

def clean_word(word: str) -> str:
    """Return word lowercased with punctuation removed (except apostrophes)."""
    if not isinstance(word, str):
        return ""
    cleaned = _punct_regex.sub("", word)
    cleaned = cleaned.strip().lower()
    return cleaned
