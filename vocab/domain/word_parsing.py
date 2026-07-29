"""Pure parsing rules for adding several vocabulary words at once."""

from __future__ import annotations

from dataclasses import dataclass
import re

from vocab.utils import clean_word


@dataclass(frozen=True)
class ParsedWordEntry:
    word: str
    translation_hint: str | None = None


def parse_word_batch(text: str) -> list[ParsedWordEntry]:
    def normalize_word_part(raw: str) -> str:
        value = re.sub(r"^[•*·\-]+\s*", "", (raw or "").strip())
        value = re.sub(r"\s*:\s*$", "", value)
        value = re.sub(
            r"\s*\((?:v|adj|adv|n|noun|verb|adjective|adverb|phrase)\)\s*$",
            "",
            value,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+(?:v|adj|adv|n)\s*$", "", value, flags=re.IGNORECASE).strip()

    def split_word_and_translation(line: str) -> tuple[str, str | None]:
        for separator in (" - ", " — ", " – "):
            if separator in line:
                word_part, translation_hint = line.split(separator, 1)
                return normalize_word_part(word_part), translation_hint.strip() or None
        if ":" in line:
            word_part, translation_hint = line.split(":", 1)
            normalized = normalize_word_part(word_part)
            if normalized and translation_hint.strip():
                return normalized, translation_hint.strip()
        cyrillic_match = re.search(r"[А-Яа-яЁё]", line)
        if cyrillic_match:
            word_part = normalize_word_part(line[: cyrillic_match.start()])
            translation_hint = line[cyrillic_match.start() :].strip()
            if word_part and translation_hint:
                return word_part, translation_hint
        return normalize_word_part(line), None

    entries: list[ParsedWordEntry] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        word_part, translation_hint = split_word_and_translation(line)
        cleaned = clean_word(word_part)
        if cleaned:
            entries.append(ParsedWordEntry(cleaned, translation_hint or None))
    return entries
