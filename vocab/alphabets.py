from __future__ import annotations

import random


ENGLISH_ALPHABET = [
    {"symbol": "A", "name": "A", "transcription": "eɪ", "hint": "эй"},
    {"symbol": "B", "name": "B", "transcription": "biː", "hint": "би"},
    {"symbol": "C", "name": "C", "transcription": "siː", "hint": "си"},
    {"symbol": "D", "name": "D", "transcription": "diː", "hint": "ди"},
    {"symbol": "E", "name": "E", "transcription": "iː", "hint": "и"},
    {"symbol": "F", "name": "F", "transcription": "ef", "hint": "эф"},
    {"symbol": "G", "name": "G", "transcription": "dʒiː", "hint": "джи"},
    {"symbol": "H", "name": "H", "transcription": "eɪtʃ", "hint": "эйч"},
    {"symbol": "I", "name": "I", "transcription": "aɪ", "hint": "ай"},
    {"symbol": "J", "name": "J", "transcription": "dʒeɪ", "hint": "джей"},
    {"symbol": "K", "name": "K", "transcription": "keɪ", "hint": "кей"},
    {"symbol": "L", "name": "L", "transcription": "el", "hint": "эл"},
    {"symbol": "M", "name": "M", "transcription": "em", "hint": "эм"},
    {"symbol": "N", "name": "N", "transcription": "en", "hint": "эн"},
    {"symbol": "O", "name": "O", "transcription": "əʊ", "hint": "оу"},
    {"symbol": "P", "name": "P", "transcription": "piː", "hint": "пи"},
    {"symbol": "Q", "name": "Q", "transcription": "kjuː", "hint": "кью"},
    {"symbol": "R", "name": "R", "transcription": "ɑːr", "hint": "ар"},
    {"symbol": "S", "name": "S", "transcription": "es", "hint": "эс"},
    {"symbol": "T", "name": "T", "transcription": "tiː", "hint": "ти"},
    {"symbol": "U", "name": "U", "transcription": "juː", "hint": "ю"},
    {"symbol": "V", "name": "V", "transcription": "viː", "hint": "ви"},
    {"symbol": "W", "name": "W", "transcription": "ˈdʌbəl juː", "hint": "дабл-ю"},
    {"symbol": "X", "name": "X", "transcription": "eks", "hint": "экс"},
    {"symbol": "Y", "name": "Y", "transcription": "waɪ", "hint": "уай"},
    {"symbol": "Z", "name": "Z", "transcription": "ziː", "hint": "зи"},
]

GEORGIAN_ALPHABET = [
    {"symbol": "ა", "name": "ანი", "transcription": "ɑ", "hint": "а"},
    {"symbol": "ბ", "name": "ბანი", "transcription": "b", "hint": "б"},
    {"symbol": "გ", "name": "განი", "transcription": "ɡ", "hint": "г"},
    {"symbol": "დ", "name": "დონი", "transcription": "d", "hint": "д"},
    {"symbol": "ე", "name": "ენი", "transcription": "e", "hint": "э"},
    {"symbol": "ვ", "name": "ვინი", "transcription": "v", "hint": "в"},
    {"symbol": "ზ", "name": "ზენი", "transcription": "z", "hint": "з"},
    {"symbol": "თ", "name": "თანი", "transcription": "tʰ", "hint": "т с придыханием"},
    {"symbol": "ი", "name": "ინი", "transcription": "i", "hint": "и"},
    {"symbol": "კ", "name": "კანი", "transcription": "kʼ", "hint": "к"},
    {"symbol": "ლ", "name": "ლასი", "transcription": "l", "hint": "л"},
    {"symbol": "მ", "name": "მანი", "transcription": "m", "hint": "м"},
    {"symbol": "ნ", "name": "ნარი", "transcription": "n", "hint": "н"},
    {"symbol": "ო", "name": "ონი", "transcription": "ɔ", "hint": "о"},
    {"symbol": "პ", "name": "პარი", "transcription": "pʼ", "hint": "п"},
    {"symbol": "ჟ", "name": "ჟანი", "transcription": "ʒ", "hint": "ж"},
    {"symbol": "რ", "name": "რაე", "transcription": "r", "hint": "р"},
    {"symbol": "ს", "name": "სანი", "transcription": "s", "hint": "с"},
    {"symbol": "ტ", "name": "ტარი", "transcription": "tʼ", "hint": "т"},
    {"symbol": "უ", "name": "უნი", "transcription": "u", "hint": "у"},
    {"symbol": "ფ", "name": "ფარი", "transcription": "pʰ", "hint": "пх"},
    {"symbol": "ქ", "name": "ქანი", "transcription": "kʰ", "hint": "кх"},
    {"symbol": "ღ", "name": "ღანი", "transcription": "ʁ", "hint": "гортанное г"},
    {"symbol": "ყ", "name": "ყარი", "transcription": "qʼ", "hint": "къ"},
    {"symbol": "შ", "name": "შინი", "transcription": "ʃ", "hint": "ш"},
    {"symbol": "ჩ", "name": "ჩინი", "transcription": "tʃʰ", "hint": "ч"},
    {"symbol": "ც", "name": "ცანი", "transcription": "tsʰ", "hint": "ц"},
    {"symbol": "ძ", "name": "ძილი", "transcription": "dz", "hint": "дз"},
    {"symbol": "წ", "name": "წილი", "transcription": "tsʼ", "hint": "цʼ"},
    {"symbol": "ჭ", "name": "ჭარი", "transcription": "tʃʼ", "hint": "чʼ"},
    {"symbol": "ხ", "name": "ხანი", "transcription": "x", "hint": "х"},
    {"symbol": "ჯ", "name": "ჯანი", "transcription": "dʒ", "hint": "дж"},
    {"symbol": "ჰ", "name": "ჰაე", "transcription": "h", "hint": "х/һ"},
]

ALPHABETS = {
    "en": ENGLISH_ALPHABET,
    "ka": GEORGIAN_ALPHABET,
}


def get_alphabet(course_code: str) -> list[dict]:
    return ALPHABETS.get(course_code, ENGLISH_ALPHABET)


def get_alphabet_letter(course_code: str, symbol: str) -> dict | None:
    for item in get_alphabet(course_code):
        if item["symbol"] == symbol:
            return item
    return None


def get_random_alphabet_options(course_code: str, current_symbol: str, count: int = 3) -> list[str]:
    entries = [item["symbol"] for item in get_alphabet(course_code) if item["symbol"] != current_symbol]
    random.shuffle(entries)
    return entries[:count]
