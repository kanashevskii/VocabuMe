import random

IRREGULAR_VERBS = [
    {"base": "be", "past": "was/were", "participle": "been", "translation": "быть"},
    {"base": "become", "past": "became", "participle": "become", "translation": "становиться"},
    {"base": "begin", "past": "began", "participle": "begun", "translation": "начинать"},
    {"base": "break", "past": "broke", "participle": "broken", "translation": "ломать"},
    {"base": "bring", "past": "brought", "participle": "brought", "translation": "приносить"},
    {"base": "build", "past": "built", "participle": "built", "translation": "строить"},
    {"base": "buy", "past": "bought", "participle": "bought", "translation": "покупать"},
    {"base": "catch", "past": "caught", "participle": "caught", "translation": "ловить"},
    {"base": "choose", "past": "chose", "participle": "chosen", "translation": "выбирать"},
    {"base": "come", "past": "came", "participle": "come", "translation": "приходить"},
    {"base": "cost", "past": "cost", "participle": "cost", "translation": "стоить"},
    {"base": "do", "past": "did", "participle": "done", "translation": "делать"},
    {"base": "drink", "past": "drank", "participle": "drunk", "translation": "пить"},
    {"base": "drive", "past": "drove", "participle": "driven", "translation": "водить"},
    {"base": "eat", "past": "ate", "participle": "eaten", "translation": "есть"},
    {"base": "fall", "past": "fell", "participle": "fallen", "translation": "падать"},
    {"base": "feel", "past": "felt", "participle": "felt", "translation": "чувствовать"},
    {"base": "find", "past": "found", "participle": "found", "translation": "находить"},
    {"base": "fly", "past": "flew", "participle": "flown", "translation": "летать"},
    {"base": "forget", "past": "forgot", "participle": "forgotten", "translation": "забывать"},
    {"base": "get", "past": "got", "participle": "got", "translation": "получать"},
    {"base": "give", "past": "gave", "participle": "given", "translation": "давать"},
    {"base": "go", "past": "went", "participle": "gone", "translation": "идти"},
    {"base": "have", "past": "had", "participle": "had", "translation": "иметь"},
    {"base": "hear", "past": "heard", "participle": "heard", "translation": "слышать"},
    {"base": "hurt", "past": "hurt", "participle": "hurt", "translation": "ранить"},
    {"base": "keep", "past": "kept", "participle": "kept", "translation": "держать"},
    {"base": "know", "past": "knew", "participle": "known", "translation": "знать"},
    {"base": "leave", "past": "left", "participle": "left", "translation": "оставлять"},
    {"base": "lend", "past": "lent", "participle": "lent", "translation": "одалживать"},
    {"base": "let", "past": "let", "participle": "let", "translation": "позволять"},
    {"base": "lose", "past": "lost", "participle": "lost", "translation": "терять"},
    {"base": "make", "past": "made", "participle": "made", "translation": "делать"},
    {"base": "meet", "past": "met", "participle": "met", "translation": "встречать"},
    {"base": "pay", "past": "paid", "participle": "paid", "translation": "платить"},
    {"base": "put", "past": "put", "participle": "put", "translation": "класть"},
    {"base": "read", "past": "read", "participle": "read", "translation": "читать"},
    {"base": "ride", "past": "rode", "participle": "ridden", "translation": "ездить"},
    {"base": "run", "past": "ran", "participle": "run", "translation": "бежать"},
    {"base": "say", "past": "said", "participle": "said", "translation": "сказать"},
    {"base": "see", "past": "saw", "participle": "seen", "translation": "видеть"},
    {"base": "sell", "past": "sold", "participle": "sold", "translation": "продавать"},
    {"base": "send", "past": "sent", "participle": "sent", "translation": "отправлять"},
    {"base": "sing", "past": "sang", "participle": "sung", "translation": "петь"},
    {"base": "sit", "past": "sat", "participle": "sat", "translation": "сидеть"},
    {"base": "sleep", "past": "slept", "participle": "slept", "translation": "спать"},
    {"base": "speak", "past": "spoke", "participle": "spoken", "translation": "говорить"},
    {"base": "spend", "past": "spent", "participle": "spent", "translation": "тратить"},
    {"base": "stand", "past": "stood", "participle": "stood", "translation": "стоять"},
    {"base": "swim", "past": "swam", "participle": "swum", "translation": "плавать"},
    {"base": "take", "past": "took", "participle": "taken", "translation": "брать"},
    {"base": "teach", "past": "taught", "participle": "taught", "translation": "обучать"},
    {"base": "tell", "past": "told", "participle": "told", "translation": "рассказывать"},
    {"base": "think", "past": "thought", "participle": "thought", "translation": "думать"},
    {"base": "throw", "past": "threw", "participle": "thrown", "translation": "бросать"},
    {"base": "understand", "past": "understood", "participle": "understood", "translation": "понимать"},
    {"base": "wear", "past": "wore", "participle": "worn", "translation": "носить"},
    {"base": "write", "past": "wrote", "participle": "written", "translation": "писать"},
]

# Prepare wrong options for each verb (not random, generated once)
for v in IRREGULAR_VERBS:
    past = v["past"]
    part = v["participle"]
    base = v["base"]
    v["wrong_pairs"] = list(
        {
            f"{past} {base}",
            f"{base} {part}",
            f"{part} {past}",
        }
    )


def get_random_pairs(current, count=1, exclude=None):
    """Return unique random V2/V3 pairs excluding given options."""
    exclude = set(exclude or [])
    correct = f"{current['past']} {current['participle']}"
    exclude.add(correct)
    pairs = []
    verbs = [v for v in IRREGULAR_VERBS if v is not current]
    random.shuffle(verbs)
    for v in verbs:
        pair = f"{v['past']} {v['participle']}"
        if pair not in exclude and pair not in pairs:
            pairs.append(pair)
            if len(pairs) >= count:
                break
    return pairs
