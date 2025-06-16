IRREGULAR_VERBS = [
    {"base": "be", "past": "was/were", "participle": "been", "translation": "быть"},
    {"base": "become", "past": "became", "participle": "become", "translation": "становиться"},
    {"base": "begin", "past": "began", "participle": "begun", "translation": "начинать"},
    {"base": "break", "past": "broke", "participle": "broken", "translation": "ломать"},
    {"base": "bring", "past": "brought", "participle": "brought", "translation": "приносить"},
    {"base": "build", "past": "built", "participle": "built", "translation": "строить"},
    {"base": "buy", "past": "bought", "participle": "bought", "translation": "покупать"},
    {"base": "choose", "past": "chose", "participle": "chosen", "translation": "выбирать"},
    {"base": "come", "past": "came", "participle": "come", "translation": "приходить"},
    {"base": "do", "past": "did", "participle": "done", "translation": "делать"},
    {"base": "drink", "past": "drank", "participle": "drunk", "translation": "пить"},
    {"base": "eat", "past": "ate", "participle": "eaten", "translation": "есть"},
    {"base": "go", "past": "went", "participle": "gone", "translation": "идти"},
    {"base": "know", "past": "knew", "participle": "known", "translation": "знать"},
    {"base": "read", "past": "read", "participle": "read", "translation": "читать"},
    {"base": "run", "past": "ran", "participle": "run", "translation": "бежать"},
    {"base": "see", "past": "saw", "participle": "seen", "translation": "видеть"},
    {"base": "take", "past": "took", "participle": "taken", "translation": "брать"},
    {"base": "write", "past": "wrote", "participle": "written", "translation": "писать"},
]

# Prepare wrong options for each verb (not random, generated once)
for v in IRREGULAR_VERBS:
    past = v["past"]
    part = v["participle"]
    base = v["base"]
    v["wrong_pairs"] = [
        f"{past} {base}",
        f"{base} {part}",
        f"{part} {past}",
    ]
