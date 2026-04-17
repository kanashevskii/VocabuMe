TRAVEL_PACKS = [
    {
        "id": "travel",
        "title": "В поездке",
        "emoji": "✈️",
        "description": "Запасной набор для коротких поездок: аэропорт, отель и ориентирование в городе.",
        "difficulty": "Средний",
        "track": "general",
        "starter_pack": False,
        "levels": [
            {
                "id": "airport_boarding",
                "title": "Аэропорт и посадка",
                "description": "Все, что нужно для рейса, регистрации и посадки.",
                "difficulty": "Легкий",
                "items": [
                    {"word": "passport", "translation": "паспорт"},
                    {"word": "ticket", "translation": "билет"},
                    {"word": "luggage", "translation": "багаж"},
                    {"word": "boarding pass", "translation": "посадочный талон"},
                    {"word": "airport", "translation": "аэропорт"},
                    {"word": "flight", "translation": "рейс"},
                    {"word": "delay", "translation": "задержка"},
                    {"word": "departure", "translation": "отправление"},
                    {"word": "arrival", "translation": "прибытие"},
                    {"word": "layover", "translation": "пересадка", "synonyms": ["стыковка"]},
                ],
            },
            {
                "id": "hotel_stay",
                "title": "Отель и проживание",
                "description": "Заселение, бронь, проживание и бытовые вопросы в отеле.",
                "difficulty": "Легкий",
                "items": [
                    {"word": "hotel", "translation": "отель"},
                    {"word": "reservation", "translation": "бронь", "synonyms": ["бронирование"]},
                    {"word": "check in", "translation": "зарегистрироваться", "synonyms": ["заселиться"]},
                    {"word": "check out", "translation": "выписаться из отеля"},
                    {
                        "word": "accommodation",
                        "translation": "жильё",
                        "synonyms": ["проживание"],
                    },
                    {"word": "fully booked", "translation": "мест нет"},
                    {"word": "travel insurance", "translation": "страховка для поездки", "synonyms": ["туристическая страховка"]},
                    {"word": "local cuisine", "translation": "местная кухня"},
                ],
            },
            {
                "id": "city_transport",
                "title": "Маршрут и город",
                "description": "Навигация, транспорт и перемещения по новому городу.",
                "difficulty": "Средний",
                "items": [
                    {"word": "map", "translation": "карта"},
                    {"word": "tourist", "translation": "турист"},
                    {"word": "itinerary", "translation": "маршрут поездки"},
                    {"word": "customs", "translation": "таможня"},
                    {"word": "destination", "translation": "пункт назначения"},
                    {
                        "word": "public transport",
                        "translation": "общественный транспорт",
                        "synonyms": ["транспорт"],
                    },
                    {
                        "word": "currency exchange",
                        "translation": "обмен валюты",
                        "synonyms": ["обмен денег"],
                    },
                    {"word": "sightseeing", "translation": "осмотр достопримечательностей"},
                    {"word": "jet lag", "translation": "смена часовых поясов"},
                    {"word": "travel advisory", "translation": "рекомендации для поездки"},
                    {"word": "off the beaten path", "translation": "в стороне от туристических маршрутов"},
                ],
            },
        ],
    }
]

ENGLISH_RELOCATION_PACKS = [
    {
        "id": "georgia_work_permit_en",
        "title": "English: работа и документы после переезда",
        "emoji": "📄",
        "description": "Чтобы заполнить анкету, договориться о старте и понять рабочие условия на английском.",
        "difficulty": "Средний",
        "track": "relocation",
        "starter_pack": False,
        "levels": [
            {
                "id": "job_documents",
                "title": "Документы и анкета",
                "description": "Подать документы, заполнить форму и не потеряться на первом шаге с работодателем.",
                "difficulty": "Легкий",
                "items": [
                    {"word": "work permit", "translation": "разрешение на работу"},
                    {"word": "passport copy", "translation": "копия паспорта"},
                    {"word": "tax ID", "translation": "налоговый номер"},
                    {"word": "application form", "translation": "форма заявления"},
                    {"word": "I need to submit my documents.", "translation": "Мне нужно подать документы."},
                    {"word": "job offer", "translation": "предложение о работе"},
                    {"word": "When can I start working?", "translation": "Когда я могу начать работать?"},
                    {"word": "I need help with my documents.", "translation": "Мне нужна помощь с документами."},
                ],
            },
            {
                "id": "contract_and_terms",
                "title": "Договор и условия",
                "description": "Обсуждение зарплаты, графика, подписи и даты выхода на работу.",
                "difficulty": "Средний",
                "items": [
                    {"word": "employment contract", "translation": "трудовой договор"},
                    {"word": "signature", "translation": "подпись"},
                    {"word": "start date", "translation": "дата начала работы"},
                    {"word": "salary", "translation": "зарплата"},
                    {"word": "working hours", "translation": "рабочие часы"},
                    {"word": "trial period", "translation": "испытательный срок"},
                    {"word": "position", "translation": "должность"},
                    {"word": "HR manager", "translation": "HR-менеджер"},
                ],
            }
        ],
    },
    {
        "id": "georgia_bank_en",
        "title": "English: банк и счет после переезда",
        "emoji": "🏦",
        "description": "Чтобы открыть счет, получить карту и спокойно решить банковские вопросы на английском.",
        "difficulty": "Средний",
        "track": "relocation",
        "starter_pack": True,
        "levels": [
            {
                "id": "open_account",
                "title": "Открытие счета",
                "description": "Первый визит в банк: открыть счет, понять список документов и задать нужные вопросы.",
                "difficulty": "Легкий",
                "items": [
                    {"word": "bank account", "translation": "банковский счет"},
                    {"word": "debit card", "translation": "дебетовая карта"},
                    {"word": "proof of address", "translation": "подтверждение адреса"},
                    {"word": "application review", "translation": "рассмотрение заявки"},
                    {"word": "branch", "translation": "отделение"},
                    {"word": "account number", "translation": "номер счета"},
                    {"word": "I want to open a bank account.", "translation": "Я хочу открыть банковский счет."},
                    {"word": "What documents do I need?", "translation": "Какие документы мне нужны?"},
                ],
            },
            {
                "id": "statements_and_transfers",
                "title": "Выписка и переводы",
                "description": "Запрос выписки, реквизитов, комиссий и сроков перевода без лишнего стресса.",
                "difficulty": "Средний",
                "items": [
                    {"word": "statement", "translation": "выписка"},
                    {"word": "cash deposit", "translation": "внесение наличных"},
                    {"word": "withdraw money", "translation": "снять деньги"},
                    {"word": "mobile banking", "translation": "мобильный банк"},
                    {"word": "currency conversion", "translation": "конвертация валюты"},
                    {"word": "transfer fee", "translation": "комиссия за перевод"},
                    {"word": "How long does it take?", "translation": "Сколько это занимает времени?"},
                ],
            }
        ],
    },
    {
        "id": "georgia_first_week_en",
        "title": "English: первые дни после переезда",
        "emoji": "🧭",
        "description": "Связь, адрес, жилье, покупки и первые бытовые вопросы, которые всплывают сразу после переезда.",
        "difficulty": "Легкий",
        "track": "relocation",
        "starter_pack": True,
        "levels": [
            {
                "id": "connection_and_address",
                "title": "Связь и адрес",
                "description": "SIM, доставка, адрес и первые бытовые действия, когда еще все непривычно.",
                "difficulty": "Легкий",
                "items": [
                    {"word": "SIM card", "translation": "сим-карта"},
                    {"word": "address registration", "translation": "регистрация адреса"},
                    {"word": "top up balance", "translation": "пополнить баланс"},
                    {"word": "Wi-Fi password", "translation": "пароль от Wi‑Fi"},
                    {"word": "delivery address", "translation": "адрес доставки"},
                    {"word": "I live at this address.", "translation": "Я живу по этому адресу."},
                    {"word": "Please write it down.", "translation": "Пожалуйста, напишите это."},
                ],
            },
            {
                "id": "housing_and_city",
                "title": "Жилье и город",
                "description": "Аренда, транспорт, магазин и повседневные просьбы, без которых тяжело в новом городе.",
                "difficulty": "Легкий",
                "items": [
                    {"word": "rental agreement", "translation": "договор аренды"},
                    {"word": "utility bill", "translation": "счет за коммунальные услуги"},
                    {"word": "bus stop", "translation": "автобусная остановка"},
                    {"word": "monthly pass", "translation": "месячный проездной"},
                    {"word": "landlord", "translation": "арендодатель"},
                    {"word": "pharmacy", "translation": "аптека"},
                    {"word": "grocery store", "translation": "продуктовый магазин"},
                    {"word": "I am new in Georgia.", "translation": "Я недавно в Грузии."},
                ],
            },
            {
                "id": "market_and_shop",
                "title": "Рынок и магазин",
                "description": "Покупки, цены, вес и короткие вопросы продавцу в реальной очереди, а не в учебнике.",
                "difficulty": "Легкий",
                "items": [
                    {"word": "market", "translation": "рынок"},
                    {"word": "store", "translation": "магазин"},
                    {"word": "price", "translation": "цена"},
                    {"word": "discount", "translation": "скидка"},
                    {"word": "receipt", "translation": "чек"},
                    {"word": "bag", "translation": "пакет"},
                    {"word": "fresh", "translation": "свежий"},
                    {"word": "How much is this?", "translation": "Сколько это стоит?"},
                    {"word": "Can I pay by card?", "translation": "Можно оплатить картой?"},
                ],
            },
            {
                "id": "post_and_parcels",
                "title": "Почта и посылки",
                "description": "Получение, отправка и поиск посылки, когда нужно быстро разобраться с доставкой.",
                "difficulty": "Средний",
                "items": [
                    {"word": "post office", "translation": "почта"},
                    {"word": "parcel", "translation": "посылка"},
                    {"word": "delivery", "translation": "доставка"},
                    {"word": "tracking number", "translation": "номер отслеживания"},
                    {"word": "pickup point", "translation": "пункт выдачи"},
                    {"word": "customs form", "translation": "таможенная декларация"},
                    {"word": "I am here to pick up a parcel.", "translation": "Я пришел забрать посылку."},
                    {"word": "Where is my package?", "translation": "Где моя посылка?"},
                ],
            }
        ],
    },
]

GEORGIAN_TEST_PACKS = [
    {
        "id": "georgian_starter",
        "title": "Грузинский: быстрый старт",
        "emoji": "🇬🇪",
        "description": "Короткий стартовый набор, чтобы не зависнуть в первом бытовом диалоге на грузинском.",
        "difficulty": "Легкий",
        "track": "relocation",
        "starter_pack": True,
        "levels": [
            {
                "id": "starter",
                "title": "Первые 10",
                "description": "Приветствие, вежливость и простые бытовые фразы.",
                "difficulty": "Легкий",
                "items": [
                    {
                        "word": "გამარჯობა",
                        "translation": "привет",
                        "synonyms": ["здравствуйте"],
                    },
                    {
                        "word": "ნახვამდის",
                        "translation": "пока",
                        "synonyms": ["до свидания"],
                    },
                    {
                        "word": "როგორ ხარ?",
                        "translation": "как дела?",
                        "synonyms": ["как ты?"],
                    },
                    {
                        "word": "მადლობა",
                        "translation": "спасибо",
                        "synonyms": ["благодарю"],
                    },
                    {
                        "word": "გთხოვ",
                        "translation": "пожалуйста",
                        "synonyms": ["прошу"],
                    },
                    {"word": "დიახ", "translation": "да"},
                    {"word": "არა", "translation": "нет"},
                    {
                        "word": "ბოდიში",
                        "translation": "извините",
                        "synonyms": ["простите"],
                    },
                    {
                        "word": "არ მესმის",
                        "translation": "я не понимаю",
                        "synonyms": ["не понимаю"],
                    },
                    {
                        "word": "რა ღირს?",
                        "translation": "сколько стоит?",
                        "synonyms": ["сколько это стоит?"],
                    },
                ],
            }
        ],
    }
]

GEORGIAN_RELOCATION_PACKS = [
    {
        "id": "georgia_work_permit_ka",
        "title": "Грузинский: работа и документы",
        "emoji": "📄",
        "description": "Фразы для документов, работодателя и рабочих ситуаций, когда нужно быстро объясниться на месте.",
        "difficulty": "Средний",
        "track": "relocation",
        "starter_pack": False,
        "levels": [
            {
                "id": "job_documents",
                "title": "Документы и анкета",
                "description": "Подача документов и базовые вопросы по оформлению, чтобы не потеряться в процессе.",
                "difficulty": "Легкий",
                "items": [
                    {"word": "სამუშაო", "translation": "работа"},
                    {"word": "საბუთები", "translation": "документы"},
                    {"word": "პასპორტი", "translation": "паспорт"},
                    {"word": "განაცხადი", "translation": "заявление"},
                    {"word": "საბუთები უნდა ჩავაბარო", "translation": "Мне нужно подать документы."},
                    {"word": "დახმარება მჭირდება", "translation": "мне нужна помощь"},
                    {"word": "დამსაქმებელი", "translation": "работодатель"},
                    {"word": "როდის შემიძლია მუშაობის დაწყება?", "translation": "Когда я могу начать работать?"},
                ],
            },
            {
                "id": "contract_and_terms",
                "title": "Договор и условия",
                "description": "Подпись, зарплата, график и детали выхода на работу без паники и догадок.",
                "difficulty": "Средний",
                "items": [
                    {"word": "კონტრაქტი", "translation": "договор"},
                    {"word": "ხელმოწერა", "translation": "подпись"},
                    {"word": "ხელფასი", "translation": "зарплата"},
                    {"word": "თანამდებობა", "translation": "должность"},
                    {"word": "სამუშაო საათები", "translation": "рабочие часы"},
                    {"word": "საცდელი ვადა", "translation": "испытательный срок"},
                    {"word": "სად უნდა მოვაწერო ხელი?", "translation": "Где я должен подписать?"},
                    {"word": "ეს დოკუმენტი მჭირდება", "translation": "Мне нужен этот документ."},
                ],
            }
        ],
    },
    {
        "id": "georgia_bank_ka",
        "title": "Грузинский: банк и счет",
        "emoji": "🏦",
        "description": "Самые нужные слова и фразы, чтобы открыть счет, получить карту и понять сотрудника банка.",
        "difficulty": "Средний",
        "track": "relocation",
        "starter_pack": False,
        "levels": [
            {
                "id": "open_account",
                "title": "Открытие счета",
                "description": "Первый визит в банк, открытие счета и документы, которые у тебя попросят.",
                "difficulty": "Легкий",
                "items": [
                    {"word": "ბანკი", "translation": "банк"},
                    {"word": "ანგარიში", "translation": "счет"},
                    {"word": "ბარათი", "translation": "карта"},
                    {"word": "ფილიალი", "translation": "отделение"},
                    {"word": "ანგარიშის ნომერი", "translation": "номер счета"},
                    {"word": "მინდა ანგარიშის გახსნა", "translation": "Я хочу открыть счет."},
                    {"word": "რა საბუთებია საჭირო?", "translation": "Какие документы нужны?"},
                ],
            },
            {
                "id": "statements_and_transfers",
                "title": "Выписка и переводы",
                "description": "Выписка, переводы, комиссии и вопросы по карте, когда нужно решить задачу быстро.",
                "difficulty": "Средний",
                "items": [
                    {"word": "ნაღდი ფული", "translation": "наличные"},
                    {"word": "ამონაწერი", "translation": "выписка"},
                    {"word": "პინ კოდი", "translation": "пин-код"},
                    {"word": "გადარიცხვა", "translation": "перевод"},
                    {"word": "კომისია", "translation": "комиссия"},
                    {"word": "ვალუტის გადაცვლა", "translation": "обмен валюты"},
                    {"word": "ბარათი როდის იქნება მზად?", "translation": "Когда карта будет готова?"},
                    {"word": "მობილური ბანკინგი", "translation": "мобильный банк"},
                    {"word": "რამდენ ხანს დასჭირდება?", "translation": "Сколько это займет по времени?"},
                ],
            }
        ],
    },
    {
        "id": "georgia_first_week_ka",
        "title": "Грузинский: первые дни после переезда",
        "emoji": "🧭",
        "description": "Связь, адрес, жилье, покупки и первые бытовые вопросы, которые всплывают сразу после переезда.",
        "difficulty": "Легкий",
        "track": "relocation",
        "starter_pack": True,
        "levels": [
            {
                "id": "connection_and_address",
                "title": "Связь и адрес",
                "description": "SIM, адрес, доставка и первые бытовые действия, когда нужно быстро освоиться.",
                "difficulty": "Легкий",
                "items": [
                    {"word": "სიმ ბარათი", "translation": "сим-карта"},
                    {"word": "მისამართი", "translation": "адрес"},
                    {"word": "ბალანსის შევსება", "translation": "пополнение баланса"},
                    {"word": "Wi‑Fi-ის პაროლი", "translation": "пароль от Wi‑Fi"},
                    {"word": "მიტანის მისამართი", "translation": "адрес доставки"},
                    {"word": "აქ ვცხოვრობ", "translation": "Я живу здесь."},
                    {"word": "ეს მისამართი დამიწერეთ, გთხოვთ", "translation": "Пожалуйста, напишите мне этот адрес."},
                ],
            },
            {
                "id": "housing_and_city",
                "title": "Жилье и город",
                "description": "Аренда, транспорт, магазин и базовые городские ситуации без языкового ступора.",
                "difficulty": "Легкий",
                "items": [
                    {"word": "ქირა", "translation": "аренда"},
                    {"word": "ავტობუსის გაჩერება", "translation": "автобусная остановка"},
                    {"word": "თვიური აბონემენტი", "translation": "месячный проездной"},
                    {"word": "მეპატრონე", "translation": "арендодатель"},
                    {"word": "აფთიაქი", "translation": "аптека"},
                    {"word": "მაღაზია", "translation": "магазин"},
                    {"word": "ახლახან ჩამოვედი საქართველოში", "translation": "Я недавно приехал в Грузию."},
                    {"word": "სად შეიძლება ამის ყიდვა?", "translation": "Где можно это купить?"},
                ],
            },
            {
                "id": "market_and_shop",
                "title": "Рынок и магазин",
                "description": "Цены, покупки, чек и короткий диалог с продавцом в обычной бытовой ситуации.",
                "difficulty": "Легкий",
                "items": [
                    {"word": "ბაზარი", "translation": "рынок"},
                    {"word": "ფასი", "translation": "цена"},
                    {"word": "ფასდაკლება", "translation": "скидка"},
                    {"word": "ჩეკი", "translation": "чек"},
                    {"word": "პარკი", "translation": "пакет"},
                    {"word": "ახალი", "translation": "свежий"},
                    {"word": "ეს რა ღირს?", "translation": "Сколько это стоит?"},
                    {"word": "ბარათით გადახდა შეიძლება?", "translation": "Можно оплатить картой?"},
                ],
            },
            {
                "id": "post_and_parcels",
                "title": "Почта и посылки",
                "description": "Получение, отправка и поиск посылки, когда нужно разобраться с почтой без переводчика.",
                "difficulty": "Средний",
                "items": [
                    {"word": "ფოსტა", "translation": "почта"},
                    {"word": "ამანათი", "translation": "посылка"},
                    {"word": "მიტანა", "translation": "доставка"},
                    {"word": "თრექინგ ნომერი", "translation": "номер отслеживания"},
                    {"word": "გაცემის პუნქტი", "translation": "пункт выдачи"},
                    {"word": "საბაჟო დეკლარაცია", "translation": "таможенная декларация"},
                    {"word": "ამანათის წასაღებად მოვედი", "translation": "Я пришел забрать посылку."},
                    {"word": "სად არის ჩემი ამანათი?", "translation": "Где моя посылка?"},
                ],
            }
        ],
    },
]

COURSE_PACKS = {
    "en": [*ENGLISH_RELOCATION_PACKS, *TRAVEL_PACKS],
    "ka": [*GEORGIAN_TEST_PACKS, *GEORGIAN_RELOCATION_PACKS],
}


def get_pack_definitions(course_code: str | None = None) -> list[dict]:
    if course_code:
        return COURSE_PACKS.get(course_code, [])

    all_packs: list[dict] = []
    for packs in COURSE_PACKS.values():
        all_packs.extend(packs)
    return all_packs


def get_pack(pack_id: str) -> dict | None:
    for pack in get_pack_definitions():
        if pack["id"] == pack_id:
            return pack
    return None


def get_pack_level(pack_id: str, level_id: str) -> dict | None:
    pack = get_pack(pack_id)
    if not pack:
        return None
    for level in pack["levels"]:
        if level["id"] == level_id:
            return level
    return None
