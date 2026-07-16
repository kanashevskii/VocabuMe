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
                    {
                        "word": "layover",
                        "translation": "пересадка",
                        "synonyms": ["стыковка"],
                    },
                ],
            },
            {
                "id": "hotel_stay",
                "title": "Отель и проживание",
                "description": "Заселение, бронь, проживание и бытовые вопросы в отеле.",
                "difficulty": "Легкий",
                "items": [
                    {"word": "hotel", "translation": "отель"},
                    {
                        "word": "reservation",
                        "translation": "бронь",
                        "synonyms": ["бронирование"],
                    },
                    {
                        "word": "check in",
                        "translation": "зарегистрироваться",
                        "synonyms": ["заселиться"],
                    },
                    {"word": "check out", "translation": "выписаться из отеля"},
                    {
                        "word": "accommodation",
                        "translation": "жильё",
                        "synonyms": ["проживание"],
                    },
                    {"word": "fully booked", "translation": "мест нет"},
                    {
                        "word": "travel insurance",
                        "translation": "страховка для поездки",
                        "synonyms": ["туристическая страховка"],
                    },
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
                    {
                        "word": "sightseeing",
                        "translation": "осмотр достопримечательностей",
                    },
                    {"word": "jet lag", "translation": "смена часовых поясов"},
                    {
                        "word": "travel advisory",
                        "translation": "рекомендации для поездки",
                    },
                    {
                        "word": "off the beaten path",
                        "translation": "в стороне от туристических маршрутов",
                    },
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
                    {
                        "word": "I need to submit my documents.",
                        "translation": "Мне нужно подать документы.",
                    },
                    {"word": "job offer", "translation": "предложение о работе"},
                    {
                        "word": "When can I start working?",
                        "translation": "Когда я могу начать работать?",
                    },
                    {
                        "word": "I need help with my documents.",
                        "translation": "Мне нужна помощь с документами.",
                    },
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
            },
            {
                "id": "payroll_and_tax",
                "title": "Зарплата и налоги",
                "description": "Банковские реквизиты, налоговый номер, выплаты и вопросы по расчету.",
                "difficulty": "Средний",
                "items": [
                    {"word": "payroll", "translation": "расчет зарплаты"},
                    {
                        "word": "tax registration",
                        "translation": "налоговая регистрация",
                    },
                    {"word": "bank details", "translation": "банковские реквизиты"},
                    {"word": "net salary", "translation": "зарплата после налогов"},
                    {"word": "gross salary", "translation": "зарплата до налогов"},
                    {
                        "word": "When is salary paid?",
                        "translation": "Когда выплачивают зарплату?",
                    },
                    {
                        "word": "Do you need my bank details?",
                        "translation": "Вам нужны мои банковские реквизиты?",
                    },
                    {
                        "word": "Can I receive a payslip?",
                        "translation": "Можно получить расчетный лист?",
                    },
                ],
            },
            {
                "id": "hr_follow_up",
                "title": "HR и уточнения",
                "description": "Уточнить статус, недостающие документы, контакты и следующий шаг.",
                "difficulty": "Средний",
                "items": [
                    {
                        "word": "onboarding",
                        "translation": "оформление перед началом работы",
                    },
                    {"word": "missing document", "translation": "недостающий документ"},
                    {"word": "deadline", "translation": "срок"},
                    {"word": "contact person", "translation": "контактное лицо"},
                    {
                        "word": "Can you confirm receipt?",
                        "translation": "Можете подтвердить получение?",
                    },
                    {
                        "word": "What is the next step?",
                        "translation": "Какой следующий шаг?",
                    },
                    {
                        "word": "Who should I contact?",
                        "translation": "К кому мне обратиться?",
                    },
                    {
                        "word": "I can send the document today.",
                        "translation": "Я могу отправить документ сегодня.",
                    },
                ],
            },
        ],
    },
    {
        "id": "georgia_residence_permit_en",
        "title": "English: ВНЖ и разрешение на работу",
        "emoji": "🛂",
        "description": "Фразы для подачи на ВНЖ, разрешение на работу, записи в госучреждение и вопросов по документам.",
        "difficulty": "Средний",
        "track": "relocation",
        "starter_pack": False,
        "levels": [
            {
                "id": "residence_application",
                "title": "ВНЖ и анкета",
                "description": "Запись, анкета, основание для ВНЖ и вопросы на приеме.",
                "difficulty": "Средний",
                "items": [
                    {"word": "residence permit", "translation": "вид на жительство"},
                    {"word": "application form", "translation": "анкета"},
                    {"word": "appointment", "translation": "запись на прием"},
                    {
                        "word": "supporting documents",
                        "translation": "подтверждающие документы",
                    },
                    {"word": "proof of income", "translation": "подтверждение дохода"},
                    {
                        "word": "I want to apply for a residence permit.",
                        "translation": "Я хочу подать на вид на жительство.",
                    },
                    {
                        "word": "What is the processing time?",
                        "translation": "Какой срок рассмотрения?",
                    },
                    {
                        "word": "Can I submit these documents today?",
                        "translation": "Могу я подать эти документы сегодня?",
                    },
                ],
            },
            {
                "id": "work_permit_questions",
                "title": "Разрешение на работу",
                "description": "Рабочее основание, работодатель, номер заявления и уточнения по статусу заявки.",
                "difficulty": "Средний",
                "items": [
                    {"word": "work permit", "translation": "разрешение на работу"},
                    {
                        "word": "employer letter",
                        "translation": "письмо от работодателя",
                    },
                    {"word": "application number", "translation": "номер заявления"},
                    {"word": "approval", "translation": "одобрение"},
                    {"word": "rejection", "translation": "отказ"},
                    {
                        "word": "My employer prepared this document.",
                        "translation": "Мой работодатель подготовил этот документ.",
                    },
                    {
                        "word": "Is anything missing from my application?",
                        "translation": "Чего-то не хватает в моем заявлении?",
                    },
                    {
                        "word": "How can I check the status?",
                        "translation": "Как я могу проверить статус?",
                    },
                ],
            },
            {
                "id": "work_interview_prep",
                "title": "Видео-интервью",
                "description": "Вопросы, которые публично описывают заявители и консультанты: личность, паспорт, деятельность, клиенты и опыт.",
                "difficulty": "Средний",
                "items": [
                    {
                        "word": "What is your full name?",
                        "translation": "Как ваше полное имя?",
                    },
                    {
                        "word": "What is your year of birth?",
                        "translation": "Какой у вас год рождения?",
                    },
                    {
                        "word": "Please show your passport to the camera.",
                        "translation": "Пожалуйста, покажите паспорт в камеру.",
                    },
                    {"word": "What do you do?", "translation": "Чем вы занимаетесь?"},
                    {
                        "word": "How long have you been in business?",
                        "translation": "Как давно вы занимаетесь этим бизнесом?",
                    },
                    {
                        "word": "Can you describe your latest project?",
                        "translation": "Можете описать ваш последний проект?",
                    },
                    {
                        "word": "Are you a contractor or an employee?",
                        "translation": "Вы подрядчик или сотрудник?",
                    },
                    {
                        "word": "Do you work with Georgian clients?",
                        "translation": "Вы работаете с грузинскими клиентами?",
                    },
                    {
                        "word": "Do you work with foreign clients?",
                        "translation": "Вы работаете с иностранными клиентами?",
                    },
                    {
                        "word": "Can you explain your business plan?",
                        "translation": "Можете объяснить ваш бизнес-план?",
                    },
                ],
            },
            {
                "id": "public_service_hall",
                "title": "Дом юстиции и запись",
                "description": "Записаться, уточнить окно, оплату услуги, перевод и копии документов.",
                "difficulty": "Средний",
                "items": [
                    {"word": "Public Service Hall", "translation": "Дом юстиции"},
                    {"word": "service fee", "translation": "плата за услугу"},
                    {
                        "word": "notarized translation",
                        "translation": "нотариальный перевод",
                    },
                    {"word": "document copy", "translation": "копия документа"},
                    {
                        "word": "biometric photo",
                        "translation": "биометрическая фотография",
                    },
                    {
                        "word": "Do I need an appointment?",
                        "translation": "Нужна ли запись?",
                    },
                    {
                        "word": "Where should I pay the fee?",
                        "translation": "Где оплатить сбор?",
                    },
                    {
                        "word": "Do I need a notarized translation?",
                        "translation": "Нужен ли нотариальный перевод?",
                    },
                ],
            },
            {
                "id": "status_and_follow_up",
                "title": "Статус и донос документов",
                "description": "Проверить статус, донести документы, понять срок и получить ответ.",
                "difficulty": "Средний",
                "items": [
                    {"word": "application status", "translation": "статус заявления"},
                    {
                        "word": "additional documents",
                        "translation": "дополнительные документы",
                    },
                    {"word": "decision date", "translation": "дата решения"},
                    {"word": "pickup date", "translation": "дата получения"},
                    {
                        "word": "Can I add documents later?",
                        "translation": "Могу я донести документы позже?",
                    },
                    {
                        "word": "How will I receive the decision?",
                        "translation": "Как я получу решение?",
                    },
                    {
                        "word": "Can I check it online?",
                        "translation": "Можно проверить это онлайн?",
                    },
                    {
                        "word": "Please write down the application number.",
                        "translation": "Пожалуйста, запишите номер заявления.",
                    },
                ],
            },
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
                    {
                        "word": "application review",
                        "translation": "рассмотрение заявки",
                    },
                    {"word": "branch", "translation": "отделение"},
                    {"word": "account number", "translation": "номер счета"},
                    {
                        "word": "I want to open a bank account.",
                        "translation": "Я хочу открыть банковский счет.",
                    },
                    {
                        "word": "What documents do I need?",
                        "translation": "Какие документы мне нужны?",
                    },
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
                    {
                        "word": "currency conversion",
                        "translation": "конвертация валюты",
                    },
                    {"word": "transfer fee", "translation": "комиссия за перевод"},
                    {
                        "word": "How long does it take?",
                        "translation": "Сколько это занимает времени?",
                    },
                ],
            },
        ],
    },
    {
        "id": "georgia_rent_utilities_en",
        "title": "English: аренда и коммуналка",
        "emoji": "🏠",
        "description": "Чтобы обсудить квартиру, договор, депозит, оплату, счета, соседей и ремонт с хозяином или агентом.",
        "difficulty": "Средний",
        "track": "relocation",
        "starter_pack": False,
        "levels": [
            {
                "id": "viewing_and_contract",
                "title": "Просмотр и договор",
                "description": "Осмотр квартиры, условия аренды, депозит и вопросы перед подписанием договора.",
                "difficulty": "Средний",
                "items": [
                    {"word": "rental agreement", "translation": "договор аренды"},
                    {"word": "security deposit", "translation": "залог"},
                    {"word": "monthly rent", "translation": "ежемесячная аренда"},
                    {"word": "landlord", "translation": "арендодатель"},
                    {
                        "word": "real estate agent",
                        "translation": "агент по недвижимости",
                    },
                    {
                        "word": "Can I see the apartment today?",
                        "translation": "Можно посмотреть квартиру сегодня?",
                    },
                    {
                        "word": "Is the deposit refundable?",
                        "translation": "Залог возвращается?",
                    },
                    {
                        "word": "What is included in the rent?",
                        "translation": "Что включено в аренду?",
                    },
                ],
            },
            {
                "id": "utilities_and_repairs",
                "title": "Счета и ремонт",
                "description": "Коммунальные счета, интернет, поломки и сообщения хозяину без неловких догадок.",
                "difficulty": "Средний",
                "items": [
                    {
                        "word": "utility bill",
                        "translation": "счет за коммунальные услуги",
                    },
                    {
                        "word": "electricity bill",
                        "translation": "счет за электричество",
                    },
                    {"word": "water bill", "translation": "счет за воду"},
                    {"word": "internet provider", "translation": "интернет-провайдер"},
                    {"word": "maintenance", "translation": "обслуживание"},
                    {
                        "word": "The heater is not working.",
                        "translation": "Обогреватель не работает.",
                    },
                    {
                        "word": "There is a leak in the bathroom.",
                        "translation": "В ванной протечка.",
                    },
                    {
                        "word": "Who should I contact for repairs?",
                        "translation": "К кому обращаться по ремонту?",
                    },
                ],
            },
            {
                "id": "payments_and_handover",
                "title": "Оплата и передача",
                "description": "Когда платить, как передать депозит, получить ключи и зафиксировать состояние квартиры.",
                "difficulty": "Средний",
                "items": [
                    {"word": "payment receipt", "translation": "квитанция об оплате"},
                    {"word": "bank transfer", "translation": "банковский перевод"},
                    {"word": "move-in date", "translation": "дата заселения"},
                    {
                        "word": "handover checklist",
                        "translation": "акт приема-передачи",
                    },
                    {"word": "meter reading", "translation": "показания счетчика"},
                    {
                        "word": "Can I pay by bank transfer?",
                        "translation": "Можно оплатить банковским переводом?",
                    },
                    {
                        "word": "When do I get the keys?",
                        "translation": "Когда я получу ключи?",
                    },
                    {
                        "word": "Can we take photos before I move in?",
                        "translation": "Можем сделать фото до моего заселения?",
                    },
                ],
            },
            {
                "id": "rules_and_neighbors",
                "title": "Правила и соседи",
                "description": "Домовые правила, шум, парковка, лифт, мусор и вопросы к соседям или управляющему.",
                "difficulty": "Средний",
                "items": [
                    {"word": "building rules", "translation": "правила дома"},
                    {"word": "quiet hours", "translation": "часы тишины"},
                    {"word": "parking space", "translation": "парковочное место"},
                    {"word": "elevator", "translation": "лифт"},
                    {"word": "garbage collection", "translation": "вывоз мусора"},
                    {"word": "building manager", "translation": "управляющий домом"},
                    {"word": "Are pets allowed?", "translation": "Можно с животными?"},
                    {
                        "word": "Who pays the building fee?",
                        "translation": "Кто платит сбор за дом?",
                    },
                ],
            },
        ],
    },
    {
        "id": "georgia_health_pharmacy_en",
        "title": "English: врач и аптека после переезда",
        "emoji": "💊",
        "description": "Чтобы записаться к врачу, объяснить симптомы, купить лекарство и понять инструкцию.",
        "difficulty": "Средний",
        "track": "relocation",
        "starter_pack": False,
        "levels": [
            {
                "id": "doctor_visit",
                "title": "Запись к врачу",
                "description": "Запись, симптомы, страховка и базовые вопросы на приеме.",
                "difficulty": "Средний",
                "items": [
                    {"word": "appointment", "translation": "запись на прием"},
                    {
                        "word": "health insurance",
                        "translation": "медицинская страховка",
                    },
                    {"word": "symptoms", "translation": "симптомы"},
                    {"word": "prescription", "translation": "рецепт"},
                    {
                        "word": "I need to see a doctor.",
                        "translation": "Мне нужно обратиться к врачу.",
                    },
                    {
                        "word": "Do you have an appointment available today?",
                        "translation": "Есть свободная запись на сегодня?",
                    },
                    {"word": "I have a fever.", "translation": "У меня температура."},
                    {
                        "word": "Does my insurance cover this?",
                        "translation": "Моя страховка это покрывает?",
                    },
                ],
            },
            {
                "id": "pharmacy_and_medicine",
                "title": "Аптека и лекарства",
                "description": "Покупка лекарства, дозировка, противопоказания и уточнение замены.",
                "difficulty": "Легкий",
                "items": [
                    {"word": "pharmacy", "translation": "аптека"},
                    {"word": "painkiller", "translation": "обезболивающее"},
                    {"word": "dosage", "translation": "дозировка"},
                    {"word": "side effects", "translation": "побочные эффекты"},
                    {"word": "allergy", "translation": "аллергия"},
                    {
                        "word": "Do I need a prescription?",
                        "translation": "Нужен ли рецепт?",
                    },
                    {
                        "word": "How often should I take it?",
                        "translation": "Как часто это принимать?",
                    },
                    {
                        "word": "Is there a cheaper alternative?",
                        "translation": "Есть ли более дешевый аналог?",
                    },
                ],
            },
            {
                "id": "clinic_and_insurance",
                "title": "Клиника и страховка",
                "description": "Выбрать клинику, уточнить покрытие, стоимость приема и документы.",
                "difficulty": "Средний",
                "items": [
                    {"word": "clinic", "translation": "клиника"},
                    {"word": "general practitioner", "translation": "терапевт"},
                    {"word": "insurance policy", "translation": "страховой полис"},
                    {"word": "medical record", "translation": "медицинская карта"},
                    {
                        "word": "consultation fee",
                        "translation": "стоимость консультации",
                    },
                    {
                        "word": "Is this clinic covered by insurance?",
                        "translation": "Эта клиника покрывается страховкой?",
                    },
                    {
                        "word": "Do I need to pay upfront?",
                        "translation": "Нужно оплатить заранее?",
                    },
                    {
                        "word": "Can I get a receipt for insurance?",
                        "translation": "Можно получить чек для страховки?",
                    },
                ],
            },
            {
                "id": "urgent_care",
                "title": "Срочно и диагностика",
                "description": "Срочная помощь, анализы, направление и понятные вопросы при ухудшении состояния.",
                "difficulty": "Средний",
                "items": [
                    {"word": "urgent care", "translation": "срочная помощь"},
                    {"word": "blood test", "translation": "анализ крови"},
                    {"word": "referral", "translation": "направление"},
                    {"word": "X-ray", "translation": "рентген"},
                    {
                        "word": "I have severe pain.",
                        "translation": "У меня сильная боль.",
                    },
                    {
                        "word": "Is it an emergency?",
                        "translation": "Это экстренный случай?",
                    },
                    {
                        "word": "When will the results be ready?",
                        "translation": "Когда будут готовы результаты?",
                    },
                    {
                        "word": "Should I come back for a follow-up?",
                        "translation": "Мне нужно прийти на повторный прием?",
                    },
                ],
            },
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
                    {
                        "word": "address registration",
                        "translation": "регистрация адреса",
                    },
                    {"word": "top up balance", "translation": "пополнить баланс"},
                    {"word": "Wi-Fi password", "translation": "пароль от Wi‑Fi"},
                    {"word": "delivery address", "translation": "адрес доставки"},
                    {
                        "word": "I live at this address.",
                        "translation": "Я живу по этому адресу.",
                    },
                    {
                        "word": "Please write it down.",
                        "translation": "Пожалуйста, напишите это.",
                    },
                ],
            },
            {
                "id": "housing_and_city",
                "title": "Жилье и город",
                "description": "Аренда, транспорт, магазин и повседневные просьбы, без которых тяжело в новом городе.",
                "difficulty": "Легкий",
                "items": [
                    {"word": "rental agreement", "translation": "договор аренды"},
                    {
                        "word": "utility bill",
                        "translation": "счет за коммунальные услуги",
                    },
                    {"word": "bus stop", "translation": "автобусная остановка"},
                    {"word": "monthly pass", "translation": "месячный проездной"},
                    {"word": "landlord", "translation": "арендодатель"},
                    {"word": "pharmacy", "translation": "аптека"},
                    {"word": "grocery store", "translation": "продуктовый магазин"},
                    {
                        "word": "I am new in Georgia.",
                        "translation": "Я недавно в Грузии.",
                    },
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
                    {
                        "word": "Can I pay by card?",
                        "translation": "Можно оплатить картой?",
                    },
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
                    {
                        "word": "I am here to pick up a parcel.",
                        "translation": "Я пришел забрать посылку.",
                    },
                    {"word": "Where is my package?", "translation": "Где моя посылка?"},
                ],
            },
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
                    {
                        "word": "საბუთები უნდა ჩავაბარო",
                        "translation": "Мне нужно подать документы.",
                    },
                    {"word": "დახმარება მჭირდება", "translation": "мне нужна помощь"},
                    {"word": "დამსაქმებელი", "translation": "работодатель"},
                    {
                        "word": "როდის შემიძლია მუშაობის დაწყება?",
                        "translation": "Когда я могу начать работать?",
                    },
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
                    {
                        "word": "სად უნდა მოვაწერო ხელი?",
                        "translation": "Где я должен подписать?",
                    },
                    {
                        "word": "ეს დოკუმენტი მჭირდება",
                        "translation": "Мне нужен этот документ.",
                    },
                ],
            },
            {
                "id": "payroll_and_tax",
                "title": "Зарплата и налоги",
                "description": "Реквизиты, налоговый номер, выплаты и вопросы по расчету.",
                "difficulty": "Средний",
                "items": [
                    {"word": "ხელფასის დარიცხვა", "translation": "расчет зарплаты"},
                    {
                        "word": "საგადასახადო რეგისტრაცია",
                        "translation": "налоговая регистрация",
                    },
                    {
                        "word": "საბანკო რეკვიზიტები",
                        "translation": "банковские реквизиты",
                    },
                    {
                        "word": "ხელზე ასაღები ხელფასი",
                        "translation": "зарплата после налогов",
                    },
                    {"word": "დარიცხული ხელფასი", "translation": "зарплата до налогов"},
                    {
                        "word": "ხელფასი როდის ირიცხება?",
                        "translation": "Когда выплачивают зарплату?",
                    },
                    {
                        "word": "ჩემი საბანკო რეკვიზიტები გჭირდებათ?",
                        "translation": "Вам нужны мои банковские реквизиты?",
                    },
                    {
                        "word": "ხელფასის ფურცლის მიღება შემიძლია?",
                        "translation": "Можно получить расчетный лист?",
                    },
                ],
            },
            {
                "id": "hr_follow_up",
                "title": "HR и уточнения",
                "description": "Статус оформления, недостающие документы, контакты и следующий шаг.",
                "difficulty": "Средний",
                "items": [
                    {
                        "word": "თანამშრომლის გაფორმება",
                        "translation": "оформление сотрудника",
                    },
                    {"word": "დაკლებული საბუთი", "translation": "недостающий документ"},
                    {"word": "ვადა", "translation": "срок"},
                    {"word": "საკონტაქტო პირი", "translation": "контактное лицо"},
                    {
                        "word": "მიღებას დამიდასტურებთ?",
                        "translation": "Можете подтвердить получение?",
                    },
                    {
                        "word": "შემდეგი ნაბიჯი რა არის?",
                        "translation": "Какой следующий шаг?",
                    },
                    {
                        "word": "ვის უნდა მივმართო?",
                        "translation": "К кому мне обратиться?",
                    },
                    {
                        "word": "საბუთის გაგზავნა დღეს შემიძლია",
                        "translation": "Я могу отправить документ сегодня.",
                    },
                ],
            },
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
                    {
                        "word": "მინდა ანგარიშის გახსნა",
                        "translation": "Я хочу открыть счет.",
                    },
                    {
                        "word": "რა საბუთებია საჭირო?",
                        "translation": "Какие документы нужны?",
                    },
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
                    {
                        "word": "ბარათი როდის იქნება მზად?",
                        "translation": "Когда карта будет готова?",
                    },
                    {"word": "მობილური ბანკინგი", "translation": "мобильный банк"},
                    {
                        "word": "რამდენ ხანს დასჭირდება?",
                        "translation": "Сколько это займет по времени?",
                    },
                ],
            },
        ],
    },
    {
        "id": "georgia_residence_permit_ka",
        "title": "Грузинский: ВНЖ и разрешения",
        "emoji": "🛂",
        "description": "Фразы для ВНЖ, разрешения на работу, записи в госучреждение и вопросов по документам.",
        "difficulty": "Средний",
        "track": "relocation",
        "starter_pack": False,
        "levels": [
            {
                "id": "residence_application",
                "title": "ВНЖ и анкета",
                "description": "Запись, анкета, основание для ВНЖ и базовые вопросы на приеме.",
                "difficulty": "Средний",
                "items": [
                    {"word": "ბინადრობის ნებართვა", "translation": "вид на жительство"},
                    {"word": "განაცხადის ფორმა", "translation": "анкета"},
                    {"word": "ჩაწერა", "translation": "запись на прием"},
                    {
                        "word": "დამადასტურებელი საბუთები",
                        "translation": "подтверждающие документы",
                    },
                    {
                        "word": "შემოსავლის დადასტურება",
                        "translation": "подтверждение дохода",
                    },
                    {
                        "word": "ბინადრობის ნებართვაზე განაცხადის შეტანა მინდა",
                        "translation": "Я хочу подать на вид на жительство.",
                    },
                    {
                        "word": "განხილვას რამდენი დრო სჭირდება?",
                        "translation": "Сколько занимает рассмотрение?",
                    },
                    {
                        "word": "დღეს შემიძლია საბუთების ჩაბარება?",
                        "translation": "Могу я подать документы сегодня?",
                    },
                ],
            },
            {
                "id": "work_permit_questions",
                "title": "Разрешение на работу",
                "description": "Рабочее основание, работодатель, номер заявления и статус заявки.",
                "difficulty": "Средний",
                "items": [
                    {"word": "სამუშაო ნებართვა", "translation": "разрешение на работу"},
                    {
                        "word": "დამსაქმებლის წერილი",
                        "translation": "письмо от работодателя",
                    },
                    {"word": "განაცხადის ნომერი", "translation": "номер заявления"},
                    {"word": "დამტკიცება", "translation": "одобрение"},
                    {"word": "უარი", "translation": "отказ"},
                    {
                        "word": "ეს საბუთი ჩემმა დამსაქმებელმა მოამზადა",
                        "translation": "Мой работодатель подготовил этот документ.",
                    },
                    {
                        "word": "ჩემს განაცხადში რამე აკლია?",
                        "translation": "Чего-то не хватает в моем заявлении?",
                    },
                    {
                        "word": "სტატუსი როგორ შევამოწმო?",
                        "translation": "Как проверить статус?",
                    },
                ],
            },
            {
                "id": "work_interview_prep",
                "title": "Видео-интервью",
                "description": "Вопросы, которые публично описывают заявители и консультанты: личность, паспорт, деятельность, клиенты и опыт.",
                "difficulty": "Средний",
                "items": [
                    {
                        "word": "თქვენი სრული სახელი რა არის?",
                        "translation": "Как ваше полное имя?",
                    },
                    {
                        "word": "რომელ წელს დაიბადეთ?",
                        "translation": "Какой у вас год рождения?",
                    },
                    {
                        "word": "პასპორტი კამერას აჩვენეთ, გთხოვთ",
                        "translation": "Пожалуйста, покажите паспорт в камеру.",
                    },
                    {"word": "რას საქმიანობთ?", "translation": "Чем вы занимаетесь?"},
                    {
                        "word": "რამდენი ხანია ამ საქმიანობას ეწევით?",
                        "translation": "Как давно вы занимаетесь этой деятельностью?",
                    },
                    {
                        "word": "ბოლო პროექტი აღწერეთ, გთხოვთ",
                        "translation": "Опишите ваш последний проект, пожалуйста.",
                    },
                    {
                        "word": "კონტრაქტორი ხართ თუ თანამშრომელი?",
                        "translation": "Вы подрядчик или сотрудник?",
                    },
                    {
                        "word": "ქართველ კლიენტებთან მუშაობთ?",
                        "translation": "Вы работаете с грузинскими клиентами?",
                    },
                    {
                        "word": "უცხოელ კლიენტებთან მუშაობთ?",
                        "translation": "Вы работаете с иностранными клиентами?",
                    },
                    {
                        "word": "თქვენი ბიზნეს გეგმა ახსენით, გთხოვთ",
                        "translation": "Объясните ваш бизнес-план, пожалуйста.",
                    },
                ],
            },
            {
                "id": "public_service_hall",
                "title": "Дом юстиции и запись",
                "description": "Запись, окно, сбор, перевод и копии документов.",
                "difficulty": "Средний",
                "items": [
                    {"word": "იუსტიციის სახლი", "translation": "Дом юстиции"},
                    {"word": "მომსახურების საფასური", "translation": "плата за услугу"},
                    {
                        "word": "ნოტარიულად დამოწმებული თარგმანი",
                        "translation": "нотариальный перевод",
                    },
                    {"word": "საბუთის ასლი", "translation": "копия документа"},
                    {
                        "word": "ბიომეტრიული ფოტო",
                        "translation": "биометрическая фотография",
                    },
                    {"word": "ჩაწერა საჭიროა?", "translation": "Нужна ли запись?"},
                    {
                        "word": "საფასური სად უნდა გადავიხადო?",
                        "translation": "Где оплатить сбор?",
                    },
                    {
                        "word": "ნოტარიული თარგმანი მჭირდება?",
                        "translation": "Нужен ли нотариальный перевод?",
                    },
                ],
            },
            {
                "id": "status_and_follow_up",
                "title": "Статус и донос документов",
                "description": "Проверить статус, донести документы, понять срок и получить ответ.",
                "difficulty": "Средний",
                "items": [
                    {"word": "განაცხადის სტატუსი", "translation": "статус заявления"},
                    {
                        "word": "დამატებითი საბუთები",
                        "translation": "дополнительные документы",
                    },
                    {"word": "გადაწყვეტილების თარიღი", "translation": "дата решения"},
                    {"word": "მიღების თარიღი", "translation": "дата получения"},
                    {
                        "word": "საბუთების დამატება მოგვიანებით შეიძლება?",
                        "translation": "Могу я донести документы позже?",
                    },
                    {
                        "word": "გადაწყვეტილებას როგორ მივიღებ?",
                        "translation": "Как я получу решение?",
                    },
                    {
                        "word": "ონლაინ შემოწმება შეიძლება?",
                        "translation": "Можно проверить онлайн?",
                    },
                    {
                        "word": "განაცხადის ნომერი ჩამიწერეთ, გთხოვთ",
                        "translation": "Пожалуйста, запишите номер заявления.",
                    },
                ],
            },
        ],
    },
    {
        "id": "georgia_rent_utilities_ka",
        "title": "Грузинский: аренда и коммуналка",
        "emoji": "🏠",
        "description": "Фразы для квартиры, договора, залога, оплаты, счетов, соседей, интернета и ремонта.",
        "difficulty": "Средний",
        "track": "relocation",
        "starter_pack": False,
        "levels": [
            {
                "id": "viewing_and_contract",
                "title": "Просмотр и договор",
                "description": "Осмотр квартиры, условия аренды, депозит и вопросы перед подписанием.",
                "difficulty": "Средний",
                "items": [
                    {
                        "word": "ქირავნობის ხელშეკრულება",
                        "translation": "договор аренды",
                    },
                    {"word": "დეპოზიტი", "translation": "залог"},
                    {"word": "თვიური ქირა", "translation": "ежемесячная аренда"},
                    {"word": "მეპატრონე", "translation": "арендодатель"},
                    {
                        "word": "უძრავი ქონების აგენტი",
                        "translation": "агент по недвижимости",
                    },
                    {
                        "word": "დღეს ბინის ნახვა შეიძლება?",
                        "translation": "Можно посмотреть квартиру сегодня?",
                    },
                    {
                        "word": "დეპოზიტი დაბრუნებადია?",
                        "translation": "Залог возвращается?",
                    },
                    {
                        "word": "ქირაში რა შედის?",
                        "translation": "Что включено в аренду?",
                    },
                ],
            },
            {
                "id": "utilities_and_repairs",
                "title": "Счета и ремонт",
                "description": "Коммунальные счета, интернет, поломки и сообщения хозяину.",
                "difficulty": "Средний",
                "items": [
                    {
                        "word": "კომუნალური გადასახადი",
                        "translation": "коммунальный счет",
                    },
                    {
                        "word": "ელექტროენერგიის გადასახადი",
                        "translation": "счет за электричество",
                    },
                    {"word": "წყლის გადასახადი", "translation": "счет за воду"},
                    {
                        "word": "ინტერნეტ პროვაიდერი",
                        "translation": "интернет-провайдер",
                    },
                    {"word": "შეკეთება", "translation": "ремонт"},
                    {
                        "word": "გამათბობელი არ მუშაობს",
                        "translation": "Обогреватель не работает.",
                    },
                    {
                        "word": "აბაზანაში წყალი ჟონავს",
                        "translation": "В ванной протечка.",
                    },
                    {
                        "word": "შეკეთებისთვის ვის მივმართო?",
                        "translation": "К кому обращаться по ремонту?",
                    },
                ],
            },
            {
                "id": "payments_and_handover",
                "title": "Оплата и передача",
                "description": "Оплата, депозит, ключи, счетчики и фиксация состояния квартиры.",
                "difficulty": "Средний",
                "items": [
                    {"word": "გადახდის ქვითარი", "translation": "квитанция об оплате"},
                    {"word": "საბანკო გადარიცხვა", "translation": "банковский перевод"},
                    {"word": "შესვლის თარიღი", "translation": "дата заселения"},
                    {
                        "word": "ბინის გადაბარების აქტი",
                        "translation": "акт приема-передачи",
                    },
                    {"word": "მრიცხველის ჩვენება", "translation": "показания счетчика"},
                    {
                        "word": "საბანკო გადარიცხვით გადახდა შეიძლება?",
                        "translation": "Можно оплатить банковским переводом?",
                    },
                    {
                        "word": "გასაღებს როდის მივიღებ?",
                        "translation": "Когда я получу ключи?",
                    },
                    {
                        "word": "შესვლამდე ფოტოები გადავიღოთ?",
                        "translation": "Можем сделать фото до заселения?",
                    },
                ],
            },
            {
                "id": "rules_and_neighbors",
                "title": "Правила и соседи",
                "description": "Правила дома, шум, парковка, лифт, мусор и управляющий.",
                "difficulty": "Средний",
                "items": [
                    {"word": "სახლის წესები", "translation": "правила дома"},
                    {"word": "სიჩუმის საათები", "translation": "часы тишины"},
                    {"word": "პარკინგის ადგილი", "translation": "парковочное место"},
                    {"word": "ლიფტი", "translation": "лифт"},
                    {"word": "ნაგვის გატანა", "translation": "вывоз мусора"},
                    {"word": "კორპუსის მენეჯერი", "translation": "управляющий домом"},
                    {
                        "word": "ცხოველები დაშვებულია?",
                        "translation": "Можно с животными?",
                    },
                    {
                        "word": "კორპუსის გადასახადს ვინ იხდის?",
                        "translation": "Кто платит сбор за дом?",
                    },
                ],
            },
        ],
    },
    {
        "id": "georgia_health_pharmacy_ka",
        "title": "Грузинский: врач и аптека",
        "emoji": "💊",
        "description": "Фразы для записи к врачу, симптомов, страховки, лекарства и дозировки.",
        "difficulty": "Средний",
        "track": "relocation",
        "starter_pack": False,
        "levels": [
            {
                "id": "doctor_visit",
                "title": "Запись к врачу",
                "description": "Запись, симптомы, страховка и базовые вопросы на приеме.",
                "difficulty": "Средний",
                "items": [
                    {"word": "ექიმთან ჩაწერა", "translation": "запись к врачу"},
                    {
                        "word": "ჯანმრთელობის დაზღვევა",
                        "translation": "медицинская страховка",
                    },
                    {"word": "სიმპტომები", "translation": "симптомы"},
                    {"word": "რეცეპტი", "translation": "рецепт"},
                    {
                        "word": "ექიმთან მისვლა მჭირდება",
                        "translation": "Мне нужно обратиться к врачу.",
                    },
                    {
                        "word": "დღეს თავისუფალი დრო გაქვთ?",
                        "translation": "Есть свободная запись на сегодня?",
                    },
                    {"word": "სიცხე მაქვს", "translation": "У меня температура."},
                    {
                        "word": "ჩემი დაზღვევა ამას ფარავს?",
                        "translation": "Моя страховка это покрывает?",
                    },
                ],
            },
            {
                "id": "pharmacy_and_medicine",
                "title": "Аптека и лекарства",
                "description": "Покупка лекарства, дозировка, противопоказания и уточнение замены.",
                "difficulty": "Легкий",
                "items": [
                    {"word": "აფთიაქი", "translation": "аптека"},
                    {"word": "ტკივილგამაყუჩებელი", "translation": "обезболивающее"},
                    {"word": "დოზა", "translation": "дозировка"},
                    {"word": "გვერდითი მოვლენები", "translation": "побочные эффекты"},
                    {"word": "ალერგია", "translation": "аллергия"},
                    {"word": "რეცეპტი საჭიროა?", "translation": "Нужен ли рецепт?"},
                    {
                        "word": "რამდენად ხშირად უნდა მივიღო?",
                        "translation": "Как часто принимать?",
                    },
                    {
                        "word": "უფრო იაფი ანალოგი გაქვთ?",
                        "translation": "Есть более дешевый аналог?",
                    },
                ],
            },
            {
                "id": "clinic_and_insurance",
                "title": "Клиника и страховка",
                "description": "Клиника, покрытие, стоимость приема и документы для страховки.",
                "difficulty": "Средний",
                "items": [
                    {"word": "კლინიკა", "translation": "клиника"},
                    {"word": "ოჯახის ექიმი", "translation": "терапевт"},
                    {"word": "სადაზღვევო პოლისი", "translation": "страховой полис"},
                    {"word": "სამედიცინო ჩანაწერი", "translation": "медицинская карта"},
                    {
                        "word": "კონსულტაციის საფასური",
                        "translation": "стоимость консультации",
                    },
                    {
                        "word": "ამ კლინიკას დაზღვევა ფარავს?",
                        "translation": "Эта клиника покрывается страховкой?",
                    },
                    {
                        "word": "წინასწარ უნდა გადავიხადო?",
                        "translation": "Нужно оплатить заранее?",
                    },
                    {
                        "word": "დაზღვევისთვის ქვითარს მომცემთ?",
                        "translation": "Можно получить чек для страховки?",
                    },
                ],
            },
            {
                "id": "urgent_care",
                "title": "Срочно и диагностика",
                "description": "Срочная помощь, анализы, направление и вопросы при ухудшении состояния.",
                "difficulty": "Средний",
                "items": [
                    {"word": "გადაუდებელი დახმარება", "translation": "срочная помощь"},
                    {"word": "სისხლის ანალიზი", "translation": "анализ крови"},
                    {"word": "მიმართვა", "translation": "направление"},
                    {"word": "რენტგენი", "translation": "рентген"},
                    {
                        "word": "ძლიერი ტკივილი მაქვს",
                        "translation": "У меня сильная боль.",
                    },
                    {
                        "word": "ეს გადაუდებელია?",
                        "translation": "Это экстренный случай?",
                    },
                    {
                        "word": "პასუხები როდის იქნება მზად?",
                        "translation": "Когда будут готовы результаты?",
                    },
                    {
                        "word": "განმეორებით უნდა მოვიდე?",
                        "translation": "Мне нужно прийти на повторный прием?",
                    },
                ],
            },
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
                    {
                        "word": "ეს მისამართი დამიწერეთ, გთხოვთ",
                        "translation": "Пожалуйста, напишите мне этот адрес.",
                    },
                ],
            },
            {
                "id": "housing_and_city",
                "title": "Жилье и город",
                "description": "Аренда, транспорт, магазин и базовые городские ситуации без языкового ступора.",
                "difficulty": "Легкий",
                "items": [
                    {"word": "ქირა", "translation": "аренда"},
                    {
                        "word": "ავტობუსის გაჩერება",
                        "translation": "автобусная остановка",
                    },
                    {"word": "თვიური აბონემენტი", "translation": "месячный проездной"},
                    {"word": "მეპატრონე", "translation": "арендодатель"},
                    {"word": "აფთიაქი", "translation": "аптека"},
                    {"word": "მაღაზია", "translation": "магазин"},
                    {
                        "word": "ახლახან ჩამოვედი საქართველოში",
                        "translation": "Я недавно приехал в Грузию.",
                    },
                    {
                        "word": "სად შეიძლება ამის ყიდვა?",
                        "translation": "Где можно это купить?",
                    },
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
                    {
                        "word": "ბარათით გადახდა შეიძლება?",
                        "translation": "Можно оплатить картой?",
                    },
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
                    {
                        "word": "საბაჟო დეკლარაცია",
                        "translation": "таможенная декларация",
                    },
                    {
                        "word": "ამანათის წასაღებად მოვედი",
                        "translation": "Я пришел забрать посылку.",
                    },
                    {
                        "word": "სად არის ჩემი ამანათი?",
                        "translation": "Где моя посылка?",
                    },
                ],
            },
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
