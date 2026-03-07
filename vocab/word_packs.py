TRAVEL_PACKS = [
    {
        "id": "travel",
        "title": "Travel",
        "emoji": "✈️",
        "description": "Слова и фразы для поездок, аэропорта, отеля и перемещений.",
        "levels": [
            {
                "id": "a1_a2",
                "title": "A1-A2",
                "description": "Базовая лексика для дороги, отеля и города.",
                "items": [
                    {"word": "passport", "translation": "паспорт"},
                    {"word": "ticket", "translation": "билет"},
                    {"word": "luggage", "translation": "багаж"},
                    {"word": "boarding pass", "translation": "посадочный талон"},
                    {"word": "airport", "translation": "аэропорт"},
                    {"word": "flight", "translation": "рейс"},
                    {"word": "hotel", "translation": "отель"},
                    {"word": "reservation", "translation": "бронь"},
                    {"word": "check in", "translation": "зарегистрироваться"},
                    {"word": "check out", "translation": "выписаться из отеля"},
                    {"word": "map", "translation": "карта"},
                    {"word": "tourist", "translation": "турист"},
                ],
            },
            {
                "id": "b1_b2",
                "title": "B1-B2",
                "description": "Более уверенная лексика для поездок и общения в пути.",
                "items": [
                    {"word": "itinerary", "translation": "маршрут поездки"},
                    {"word": "delay", "translation": "задержка"},
                    {"word": "customs", "translation": "таможня"},
                    {"word": "destination", "translation": "пункт назначения"},
                    {"word": "accommodation", "translation": "жильё"},
                    {"word": "currency exchange", "translation": "обмен валюты"},
                    {"word": "departure", "translation": "отправление"},
                    {"word": "arrival", "translation": "прибытие"},
                    {"word": "sightseeing", "translation": "осмотр достопримечательностей"},
                    {"word": "public transport", "translation": "общественный транспорт"},
                    {"word": "travel insurance", "translation": "страховка для поездки"},
                    {"word": "local cuisine", "translation": "местная кухня"},
                ],
            },
            {
                "id": "c1_c2",
                "title": "C1-C2",
                "description": "Продвинутая travel-лексика для сложных ситуаций и нюансов.",
                "items": [
                    {"word": "jet lag", "translation": "смена часовых поясов"},
                    {"word": "layover", "translation": "пересадка"},
                    {"word": "travel advisory", "translation": "рекомендации для поездки"},
                    {"word": "fully booked", "translation": "мест нет"},
                    {"word": "off the beaten path", "translation": "в стороне от туристических маршрутов"},
                    {"word": "compensation claim", "translation": "требование компенсации"},
                    {"word": "breathtaking view", "translation": "захватывающий вид"},
                    {"word": "travel light", "translation": "путешествовать налегке"},
                    {"word": "miss a connection", "translation": "опоздать на стыковку"},
                    {"word": "cultural etiquette", "translation": "культурный этикет"},
                    {"word": "peak season", "translation": "высокий сезон"},
                    {"word": "hidden gem", "translation": "скрытая жемчужина"},
                ],
            },
        ],
    }
]


def get_pack_definitions() -> list[dict]:
    return TRAVEL_PACKS


def get_pack(pack_id: str) -> dict | None:
    for pack in TRAVEL_PACKS:
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
