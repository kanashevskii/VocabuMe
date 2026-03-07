# VocabuMe

Телеграм‑бот на Django для изучения английских слов. Теперь включает React-интерфейс, который можно открыть как Telegram Mini App и как обычный сайт с входом через Telegram.

## Что умеет бот
- `/add` — добавить слова или фразы (можно пачкой, поддерживается формат `word - перевод`)
- `/learn` — режим карточек EN→RU (10 штук за сессию, в конце можно повторить те же или взять следующие 10)
- `/practice` — практика перевода с вариантами ответа  
  - «Классический» EN→RU  
  - «Обратный» RU→EN
- `/listening` — аудирование по добавленным словам
- `/irregular` — тренировки неправильных глаголов (глагол засчитывается после 5 верных)
- `/mywords` — список своих слов
- `/progress` — статистика и достижения
- `/settings` — напоминания и параметры тренировок
- React SPA для Mini App и сайта с общим прогрессом
- Напоминания рассылаются автоматически, без cron

## Требования
- Python 3.10+
- PostgreSQL
- Node.js 20+
- Токен Telegram‑бота
- Ключ OpenAI API

## Установка и запуск
1) Установите зависимости:
```bash
pip install -r requirements.txt
```
```bash
cd frontend && npm install
```
2) Создайте `.env`:
Скопируйте `.env.example` → `.env` и заполните значения.
3) Примените миграции:
```bash
python manage.py migrate --noinput
```
4) Соберите фронтенд:
```bash
cd frontend && npm run build
```
5) Запустите бота и веб‑сервер:
```bash
python run.py
```

## Обслуживание
После обновления логики очистки слов один раз выполните:
```bash
python scripts/clean_existing_words.py
```

## Деплой
В production нужно:
- собрать React: `cd frontend && npm ci && npm run build`
- выставить `DEBUG=False`
- задать `WEBAPP_URL`, `TELEGRAM_BOT_USERNAME`, `CSRF_TRUSTED_ORIGINS`
- проксировать `vocabume.k1prod.com` на `127.0.0.1:8000`
- раздавать `/static/` из `frontend/dist/`
- выпустить SSL для `vocabume.k1prod.com`

Для обычного веб-входа через Telegram Login Widget нужно дополнительно указать домен бота через BotFather: `/setdomain` → `vocabume.k1prod.com`.

## Лицензия
MIT License.
