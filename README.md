# VocabuMe

Телеграм‑бот на Django для изучения английских слов. Помогает добавлять лексику, тренироваться в карточках и проверять прогресс.

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
- Напоминания рассылаются автоматически, без cron

## Требования
- Python 3.10+
- PostgreSQL
- Токен Telegram‑бота
- Ключ OpenAI API

## Установка и запуск
1) Установите зависимости:
```bash
pip install -r requirements.txt
```
2) Создайте `.env`:
```env
SECRET_KEY=your-secret-key
DEBUG=False
DB_NAME=dbname
DB_USER=dbuser
DB_PASSWORD=dbpass
DB_HOST=localhost
DB_PORT=5432
TELEGRAM_TOKEN=your-telegram-token
OPENAI_API_KEY=your-openai-key
ALERT_CHAT_ID=your-telegram-id
```
3) Примените миграции:
```bash
python manage.py migrate --noinput
```
4) Запустите бота и веб‑сервер:
```bash
python run.py
```

## Обслуживание
После обновления логики очистки слов один раз выполните:
```bash
python scripts/clean_existing_words.py
```

## Деплой
В `.github/workflows/deploy.yml` есть пример GitHub Actions: он подтягивает код на сервер, ставит зависимости, накатывает миграции и рестартует `englishbot.service`. Настройте секреты репозитория:
- `SSH_HOST`
- `SSH_USER`
- `SSH_KEY`
- `SSH_PORT`
- `REMOTE_PATH`

## Лицензия
MIT License.
