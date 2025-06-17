# VocabuMe

VocabuMe is a Telegram bot built with Django that helps users learn English vocabulary. It provides commands to add words, start training sessions and receive learning reminders.

## Features
- **/add** – add new words or phrases
- **/learn** – start training (EN→RU)
  - includes Russian translation of examples hidden with Telegram spoiler markup
- **/learnreverse** – reverse training (RU→EN)
- **/listening** – listening practice
- **/irregular** – practice irregular verbs (each verb is learned after 5 correct answers)
- **/mywords** – view your vocabulary list
- **/progress** – show statistics and achievements
- Daily reminders run automatically without cron

## Requirements
- Python 3.10+
- PostgreSQL database
- Telegram bot token
- OpenAI API key

## Installation
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file with the following variables:
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
3. Apply database migrations:
   ```bash
   python manage.py migrate --noinput
   ```
4. Start the bot and web server:
   ```bash
   python run.py
   ```

## Maintenance
If you updated the bot to sanitize words, run the following script once to clean
existing entries:

```bash
python scripts/clean_existing_words.py
```

## Deployment
A sample GitHub Actions workflow is provided in `.github/workflows/deploy.yml`. It pulls the latest code on the server, installs dependencies, runs migrations and restarts the `englishbot.service` systemd unit. Configure SSH credentials and the deployment path via repository secrets:
- `SSH_HOST`
- `SSH_USER`
- `SSH_KEY`
- `SSH_PORT`
- `REMOTE_PATH`

## License
This project is released under the MIT License.
