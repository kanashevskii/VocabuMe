FROM node:22.14.0-bookworm-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.10.16-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app
RUN python -m venv /opt/venv \
    && groupadd --gid 10001 vocabume \
    && useradd --uid 10001 --gid vocabume --create-home --shell /usr/sbin/nologin vocabume

COPY requirements-prod.lock ./
RUN pip install --upgrade pip==25.0.1 \
    && pip install --no-cache-dir -r requirements-prod.lock

COPY --chown=vocabume:vocabume . ./
COPY --from=frontend-build --chown=vocabume:vocabume /app/frontend/dist ./frontend/dist
RUN SECRET_KEY=build-only-not-a-production-secret-000000000000000000000 \
    DB_NAME=build DB_USER=build DB_PASSWORD=build DB_HOST=build DB_PORT=5432 \
    python manage.py collectstatic --noinput

USER vocabume
EXPOSE 8000
CMD ["gunicorn", "core.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3", "--threads", "2", "--timeout", "90", "--access-logfile", "-", "--error-logfile", "-"]
