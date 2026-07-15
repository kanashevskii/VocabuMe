PYTHON_QUALITY_FILES=core/env.py core/logging_config.py core/settings.py core/test_settings.py run.py vocab/services.py vocab/views.py vocab/openai_utils.py vocab/reminders.py vocab/management/commands/send_reminders.py tests
BLACK_QUALITY_FILES=core/env.py core/logging_config.py core/test_settings.py run.py vocab/openapi.py vocab/tasks.py vocab/tts.py vocab/utils.py vocab/models.py vocab/jobs.py vocab/openai_limits.py vocab/openai_budget.py vocab/api/common.py vocab/api/docs.py vocab/api/errors.py vocab/api/media.py vocab/api/images.py vocab/api/irregular.py vocab/api/learning.py vocab/application/streaks.py vocab/application/irregular_questions.py vocab/media_storage.py vocab/ratelimit.py vocab/telegram_auth.py vocab/selectors/progress.py vocab/integrations/images.py vocab/integrations/telegram/messaging.py vocab/integrations/telegram/settings_ui.py tests/test_openapi.py tests/test_telegram_auth.py tests/test_progress_read_model.py tests/test_streaks.py tests/test_utils.py tests/test_openai_limits.py tests/test_openai_budget.py tests/test_irregular_questions.py tests/test_avatar_integrity.py tests/test_api_media_facade.py tests/test_jobs.py tests/test_image_generation_recovery.py tests/test_read_models_no_mutation.py
PYTHON ?= python

.PHONY: install-dev check-dependency-lock format-backend format-check lint-backend typecheck test security-audit lint-frontend build-frontend quality

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

check-dependency-lock:
	$(PYTHON) scripts/check_dependency_lock.py

format-backend:
	$(PYTHON) -m black $(PYTHON_QUALITY_FILES)

format-check:
	$(PYTHON) -m black --check $(BLACK_QUALITY_FILES)

lint-backend:
	$(PYTHON) -m ruff check core vocab

typecheck:
	$(PYTHON) -m mypy -p vocab
	$(PYTHON) -m mypy core/env.py core/settings.py core/test_settings.py run.py

test:
	$(PYTHON) -m pytest --cov=vocab --cov=core --cov-fail-under=40 -q

security-audit:
	$(PYTHON) -m pip_audit -r requirements-prod.lock

lint-frontend:
	cd frontend && npm run lint

build-frontend:
	cd frontend && npm run build

quality: check-dependency-lock format-check lint-backend typecheck test security-audit lint-frontend build-frontend
