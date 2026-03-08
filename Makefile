PYTHON_QUALITY_FILES=core/env.py core/logging_config.py core/settings.py core/test_settings.py run.py vocab/services.py vocab/views.py vocab/openai_utils.py vocab/reminders.py vocab/management/commands/send_reminders.py tests

.PHONY: format-backend lint-backend typecheck test lint-frontend build-frontend quality

format-backend:
	python -m black $(PYTHON_QUALITY_FILES)

lint-backend:
	python -m flake8 $(PYTHON_QUALITY_FILES)

typecheck:
	python -m mypy core/env.py core/settings.py core/test_settings.py run.py

test:
	python -m pytest -q

lint-frontend:
	cd frontend && npm run lint

build-frontend:
	cd frontend && npm run build

quality: lint-backend typecheck lint-frontend build-frontend test
