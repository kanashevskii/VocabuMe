PYTHON_QUALITY_FILES=core/env.py core/logging_config.py core/settings.py core/test_settings.py run.py vocab/services.py vocab/views.py vocab/openai_utils.py vocab/reminders.py vocab/management/commands/send_reminders.py tests

.PHONY: install-dev check-dependency-lock format-backend lint-backend typecheck test lint-frontend build-frontend quality

install-dev:
	python -m pip install -r requirements-dev.txt

check-dependency-lock:
	python scripts/check_dependency_lock.py

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

quality: check-dependency-lock lint-backend typecheck lint-frontend build-frontend test
