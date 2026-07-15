PYTHON_QUALITY_FILES=core/env.py core/logging_config.py core/settings.py core/test_settings.py run.py vocab/services.py vocab/views.py vocab/openai_utils.py vocab/reminders.py vocab/management/commands/send_reminders.py tests
PYTHON ?= python

.PHONY: install-dev check-dependency-lock format-backend format-check lint-backend typecheck test security-audit lint-frontend build-frontend quality

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

check-dependency-lock:
	$(PYTHON) scripts/check_dependency_lock.py

format-backend:
	$(PYTHON) -m black $(PYTHON_QUALITY_FILES)

format-check:
	$(PYTHON) -m black --check core vocab tests run.py

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
