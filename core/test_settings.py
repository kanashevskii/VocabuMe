import os

from .settings import *  # noqa: F403

if os.environ.get("USE_POSTGRES_TESTS") != "1":
    DATABASES["default"] = {  # noqa: F405
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "test.sqlite3",  # noqa: F405
    }
