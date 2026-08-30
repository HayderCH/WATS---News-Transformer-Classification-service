import os
from pathlib import Path

os.environ["API_KEY"] = "test-key"
os.environ["API_KEY_HEADER"] = "x-api-key"

test_db_path = Path("test_feedback.db")
if test_db_path.exists():
    test_db_path.unlink()
os.environ.setdefault("DB_URL", f"sqlite:///{test_db_path.as_posix()}")

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()
