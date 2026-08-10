import json
from pathlib import Path
import os

APP_DIR = Path(os.getenv("APPDATA")) / "LDC"
APP_DIR.mkdir(parents=True, exist_ok=True)

USER_FILE = APP_DIR / "user.json"


def user_exists():
    return USER_FILE.exists()


def load_user():
    with open(USER_FILE, "r") as f:
        return json.load(f)


def save_user(data):
    with open(USER_FILE, "w") as f:
        json.dump(data, f, indent=4)
