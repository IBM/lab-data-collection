import json
from pathlib import Path
import os

APP_DIR = Path(os.getenv("APPDATA")) / "LDC"
APP_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = APP_DIR / "config.json"


def config_exists():
    return CONFIG_FILE.exists()


def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def save_config(data):
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=4)
