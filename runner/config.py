import json
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
DEFAULT_CONFIG = {
    "base_url": "http://127.0.0.1:11434/v1",
    "api_key": "ollama",
    "model": "qwen3:8b",
    "stream": False,
}


def load_config() -> dict:
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except Exception:
        return DEFAULT_CONFIG.copy()

    if not isinstance(config, dict):
        return DEFAULT_CONFIG.copy()

    loaded_config = DEFAULT_CONFIG.copy()
    loaded_config.update({
        key: value
        for key, value in config.items()
        if key in loaded_config
    })

    loaded_config["stream"] = parse_bool(loaded_config["stream"])
    return loaded_config


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}

    return bool(value)
