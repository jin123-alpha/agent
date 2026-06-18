import json
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
DEFAULT_CONFIG = {
    "base_url": "http://127.0.0.1:11434/v1",
    "api_key": "ollama",
    "model": "qwen3:8b",
    "stream": False,
    # GitHubSearchAgent DAG 参数
    "github_api_key": "",
    "github_max_results": 100,
    "github_per_page": 25,
    "dense_retrieval_k": 100,
    "llm_rerank_top_n": 50,
    "retrieval_alpha": 0.7,
    "min_stars": 50,
    "rerank_threshold": 5.5,
}


def load_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path).expanduser().resolve() if config_path else CONFIG_PATH
    try:
        with path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except Exception as exc:
        if config_path is not None:
            raise ValueError(f"无法读取配置文件 {path}: {exc}") from exc
        return DEFAULT_CONFIG.copy()

    if not isinstance(config, dict):
        if config_path is not None:
            raise ValueError(f"配置文件必须包含 JSON 对象: {path}")
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
