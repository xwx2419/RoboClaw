import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from agent_demo.common.project_env import get_env_int, get_env_str, load_project_dotenv


@dataclass(frozen=True)
class CodexOpenAIConfig:
    model: str
    base_url: str
    api_key: str
    wire_api: str
    reasoning_effort: str
    model_context_window: int
    model_auto_compact_token_limit: int


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DEFAULTS_PATH = PROJECT_ROOT / "src/agent_demo/config/project_openai_defaults.json"


def _normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if not path or path != "/v1":
        return f"{base_url}/v1"
    return base_url


def _load_project_defaults() -> dict:
    if not PROJECT_DEFAULTS_PATH.exists():
        return {}
    return json.loads(PROJECT_DEFAULTS_PATH.read_text(encoding="utf-8"))


def load_codex_openai_config() -> CodexOpenAIConfig:
    load_project_dotenv()
    defaults = _load_project_defaults()
    api_key = get_env_str("OPENAI_API_KEY", "CHAT_API_KEY")
    if not api_key:
        raise ValueError(
            "No OpenAI API key configured. Set OPENAI_API_KEY/CHAT_API_KEY in .env or environment variables."
        )

    model = get_env_str("OPENAI_MODEL", "CHAT_API_MODEL", default=str(defaults.get("model", "gpt-5.4")))
    base_url = _normalize_base_url(
        get_env_str(
            "OPENAI_BASE_URL", "CHAT_API_BASE_URL", default=str(defaults.get("base_url", "https://api.openai.com/v1"))
        )
    )
    wire_api = get_env_str("OPENAI_WIRE_API", "CHAT_API_WIRE_API", default=str(defaults.get("wire_api", "responses")))
    reasoning_effort = (
        get_env_str(
            "OPENAI_REASONING_EFFORT",
            "CHAT_API_REASONING_EFFORT",
            "OPENAI_THINK_LEVEL",
            "CHAT_API_THINK_LEVEL",
            default=str(defaults.get("reasoning_effort", "low")),
        ).lower()
        or "low"
    )
    model_context_window = get_env_int(
        "OPENAI_MODEL_CONTEXT_WINDOW",
        "CHAT_API_MODEL_CONTEXT_WINDOW",
        default=int(defaults.get("model_context_window", 256000) or 256000),
    )
    model_auto_compact_token_limit = get_env_int(
        "OPENAI_MODEL_AUTO_COMPACT_TOKEN_LIMIT",
        "CHAT_API_MODEL_AUTO_COMPACT_TOKEN_LIMIT",
        default=int(defaults.get("model_auto_compact_token_limit", 128000) or 128000),
    )

    return CodexOpenAIConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
        wire_api=wire_api,
        reasoning_effort=reasoning_effort,
        model_context_window=model_context_window,
        model_auto_compact_token_limit=model_auto_compact_token_limit,
    )
