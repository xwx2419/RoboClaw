from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DOTENV_LOADED = False


def load_project_dotenv() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    default_env_path = PROJECT_ROOT / ".env"
    local_env_path = PROJECT_ROOT / ".env.local"

    if default_env_path.exists():
        load_dotenv(default_env_path, override=False)
    if local_env_path.exists():
        load_dotenv(local_env_path, override=True)

    _DOTENV_LOADED = True


def get_env_str(*names: str, default: str = "") -> str:
    load_project_dotenv()
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return default


def get_env_int(*names: str, default: int) -> int:
    load_project_dotenv()
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        stripped = value.strip()
        if not stripped:
            continue
        return int(stripped)
    return default


@dataclass(frozen=True)
class FeishuAppConfig:
    app_id: str
    app_secret: str
    verification_token: str
    api_base: str
    event_receiver: str
    sdk_log_level: str

    @property
    def has_credentials(self) -> bool:
        return bool(self.app_id and self.app_secret)


def load_feishu_app_config() -> FeishuAppConfig:
    return FeishuAppConfig(
        app_id=get_env_str("FEISHU_APP_ID"),
        app_secret=get_env_str("FEISHU_APP_SECRET"),
        verification_token=get_env_str("FEISHU_VERIFICATION_TOKEN"),
        api_base=get_env_str("FEISHU_API_BASE", default="https://open.feishu.cn/open-apis"),
        event_receiver=get_env_str("FEISHU_EVENT_RECEIVER", default="long_connection").lower(),
        sdk_log_level=get_env_str("FEISHU_SDK_LOG_LEVEL", default="INFO") or "INFO",
    )
