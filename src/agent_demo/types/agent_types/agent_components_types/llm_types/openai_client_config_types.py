from typing import Optional
from pydantic import BaseModel, Field
from rich.table import Table
from agent_demo.common.root_logger import table_to_str
from agent_demo.common.codex_openai_config import load_codex_openai_config
from agent_demo.common.project_env import get_env_int, get_env_str, load_project_dotenv
import logging

logger = logging.getLogger(__name__)

load_project_dotenv()
DEFAULT_API_KEY = get_env_str("OPENAI_API_KEY", "CHAT_API_KEY")


class OpenaiGroupInfoDict(BaseModel):
    name: str
    code: str


class UserGroupInfo(BaseModel):
    organization: Optional[OpenaiGroupInfoDict] = None
    project: Optional[OpenaiGroupInfoDict] = None

    @classmethod
    def from_env(cls) -> Optional["UserGroupInfo"]:
        organization_code = get_env_str("CHAT_API_ORGANIZATION", "OPENAI_ORGANIZATION")
        project_code = get_env_str("CHAT_API_PROJECT", "OPENAI_PROJECT")
        if not organization_code and not project_code:
            return None

        organization_name = get_env_str(
            "CHAT_API_ORGANIZATION_NAME",
            "OPENAI_ORGANIZATION_NAME",
            default="ConfiguredOrganization",
        )
        project_name = get_env_str(
            "CHAT_API_PROJECT_NAME",
            "OPENAI_PROJECT_NAME",
            default="ConfiguredProject",
        )
        return cls(
            organization=(
                OpenaiGroupInfoDict(name=organization_name, code=organization_code) if organization_code else None
            ),
            project=OpenaiGroupInfoDict(name=project_name, code=project_code) if project_code else None,
        )

    @classmethod
    def genie_software(cls) -> Optional["UserGroupInfo"]:
        return cls.from_env()


class ChatAPIConfig(BaseModel):
    client_name: str  # 当前客户端实例的名字(sn_code)
    user_group_info: Optional[UserGroupInfo]  # 该参数指定组织相关的参数（openai风格）
    # ========== 基础参数 ==========
    model: str  # 模型名称，如 "gpt-3.5-turbo"
    llm_provider: str  # 提供方，如 "openai", "azure", "anthropic"
    base_url: str  # 接口地址，如 "https://api.openai.com/v1"
    api_key: str  # 接口密钥，默认从环境变量 CHAT_API_KEY 获取
    wire_api: str = Field(default="chat.completions")  # 协议类型，如 "chat.completions" / "responses"
    reasoning_effort: str = Field(default="low")  # 推理强度，默认 low
    max_retries: int = Field(default=3)
    timeout: int = Field(default=30)  # 请求超时（秒）
    # ========== 输出控制参数 ==========
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)  # 采样温度
    max_completion_tokens: int = Field(default=1000, ge=1)  # 最大生成 token 数
    compression_threshold: int  # 压缩阈值
    choices_n: int = Field(default=1, ge=1)  # 返回的候选数
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)  #  nucleus sampling 温度(保守、创造)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)  # 惩罚已出现过的词（鼓励模型换话题）
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)  # 惩罚频繁重复词
    stop: Optional[list[str]] = Field(default=None)  # 停止词列表
    # ========== 推理相关 ==========

    @property
    def cache_key(self) -> str:
        return f"{self.model}-{self.temperature}-{self.max_completion_tokens}"

    @classmethod
    def calculate_compression_threshold(cls, max_completion_tokens) -> int:
        threshold = int(max_completion_tokens * 0.8)
        return threshold

    @classmethod
    def openai_compatible_api(
        cls,
        *,
        client_name: str,
        model: str,
        base_url: str,
        api_key: str,
        max_completion_tokens: int = 8192,
        compression_threshold: Optional[int] = None,
        llm_provider: str = "openai",
        user_group_info: Optional[UserGroupInfo] = None,
        wire_api: str = "chat.completions",
        reasoning_effort: str = "low",
    ) -> "ChatAPIConfig":
        return cls(
            client_name=client_name,
            model=model,
            llm_provider=llm_provider,
            base_url=base_url,
            api_key=api_key,
            wire_api=cls._normalize_wire_api(wire_api),
            reasoning_effort=reasoning_effort.strip().lower() or "low",
            user_group_info=user_group_info,
            max_completion_tokens=max_completion_tokens,
            compression_threshold=(
                compression_threshold
                if compression_threshold is not None
                else ChatAPIConfig.calculate_compression_threshold(max_completion_tokens)
            ),
        )

    @staticmethod
    def _normalize_wire_api(wire_api: str) -> str:
        return wire_api.strip().lower().replace("_", ".").replace("-", ".")

    @classmethod
    def openai_gpt_4o(cls, api_key: str = DEFAULT_API_KEY) -> "ChatAPIConfig":
        return cls(
            client_name="test",
            model="gpt-4o",
            llm_provider="openai",
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            user_group_info=UserGroupInfo.from_env(),
            reasoning_effort="low",
            max_completion_tokens=8192,
            compression_threshold=ChatAPIConfig.calculate_compression_threshold(8192),
        )

    @classmethod
    def openai_gpt_5_mini(cls, api_key: str = DEFAULT_API_KEY) -> "ChatAPIConfig":
        return cls(
            client_name="test",
            model="gpt-5-mini",
            llm_provider="openai",
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            user_group_info=UserGroupInfo.from_env(),
            reasoning_effort="low",
            max_completion_tokens=128000,
            compression_threshold=ChatAPIConfig.calculate_compression_threshold(400000),
        )

    @classmethod
    def openai_gpt_4o_mini(cls, api_key: str = DEFAULT_API_KEY) -> "ChatAPIConfig":
        return cls(
            client_name="test",
            model="gpt-4o-mini",
            llm_provider="openai",
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            user_group_info=UserGroupInfo.from_env(),
            reasoning_effort="low",
            max_completion_tokens=8192,
            compression_threshold=ChatAPIConfig.calculate_compression_threshold(8192),
        )

    @classmethod
    def openai_gpt_41(cls, api_key: str = DEFAULT_API_KEY) -> "ChatAPIConfig":
        return cls(
            client_name="test",
            model="gpt-4.1",
            llm_provider="openai",
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            user_group_info=UserGroupInfo.from_env(),
            reasoning_effort="low",
            max_completion_tokens=8192,
            compression_threshold=ChatAPIConfig.calculate_compression_threshold(8192),
        )

    @classmethod
    def project_runtime_default(cls) -> "ChatAPIConfig":
        codex_config = load_codex_openai_config()
        max_completion_tokens = 8192
        return cls.openai_compatible_api(
            client_name="codex",
            model=codex_config.model,
            llm_provider="openai",
            base_url=codex_config.base_url,
            api_key=codex_config.api_key,
            user_group_info=None,
            max_completion_tokens=max_completion_tokens,
            compression_threshold=codex_config.model_auto_compact_token_limit,
            wire_api=codex_config.wire_api,
            reasoning_effort=codex_config.reasoning_effort,
        )

    @classmethod
    def codex_gpt_54(cls) -> "ChatAPIConfig":
        return cls.project_runtime_default()

    @classmethod
    def resolve_runtime_default(cls) -> "ChatAPIConfig":
        api_key = get_env_str("CHAT_API_KEY", "OPENAI_API_KEY")
        base_url = get_env_str("CHAT_API_BASE_URL", "OPENAI_BASE_URL", default="https://api.openai.com/v1")
        model = get_env_str("CHAT_API_MODEL", "OPENAI_MODEL", default="gpt-4o-mini")
        wire_api = get_env_str("CHAT_API_WIRE_API", "OPENAI_WIRE_API", default="chat.completions")
        reasoning_effort = get_env_str(
            "CHAT_API_REASONING_EFFORT",
            "OPENAI_REASONING_EFFORT",
            "CHAT_API_THINK_LEVEL",
            "OPENAI_THINK_LEVEL",
            default="low",
        )
        compression_threshold = get_env_int(
            "CHAT_API_MODEL_AUTO_COMPACT_TOKEN_LIMIT",
            "OPENAI_MODEL_AUTO_COMPACT_TOKEN_LIMIT",
            default=ChatAPIConfig.calculate_compression_threshold(8192),
        )

        if api_key:
            return cls.openai_compatible_api(
                client_name="runtime",
                model=model,
                base_url=base_url,
                api_key=api_key,
                user_group_info=None,
                wire_api=wire_api,
                reasoning_effort=reasoning_effort,
                compression_threshold=compression_threshold,
            )

        try:
            return cls.project_runtime_default()
        except Exception as exc:
            raise ValueError(
                "No API configuration found. "
                "Set OPENAI_API_KEY/OPENAI_BASE_URL/OPENAI_MODEL "
                "(optional: OPENAI_WIRE_API) or "
                "CHAT_API_KEY/CHAT_API_BASE_URL/CHAT_API_MODEL "
                "(optional: CHAT_API_WIRE_API). "
                f"Project-local fallback is unavailable: {exc}"
            ) from exc

    @classmethod
    def deepseek_chat_api(cls, api_key: str = ""):
        return cls(
            client_name="test",
            model="deepseek-chat",
            llm_provider="deepseek",
            base_url="https://api.deepseek.com",
            api_key=api_key,
            user_group_info=UserGroupInfo.from_env(),
            reasoning_effort="low",
            max_completion_tokens=8192,
            compression_threshold=ChatAPIConfig.calculate_compression_threshold(8192),
        )

    @classmethod
    def Qwen3_turbo_api(cls, api_key: str = ""):
        return cls(
            client_name="test",
            model="qwen2.5-7b-instruct-1m",
            llm_provider="qwen",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
            user_group_info=None,
            reasoning_effort="low",
            max_completion_tokens=8192,
            compression_threshold=ChatAPIConfig.calculate_compression_threshold(8192),
        )

    def mask_api_key(self) -> str:
        if len(self.api_key) <= 10:
            return self.api_key[:3] + "******"
        return self.api_key[:3] + "******" + self.api_key[-6:]

    def get_config_table(self) -> str:
        config_dict = {
            "Organization": (
                self.user_group_info.organization.name
                if self.user_group_info and self.user_group_info.organization
                else "None"
            ),
            "Project": (
                self.user_group_info.project.name if self.user_group_info and self.user_group_info.project else "None"
            ),
            "Client Name": self.client_name,
            "Model": self.cache_key,
            "Wire API": self.wire_api,
            "Reasoning Effort": self.reasoning_effort,
            "API Key": self.mask_api_key(),
            "Max Retries": str(self.max_retries),
            "Compression Threshold": str(self.compression_threshold),
            "Timeout": str(self.timeout),
        }
        table = Table(title="Configuration Info", show_lines=True)
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        for key, value in config_dict.items():
            table.add_row(str(key), str(value))
        return table_to_str(table)
