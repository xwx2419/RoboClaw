from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from collections import deque
from ...agent_components_types.llm_types.openai_client_config_types import ChatAPIConfig
from ...agent_components_types.memory_types.base.base_memory_info import CompressPolicy
from ....interaction_types.interaction_package import InteractionPackage
from agent_demo.machine_layer.base_dataloader import BaseDataloader
from typing import Optional


class BaseAgentCard(BaseModel):
    # base
    silence: bool
    config: ChatAPIConfig
    service_config_path: str
    skill_paths: list[str] = Field(default_factory=list)
    agent_name: str = Field(default="")
    agent_id: str = Field(default="")
    # memory
    agent_memory_prompt: dict[str, str]  # 初始化当前agent的prompt
    img_threshold: int = Field(default=6)  # 超过这个阈值的图片会被隐藏
    compress_policy: str = Field(default=CompressPolicy.DISCARD_OLDEST)  # 压缩策略
    root_index: int = Field(default=0)  # 记忆索引计数器
    all_token_usage: int = Field(default=0)  # 总token使用量
    prompt_token_usage: int = Field(default=0)  # prompt token使用量
    compress_cnt: int = Field(default=0)  # 压缩次数
    # dataloader
    robot_dataloader: Optional[BaseDataloader] = None
    # display用queue
    display_deque: deque[InteractionPackage] = Field(default=deque(maxlen=50))
    # 允许存储py对象
    model_config = ConfigDict(arbitrary_types_allowed=True)
