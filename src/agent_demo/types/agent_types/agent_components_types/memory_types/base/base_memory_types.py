from pydantic import BaseModel
from rich.table import Table
import logging
from .base_memory_info import MemoryBaseinfo
from .base_context_types import (
    AssistantMessageType,
    ToolMessageType,
    UserMessageType,
    RobotImgMessageType,
    SystemMessageType,
    SystemDynamicMessageType,
)

logger = logging.getLogger(__name__)


class BaseContent(BaseModel):
    base_info: MemoryBaseinfo
    index: int
    content: (
        SystemMessageType
        | SystemDynamicMessageType
        | UserMessageType
        | AssistantMessageType
        | ToolMessageType
        | RobotImgMessageType
    )

    def rich_table(self) -> Table:
        table = Table(show_lines=False, box=None, expand=False)
        table.add_column(self.base_info.emoji, style="cyan", no_wrap=True)
        table.add_column(str(self.index), style="magenta")
        table.add_row("content.type", self.content.__class__.__name__)
        table.add_row("content", str(self.content))
        return table

    def __str__(self) -> str:
        return (
            f"{self.content}"
            f"[{self.index}][{self.base_info.emoji} {self.base_info.name_en}]------[{self.base_info.created_at}] \n"
        )

    def to_openai_format(self, hide_image: bool = False) -> dict:
        if isinstance(self.content, UserMessageType):
            return self.content.to_openai_format()
        elif isinstance(self.content, AssistantMessageType):
            return self.content.to_openai_format()
        elif isinstance(self.content, ToolMessageType):
            return self.content.to_openai_format()
        elif isinstance(self.content, RobotImgMessageType):
            return self.content.to_openai_format(hide_image)
        elif isinstance(self.content, SystemMessageType):
            return self.content.to_openai_format()
        elif isinstance(self.content, SystemDynamicMessageType):
            return self.content.to_openai_format()
        else:
            logger.error(f"Unsupported content type: {self.content.__class__.__name__}")
            return {}


class BaseArea(BaseModel):
    base_info: MemoryBaseinfo

    def __str__(self) -> str:
        return f"[{self.base_info.emoji} {self.base_info.name_cn}] {self.base_info.describe_cn}"


class BaseBlock(BaseModel):
    base_info: MemoryBaseinfo

    def __str__(self) -> str:
        return f"[{self.base_info.emoji} {self.base_info.name_cn}] {self.base_info.describe_cn}"


class BaseTaskNode(BaseModel):
    base_info: MemoryBaseinfo

    def __str__(self) -> str:
        return f"[{self.base_info.emoji} {self.base_info.name_cn}] {self.base_info.describe_cn}"
