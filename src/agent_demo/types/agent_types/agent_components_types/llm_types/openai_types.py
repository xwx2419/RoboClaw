from pydantic import BaseModel, Field
from typing import Optional
from ..memory_types.base.base_context_types import ToolCallParam, AssistantMessageType
from rich.table import Table


class OpenAIChoice(BaseModel):
    finish_reason: str = Field(default_factory=str)
    index: int
    has_tool_call: bool
    message: AssistantMessageType

    @property
    def tool_calls(self) -> list[ToolCallParam]:
        return self.message.tool_calls

    def __str__(self) -> str:
        return (
            f"OpenAIChoice(\n"
            f"  index={self.index},\n"
            f"  finish_reason='{self.finish_reason}',\n"
            f"  message={self.message}\n"
            f")"
        )

    def rich_table(self) -> Table:
        table = Table(title="   📜   ", show_lines=False, box=None, expand=False)
        table.add_column(str(self.index), style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_row("finish_reason", str(self.finish_reason))
        table.add_row("role", str(self.message.role))
        table.add_row("tool_calls", str(self.message.tool_calls))
        table.add_row("content", str(self.message.content.to_openai_format() if self.message.content else "None"))
        table.add_row("refusal", str(self.message.refusal.to_openai_format() if self.message.refusal else "None"))
        return table


class OpenAIResponseMsg(BaseModel):
    need_compress: bool = Field(default=False)
    # response main_info
    id: str = Field(default_factory=str)
    model: str = Field(default_factory=str)
    created: int = Field(default_factory=int)
    object: str = Field(default_factory=str)
    system_fingerprint: str = Field(default_factory=str)
    # response usage
    prompt_tokens: int = Field(default_factory=int)
    completion_tokens: int = Field(default_factory=int)
    total_tokens: int = Field(default_factory=int)
    cache_hit_rate: Optional[float] = Field(default=None)
    # response choice
    choices: list[OpenAIChoice] = Field(default_factory=list)

    @property
    def first_choice(self) -> OpenAIChoice:
        return self.choices[0]

    @property
    def has_tool_call(self) -> bool:
        return self.first_choice.has_tool_call

    def __str__(self) -> str:
        choices_str = "\n".join([str(choice) for choice in self.choices]) if self.choices else "None"
        return (
            f"OpenAIResponseMsg(\n"
            f"  id='{self.id}',\n"
            f"  model='{self.model}',\n"
            f"  created={self.created},\n"
            f"  object='{self.object}',\n"
            f"  system_fingerprint='{self.system_fingerprint}',\n"
            f"  prompt_tokens={self.prompt_tokens},\n"
            f"  completion_tokens={self.completion_tokens},\n"
            f"  total_tokens={self.total_tokens},\n"
            f"  cache_hit_rate={self.cache_hit_rate},\n"
            f"  choices=[\n{choices_str}\n  ]\n"
            f")"
        )

    def rich_table(self) -> list[Table]:
        table_list: list[Table] = []
        table = Table(title="  🫡  ", show_lines=False, box=None, expand=False)
        table.add_column("Main_Info", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_row("id", str(self.id))
        table.add_row("model", str(self.model))
        table.add_row("created", str(self.created))
        table.add_row("object", str(self.object))
        table.add_row("system_fingerprint", str(self.system_fingerprint))
        table.add_row("prompt_tokens", str(self.prompt_tokens))
        table.add_row("completion_tokens", str(self.completion_tokens))
        table.add_row("total_tokens", str(self.total_tokens))
        table.add_row("cache_hit_rate", f"{self.cache_hit_rate}%" if self.cache_hit_rate is not None else "N/A")
        table_list.append(table)
        for choice in self.choices:
            table_list.append(choice.rich_table())
        return table_list


class OpenAISendMsg(BaseModel):
    contexts: list = Field(default_factory=list)  # 本次对话上下文消息
    tools_list: list = Field(default_factory=list)  # 工具列表

    def __str__(self) -> str:
        contexts_len = f"{len(self.contexts)} items" if self.contexts else "None"
        tools_list_len = f"{len(self.tools_list)} tools" if self.tools_list else "None"
        return f"OpenAISendMsg(contexts={contexts_len}|tools_list={tools_list_len})"
