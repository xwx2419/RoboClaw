from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel, Field, model_validator
from rich.table import Table
import logging

logger = logging.getLogger(__name__)

ORMCP_TOOLS_SPLICE = "___"
SERVER_REGISTRY_TEMPLATE_SUB = "- [{service_name}|{description}|{is_activation}|({tools_list})]\n"


class FunctionDef(BaseModel):
    name: str = Field(default_factory=str)
    description: str = Field(default_factory=str)
    parameters: dict[str, object] = Field(default_factory=dict)
    strict: bool = Field(default=False)

    def to_openai_format(self, service_name: str) -> dict:
        return {
            "name": service_name + ORMCP_TOOLS_SPLICE + self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": self.strict,
        }


class ORMCPTool(BaseModel):
    service_name: str
    func_definition: FunctionDef
    openai_format: dict = Field(default_factory=dict)

    @property
    def tool_name(self) -> str:
        return self.func_definition.name

    @model_validator(mode="after")
    def init_openai_format(self) -> "ORMCPTool":
        self.openai_format = ChatCompletionToolParam(
            function=self.func_definition.to_openai_format(self.service_name), type="function"  # type: ignore
        )
        return self


class ServiceRegister(BaseModel):
    service_name: str = Field(default_factory=str)
    description: str = Field(default_factory=str)
    is_activation: bool = Field(default=False)
    is_agent_service: bool = Field(default=False)
    tools_list: list[ORMCPTool] = Field(default_factory=list)

    def get_tools_list_str(self) -> str:
        tools_list_str: str = ""
        if self.is_activation:
            for tool in self.tools_list:
                tools_list_str += tool.tool_name + ","
        return tools_list_str

    def to_service_registry_block_prompt(self) -> str:
        return SERVER_REGISTRY_TEMPLATE_SUB.format(
            service_name=self.service_name,
            description=self.description,
            is_activation=self.is_activation,
            tools_list=self.get_tools_list_str(),
        )

    def __str__(self) -> str:
        return f"N:{self.service_name}|A:{self.is_activation}|L:{len(self.tools_list)}|D:{self.description}"

    def rich_table(self) -> Table:
        tool_list_str: str = ""
        for tool in self.tools_list:
            tool_list_str += tool.tool_name + "|"
        tool_list_str += str(len(self.tools_list))
        table = Table(title=self.service_name, show_lines=False, box=None, expand=False)
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_row("service_name", self.service_name)
        table.add_row("description", self.description)
        table.add_row("is_activation", str(self.is_activation))
        table.add_row("is_agent_service", str(self.is_agent_service))
        table.add_row("tools_list_info", tool_list_str)
        return table
