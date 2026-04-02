from pydantic import BaseModel, Field
from typing import Optional
from typing_extensions import Literal
from .base_data_types import (
    TextParam,
    ImageParam,
    ToolCallParam,
    RefusalParam,
)


class ChatRole:
    user = "user"
    assistant = "assistant"
    system = "system"
    tool = "tool"


class CustomRole:  # 区分多个模块用的，暂时用不到
    user = "User"
    llm = "Cloud-LLM"
    robot = "Robot"


# 基础内容类型
# 机器人返回的工具调用结果
class ToolMessageType(BaseModel):
    content: TextParam
    role: str = Field(default_factory=lambda: str(ChatRole.tool))
    name: str = Field(default_factory=lambda: str(CustomRole.robot))
    tool_call_id: str

    def __str__(self):  # 方便调试打印
        return f"--> [{self.role}][{self.name}]<{ToolMessageType.__name__}>:\n{self.content.text}\n"

    def to_openai_format(self) -> dict:
        return {"role": self.role, "content": self.content.to_openai_format(), "tool_call_id": self.tool_call_id}  # type: ignore

    @classmethod
    def text_param(cls, text: str, tool_call_id: str) -> "ToolMessageType":
        return cls(content=TextParam(text=text), tool_call_id=tool_call_id)


# 云端模型返回的消息类型
class AssistantMessageType(BaseModel):
    content: TextParam | None = Field(default=None)
    refusal: RefusalParam | None = Field(default=None)
    role: str = Field(default_factory=lambda: str(ChatRole.assistant))
    name: str = Field(default_factory=lambda: str(CustomRole.llm))
    tool_calls: list[ToolCallParam] = Field(default_factory=list)

    def __str__(self):  # 方便调试打印
        if self.content:
            return f"--> [{self.role}][{self.name}]<{AssistantMessageType.__name__}>:\n{self.content.text}\n[{len(self.tool_calls)}]"
        elif self.refusal:
            return f"--> [{self.role}][{self.name}]<{AssistantMessageType.__name__}>:\n{self.refusal.refusal}\n[{len(self.tool_calls)}]"
        elif len(self.tool_calls) > 0:
            return f"--> [{self.role}][{self.name}]<{AssistantMessageType.__name__}>:\ntools_call\n[{len(self.tool_calls)}]"
        else:
            return f"[{self.role}][{self.name}]<{AssistantMessageType.__name__}>:Error\n"

    def to_openai_format(self) -> dict:
        if self.content:
            content = self.content.to_openai_format()
        elif self.refusal:
            content = self.refusal.to_openai_format()
        else:
            content = None

        msg: dict = {"role": self.role, "content": content, "name": self.name}  # type: ignore

        if self.tool_calls:
            msg["tool_calls"] = [tool_call.to_openai_format() for tool_call in self.tool_calls]

        return msg

    @classmethod
    def text_param(cls, text: str, tool_calls: Optional[list[ToolCallParam]] = None) -> "AssistantMessageType":
        return cls(content=TextParam(text=text), tool_calls=tool_calls or [])

    @classmethod
    def refusal_param(cls, refusal: str, tool_calls: Optional[list[ToolCallParam]] = None) -> "AssistantMessageType":
        return cls(refusal=RefusalParam(refusal=refusal), tool_calls=tool_calls or [])


# 用户输入的消息类型
class UserMessageType(BaseModel):
    content: TextParam
    role: str = Field(default_factory=lambda: str(ChatRole.user))
    name: str = Field(default_factory=lambda: str(CustomRole.user))

    def __str__(self):  # 方便调试打印
        return f"--> [{self.role}][{self.name}]<{UserMessageType.__name__}>:\n{self.content.text}\n"

    def to_openai_format(self) -> dict:
        return {"role": self.role, "content": self.content.to_openai_format(), "name": self.name}

    @classmethod
    def text_param(cls, text: str) -> "UserMessageType":
        return cls(content=TextParam(text=text))


# 机器人返回的视觉输入消息类型
class RobotImgMessageType(BaseModel):
    frame_id: int
    content: TextParam
    content_img: ImageParam
    role: str = Field(default_factory=lambda: str(ChatRole.user))
    name: str = Field(default_factory=lambda: str(CustomRole.robot))

    def __str__(self):  # 方便调试打印
        return (
            f"--> [{self.role}][{self.name}]<{RobotImgMessageType.__name__}>:\n{self.content.text}\n[暂不支持显示图片]"
        )

    def to_openai_format(self, hide_image: bool = False) -> dict:
        if hide_image:
            res: dict[str, object] = {
                "role": self.role,
                "content": f"因为记忆大小有限，第{self.frame_id}帧已经被隐藏",
                "name": self.name,
            }
        else:
            res = {
                "role": self.role,
                "content": [self.content.to_openai_format_text(), self.content_img.to_openai_format(hide_image)],
                "name": self.name,
            }
        return res  # type: ignore

    @classmethod
    def image_param(
        cls,
        img_frame_id: int,
        text: str,
        img_type: Literal["jpeg", "png", "jpg"],
        base64_str: str,
        detail: Literal["auto", "low", "high"] = "auto",
    ) -> "RobotImgMessageType":
        return cls(
            frame_id=img_frame_id,
            content=TextParam(text=text),
            content_img=ImageParam.from_base64(img_type, base64_str, detail),
        )


# 系统提示词类型
class SystemMessageType(BaseModel):
    content: TextParam
    role: str = Field(default_factory=lambda: str(ChatRole.system))
    name: str = Field(default_factory=lambda: str(CustomRole.robot))

    def __str__(self):  # 方便调试打印
        return f"--> [{self.role}][{self.name}]<{SystemMessageType.__name__}>:\n{self.content.text}\n"

    def to_openai_format(self) -> dict:
        return {"role": self.role, "content": self.content.to_openai_format(), "name": self.name}  # type: ignore

    @classmethod
    def text_param(cls, text: str) -> "SystemMessageType":
        return cls(content=TextParam(text=text))


# 系统提示词-动态更新类型
class SystemDynamicMessageType(BaseModel):
    content: TextParam
    role: str = Field(default_factory=lambda: str(ChatRole.user))
    name: str = Field(default_factory=lambda: str(CustomRole.robot))

    def __str__(self):  # 方便调试打印
        return f"--> [{self.role}][{self.name}]<{SystemDynamicMessageType.__name__}>:\n{self.content.text}\n"

    def to_openai_format(self) -> dict:
        return {"role": self.role, "content": self.content.to_openai_format(), "name": self.name}  # type: ignore

    @classmethod
    def text_param(cls, text: str) -> "SystemDynamicMessageType":
        return cls(content=TextParam(text=text))
