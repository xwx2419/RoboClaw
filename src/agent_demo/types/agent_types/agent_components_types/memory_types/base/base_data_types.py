from pydantic import BaseModel, Field
from typing_extensions import Literal
from typing import ClassVar


# 常见数据参数：文本、文件、音频、视频...
class FileSubParam(BaseModel):
    file_data: str  # base64 encoded
    file_id: str
    filename: str


class ImageURLSubParam(BaseModel):
    url: str  # base64 encoded
    detail: Literal["auto", "low", "high"] = Field(default="auto")

    def __str__(self):
        return {
            "url": "......",
            "detail": self.detail,
        }

    @classmethod
    def from_base64(
        cls, img_type: Literal["jpeg", "png", "jpg"], base64_str: str, detail: Literal["auto", "low", "high"] = "auto"
    ) -> "ImageURLSubParam":
        return cls(url=f"data:image/{img_type};base64,{base64_str}", detail=detail)

    def to_openai_format(self, hide_image: bool = False) -> dict:
        if hide_image:
            return {
                "url": "本图片已经被设置隐藏",
                "detail": self.detail,
            }
        else:
            return {
                "url": self.url,
                "detail": self.detail,
            }


class InputAudioSubParam(BaseModel):
    data: str  # base64 encoded
    format: Literal["wav", "mp3"]


class FunctionSubParam(BaseModel):
    arguments: str
    name: str

    def __str__(self) -> str:
        return f"FunctionSubParam(\n" f"  name={self.name},\n" f"  arguments={self.arguments}\n" f")"

    def to_openai_format(self) -> dict:
        return {
            "name": self.name,
            "arguments": self.arguments,
        }


class FileParam(BaseModel):
    file: FileSubParam
    type: Literal["file"] = Field(default="file")


class TextParam(BaseModel):
    text: str
    type: Literal["text"] = Field(default="text")

    max_text_len: ClassVar[int] = 4000

    def to_openai_format(self) -> str:
        return self.text

    def to_openai_format_text(self) -> dict:
        return {"text": self.text, "type": "text"}


class ImageParam(BaseModel):
    image_url: ImageURLSubParam
    type: Literal["image_url"] = Field(default="image_url")

    @classmethod
    def from_base64(
        cls,
        img_type: Literal["jpeg", "png", "jpg"],
        base64_str: str,
        detail: Literal["auto", "low", "high"] = "auto",
    ) -> "ImageParam":
        return cls(
            image_url=ImageURLSubParam.from_base64(img_type, base64_str, detail),
            type="image_url",
        )

    def to_openai_format(self, hide_image: bool = False) -> dict:
        return {
            "image_url": self.image_url.to_openai_format(hide_image),
            "type": self.type,
        }


class InputAudioParam(BaseModel):
    input_audio: InputAudioSubParam
    type: Literal["input_audio"] = Field(default="input_audio")


class ToolCallParam(BaseModel):
    id: str
    function: FunctionSubParam
    type: Literal["function"] = Field(default="function")

    def to_openai_format(self) -> dict:
        return {
            "id": self.id,
            "function": self.function.to_openai_format(),  # type: ignore
            "type": self.type,
        }

    def __str__(self) -> str:
        return (
            f"ToolCallParam(\n" f"  id='{self.id}',\n" f"  type='{self.type}',\n" f"  function={self.function}\n" f")"
        )


class RefusalParam(BaseModel):
    refusal: str
    type: Literal["refusal"] = Field(default="refusal")

    def to_openai_format(self) -> dict:
        return {
            "refusal": self.refusal,
            "type": self.type,
        }
