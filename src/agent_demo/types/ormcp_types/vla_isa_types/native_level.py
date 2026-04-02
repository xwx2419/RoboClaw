from pydantic import BaseModel, Field
from typing_extensions import Literal
from enum import Enum

# 拿起xxx、调整xxx、放下xxx、擦拭xxx、折叠、将xxx放入xxx、推/拉xxx、抓握并旋转xxx、移动至xxx、从xxx移开、打开/关闭xxx


# 定义动作模板
class NativeActionTemplate_cn(str, Enum):
    pick_up = "拿起{item}"
    adjust = "调整{item}"
    put_down = "放下{item}"
    wipe = "擦拭{position}"
    fold = "折叠{item}"
    put_into = "将{item}放入{container}"  # container的本质也是item
    push = "推{item}"
    pull = "拉{item}"
    grasp_and_turn = "抓握并旋转{item}"
    move_to = "移动至{position}"
    move_away_from = "从{position}移开"
    return_to_init_position = "返回初始位置{position}"
    open = "打开{item}"
    close = "关闭{item}"
    rotate = "旋转{item}"

    def to_str(self) -> str:
        return self.value


# 定义Baseitem类
class Baseitem(BaseModel):
    name: str
    property: Literal["item", "container"]


# 定义BaseNativeISA类
class BaseNativeISA(BaseModel):
    action: NativeActionTemplate_cn
    items: list[Baseitem]
    position: str
    action_text: str = Field(default="", exclude=True)

    # @model_validator(mode='after')
    # def fill_template_after(cls, values: 'BaseNativeISA'):
    #     action_template: NativeActionTemplate_cn = values.action
    #     items: list[Baseitem] = values.items
    #     position: str = values.position

    #     if not action_template:
    #         return values

    #     item_names = [item.name for item in items if item.property == "item"]
    #     container_names = [item.name for item in items if item.property == "container"]

    #     item_count = action_template.value.count("{item}")
    #     if item_count > 1:
    #         raise ValueError("'{item}' placeholder cannot appear more than once.")
    #     if item_count == 1 and len(item_names) != 1:
    #         raise ValueError("Exactly one item is required to replace '{item}'.")

    #     container_count = action_template.value.count("{container}")
    #     if container_count > 0 and action_template != NativeActionTemplate_cn.put_into:
    #         raise ValueError("'{container}' is only valid in the 'put_into' action.")
    #     if container_count > 1:
    #         raise ValueError("'{container}' placeholder cannot appear more than once.")
    #     if container_count == 1 and len(container_names) != 1:
    #         raise ValueError("Exactly one container is required to replace '{container}'.")

    #     result_action = action_template.value
    #     if "{item}" in result_action:
    #         result_action = result_action.replace("{item}", item_names[0])
    #     if "{container}" in result_action:
    #         result_action = result_action.replace("{container}", container_names[0])
    #     if "{position}" in result_action:
    #         if not position:
    #             raise ValueError("Position must be provided to replace '{position}'.")
    #         result_action = result_action.replace("{position}", position)

    #     values.action_text = result_action
    #     return values

    def __str__(self):
        return self.action_text or self.action.value
