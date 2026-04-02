from enum import Enum, auto
from ...agent_components_types.memory_types.base.base_context_types import TextParam
from typing import Callable, Awaitable, TypeAlias


# 定义act agent工作状态
class ActAgentState(Enum):
    INIT = auto()  # 初始化状态
    READY = auto()  # 准备就绪
    CHAT = auto()  # 获取用户输入，正在生成响应
    ACT = auto()  # 云端模型请求工具调用，正在调用工具
    RESPONSE = auto()  # 已经获得工具返回结果，正在通知云端模型
    COMPRESS = auto()  # 正在压缩上下文
    ERROR = auto()  # 错误状态
    CLOSE = auto()  # 关闭状态


class ActAgentStateTransition:
    _TRANSITIONS: dict[ActAgentState, frozenset[ActAgentState]] = {
        ActAgentState.INIT: frozenset({ActAgentState.READY}),
        ActAgentState.READY: frozenset({ActAgentState.CHAT, ActAgentState.CLOSE}),
        ActAgentState.CHAT: frozenset({ActAgentState.ACT, ActAgentState.COMPRESS, ActAgentState.READY}),
        ActAgentState.ACT: frozenset({ActAgentState.RESPONSE}),
        ActAgentState.RESPONSE: frozenset({ActAgentState.ACT, ActAgentState.COMPRESS, ActAgentState.READY}),
        ActAgentState.COMPRESS: frozenset({ActAgentState.READY}),
    }


AgentToolsFunc: TypeAlias = Callable[..., str | TextParam] | Callable[..., Awaitable[str | TextParam]]
