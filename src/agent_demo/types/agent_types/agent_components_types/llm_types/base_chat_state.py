from enum import Enum, auto


# 定义对话的状态
class llmState(Enum):
    INIT = auto()  # 等待输入
    READY = auto()  # 等待输入
    SYNC_GENERATING = auto()  # 同步生成响应中
    STREAM_GENERATING = auto()  # 流式生成响应中
    CANCEL_STREAM_GENERATING = auto()  # 取消流式生成
    STREAM_PAUSED = auto()  # 暂停流式生成
    STREAM_RESUMING = auto()  # 恢复流式生成
    COMPLETED = auto()  # 完成


class llmStateTransition:
    _TRANSITIONS: dict[llmState, frozenset[llmState]] = {
        llmState.INIT: frozenset({llmState.READY}),
        llmState.READY: frozenset({llmState.SYNC_GENERATING, llmState.STREAM_GENERATING}),
        llmState.STREAM_GENERATING: frozenset(
            {llmState.CANCEL_STREAM_GENERATING, llmState.STREAM_PAUSED, llmState.COMPLETED}
        ),
        llmState.CANCEL_STREAM_GENERATING: frozenset({llmState.COMPLETED}),
        llmState.STREAM_PAUSED: frozenset({llmState.STREAM_RESUMING}),
        llmState.STREAM_RESUMING: frozenset({llmState.COMPLETED, llmState.STREAM_PAUSED}),
        llmState.SYNC_GENERATING: frozenset(
            {llmState.COMPLETED, llmState.READY}
        ),  # 允许转换到 COMPLETED 或 READY（异常恢复）
        llmState.COMPLETED: frozenset({llmState.READY}),
    }
