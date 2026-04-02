from enum import Enum, auto


class SessionStatus(Enum):
    """任务执行模式枚举"""

    INIT = auto()  # 初始状态(刚创建)
    READY = auto()  # 就绪状态(初始化完成，已经准备好处理对话请求)
    RUNNING = auto()  # 运行状态
    SHUTDOWN = auto()  # 优雅关闭状态(正在关闭)
    CLOSED = auto()  # 关闭状态(最终状态)


class SessionStatusTransition:
    _TRANSITIONS: dict[SessionStatus, frozenset[SessionStatus]] = {
        SessionStatus.INIT: frozenset({SessionStatus.READY}),
        SessionStatus.READY: frozenset({SessionStatus.RUNNING, SessionStatus.SHUTDOWN}),
        SessionStatus.RUNNING: frozenset({SessionStatus.SHUTDOWN, SessionStatus.CLOSED}),
        SessionStatus.SHUTDOWN: frozenset({SessionStatus.CLOSED}),
    }
