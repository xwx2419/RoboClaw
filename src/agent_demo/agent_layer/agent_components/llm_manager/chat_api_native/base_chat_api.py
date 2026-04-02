import logging
from abc import ABC, abstractmethod
from agent_demo.types.agent_types import (
    llmState,
    llmStateTransition,
    BaseAgentCard,
)

logger = logging.getLogger(__name__)


class BaseChatAPI(ABC):
    _TRANSITIONS: dict[llmState, frozenset[llmState]] = llmStateTransition._TRANSITIONS

    def __init__(self, agent_card: BaseAgentCard):
        logger.debug("[Init][OpenAI_Client][Start]")
        self._state = llmState.INIT
        self._agent_card_ref: BaseAgentCard = agent_card

    # ========== 只读 ==========

    @property
    def state(self) -> llmState:
        return self._state

    # ========== 核心抽象接口 ==========

    @abstractmethod
    async def _chat(
        self,
        send_package,
    ):
        raise NotImplementedError

    @abstractmethod
    async def _stream_chat(
        self,
        data_package,
    ):
        raise NotImplementedError

    # ========== 状态管理 ==========

    def get_allowed_transitions(self, current_state) -> frozenset[llmState]:
        return self._TRANSITIONS.get(current_state, frozenset())

    def _transition_state(self, new_state: llmState):
        allowed: frozenset[llmState] = self.get_allowed_transitions(self._state)
        if new_state not in allowed:
            raise RuntimeError(f"State transition ERROR | {self._state} → {new_state}")
        logger.debug(f"State transition | {self._state} → {new_state}")
        self._state = new_state
