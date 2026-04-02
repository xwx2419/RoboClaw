import logging
import asyncio
from agent_demo.agent_layer.agent_components import OpenAIClient, MemoryManager, ORMCPServiceManager, AgentTools
from agent_demo.types.agent_types import (
    ActAgentState,
    ActAgentStateTransition,
    TaskNode,
    BaseAgentCard,
)
from collections import deque
from agent_demo.types.interaction_types import InteractionPackage
from agent_demo.common.root_logger import table_to_str
from rich.table import Table

logger = logging.getLogger(__name__)


class BaseAgent:

    _TRANSITIONS: dict[ActAgentState, frozenset[ActAgentState]] = ActAgentStateTransition._TRANSITIONS

    def __init__(
        self,
        agent_card: BaseAgentCard,
    ):
        self._state: ActAgentState = ActAgentState.INIT
        self._agent_card_ref: BaseAgentCard = agent_card
        self._llm_client: OpenAIClient = OpenAIClient(agent_card=agent_card)
        self._memory_manager: MemoryManager = MemoryManager(agent_card=agent_card)
        self._service_manager: ORMCPServiceManager = ORMCPServiceManager(agent_card=agent_card)
        self._agent_tools: AgentTools = AgentTools(
            memory_manager=self._memory_manager,
            service_manager=self._service_manager,
            agent_card=agent_card,
            agent_instance=self,  # 传递 agent 实例引用，用于访问 LLM 客户端
        )

    @property
    def current_total_tokens(self) -> int:
        return self._agent_card_ref.all_token_usage

    # ========== 优雅退出 ==========
    async def shutdown(self) -> None:
        await self._llm_client.shutdown()
        await self._memory_manager.shutdown()
        await self._service_manager.shutdown()
        await asyncio.sleep(1)  # 等待1秒后退出事件循环
        await self._agent_tools.shutdown()

    async def terminate(self) -> None:
        await self._llm_client.terminate()
        await self._memory_manager.terminate()
        await self._service_manager.terminate()
        await self._agent_tools.terminate()

    # ========= 初始化方法 =========
    async def init_agent(self):
        logger.info("[Init][Agent][Start]")
        await self._llm_client.init_client()
        await self._memory_manager.init_memory()
        await self._service_manager.init_services()
        await self._agent_tools.init_agent_tools()
        self._transition_state(ActAgentState.READY)
        logger.info("[Init][Agent][Done]")

    # ========== 状态管理 ==========

    def get_allowed_transitions(self, current_state) -> frozenset:
        return self._TRANSITIONS.get(current_state, frozenset())

    def _transition_state(self, new_state: ActAgentState):
        allowed = self.get_allowed_transitions(self._state)
        logger.info(f"{self.agent_id} | {self._state} → {new_state}")
        if new_state not in allowed:
            raise RuntimeError(f"{self.agent_id} ERROR | {self._state} → {new_state}")
        self._state = new_state

    # ========= 属性方法 =========
    @property
    def llm_client(self) -> OpenAIClient:
        return self._llm_client

    @property
    def memory_manager(self) -> MemoryManager:
        return self._memory_manager

    @property
    def service_manager(self) -> ORMCPServiceManager:
        return self._service_manager

    @property
    def agent_tools(self) -> AgentTools:
        return self._agent_tools

    @property
    def agent_id(self) -> str:
        return self._agent_card_ref.agent_id

    @property
    def display_deque(self) -> deque[InteractionPackage]:
        return self._agent_card_ref.display_deque

    @property
    def current_task_node(self) -> TaskNode:
        return self._memory_manager.current_task_node

    @property
    def ready_to_chat(self) -> bool:
        if self._state == ActAgentState.READY:
            return True
        return False

    @property
    def state(self) -> ActAgentState:
        return self._state

    @property
    def state_str(self) -> str:
        return self._state.name

    @property
    def available_tools(self) -> list[dict]:
        return self._service_manager.activate_tools_list

    @property
    def current_contexts(self) -> list[dict]:
        return self._memory_manager.current_contexts

    # ========== 打印方法 ===========
    async def show_workflow_info_as_table(self) -> None:
        table = Table(title=f"{self._agent_card_ref.agent_id}", show_lines=False, box=None, expand=False)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="magenta")
        table.add_row("llm_model", self._agent_card_ref.config.cache_key)
        table.add_row("max_completion_tokens", str(self._agent_card_ref.config.max_completion_tokens))
        table.add_row(
            "memory_tokens/compression_threshold",
            str(self._agent_card_ref.all_token_usage) + "/" + str(self._agent_card_ref.config.compression_threshold),
        )
        table.add_row("services_register_len", str(len(self._service_manager._services_register_list)))
        self._agent_card_ref.display_deque.append(
            InteractionPackage(
                display_widget="left_log",
                content_type="SystemMessageType",
                agent_id=self._agent_card_ref.agent_id,
                content=table,
            )
        )
        logger.info(table_to_str(table))
