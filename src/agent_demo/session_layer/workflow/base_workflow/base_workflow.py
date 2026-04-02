from agent_demo.types.interaction_types import InteractionPackage
from agent_demo.agent_layer.agent_core import BaseAgent
from collections import deque
from agent_demo.common.root_logger import table_to_str
from rich.table import Table
import logging

logger = logging.getLogger(name=__name__)


class BaseWorkflow:
    def __init__(self, agent: BaseAgent, display_deque: deque[InteractionPackage]):
        self._agent: BaseAgent = agent
        self._display_deque: deque[InteractionPackage] = display_deque

    # ========== 运行 ==========
    async def run_once(self):
        pass

    # ========== 打印方法 ===========
    async def show_workflow_info_as_table(self) -> None:
        table = Table(title=f"{self._agent.agent_id}", show_lines=False, box=None, expand=False)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="magenta")
        table.add_row("llm_model", self._agent._agent_card_ref.config.cache_key)
        table.add_row("max_completion_tokens", str(self._agent._agent_card_ref.config.max_completion_tokens))
        table.add_row(
            "memory_tokens/compression_threshold",
            str(self._agent.current_total_tokens) + "/" + str(self._agent._agent_card_ref.config.compression_threshold),
        )
        table.add_row("services_register_len", str(len(self._agent.service_manager._services_register_list)))
        self._display_deque.append(
            InteractionPackage(
                display_widget="left_log",
                content_type="SystemMessageType",
                agent_id=self._agent.agent_id,
                content=table,
            )
        )
        logger.info(table_to_str(table))
