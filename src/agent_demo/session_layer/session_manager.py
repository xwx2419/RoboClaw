# from mcp_client_demo.types.agent_types import ChatAPIConfig, BaseAgentCard
# from mcp_client_demo.types.interaction_types import InteractionPackage
# from mcp_client_demo.agent_layer.agent_core import ActAgent
# from .base_session.base_session import BaseSession
# from collections import deque
# import logging

# logger = logging.getLogger(__name__)


# class SessionManager(BaseSession):
#     def __init__(self, agent_card: BaseAgentCard):
#         super().__init__()
#         self._agent_card: BaseAgentCard = agent_card
#         self.agent = ActAgent(agent_card=self._agent_card)

#     # ========= 属性方法 =========
#     @property
#     def ready_to_chat(self) -> bool:
#         return self.agent.ready_to_chat

#     @property
#     def config(self) -> ChatAPIConfig:
#         return self._agent_card.config

#     @property
#     def service_config_path(self) -> str:
#         return self._agent_card.service_config_path

#     @property
#     def display_deque(self) -> deque[InteractionPackage]:
#         return self._agent_card.display_deque

#     # ========== 初始化 ==========
#     async def init(self) -> None:
#         await self.agent.init_agent()

#     # ========== 优雅退出 ==========
#     async def shutdown(self) -> None:
#         await self.agent.shutdown()
#         logger.info("[🚪] agent shutdown down")

#     async def terminate(self) -> None:
#         raise NotImplementedError
