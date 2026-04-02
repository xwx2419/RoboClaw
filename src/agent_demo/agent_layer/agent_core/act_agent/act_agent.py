import uuid
import json
import logging
from agent_demo.types.interaction_types import InteractionPackage
from ..base_agent.base_agent import BaseAgent
from agent_demo.types.agent_types import (
    OpenAISendMsg,
    OpenAIResponseMsg,
    ORMCP_TOOLS_SPLICE,
    TextParam,
    ActAgentState,
    BaseAgentCard,
)

logger = logging.getLogger(name=__name__)


class ActAgent(BaseAgent):

    _agent_name: str = "Hephaestus"

    def __init__(
        self,
        agent_card: BaseAgentCard,
    ):
        super().__init__(agent_card=agent_card)
        self._agent_card_ref.agent_name = self._agent_name
        self._agent_card_ref.agent_id = self._agent_name + "-" + uuid.uuid4().hex[:8]

    # ========= 核心方法 =========
    # user -> chat -> act -> response -> compress_memory
    async def chat(self, messages: str) -> OpenAIResponseMsg:
        self._transition_state(ActAgentState.CHAT)
        await self._memory_manager.add_user_str_message(messages)
        res: OpenAIResponseMsg = await self._llm_client.sync_chat(
            send_package=OpenAISendMsg(contexts=self.current_contexts, tools_list=self.available_tools)
        )
        self._current_total_tokens = res.total_tokens
        await self._memory_manager.add_agent_message_type(msg=res.first_choice.message)
        return res

    async def act(self, messages: OpenAIResponseMsg) -> TextParam:
        self._transition_state(ActAgentState.ACT)
        res: TextParam = TextParam(text="未知错误")
        for tool_call in messages.first_choice.tool_calls:
            tool_call_id = tool_call.id
            service_name, tool_name = tool_call.function.name.split(ORMCP_TOOLS_SPLICE)
            tool_args: dict[str, object] = json.loads(tool_call.function.arguments)
            logger.info(f"[Routing Info] ---> [{service_name}][{tool_name}][{tool_args}]")
            self.display_deque.append(
                InteractionPackage(
                    content_type="analyze_response",
                    agent_id=self.agent_id,
                    content=f"[Routing Info] ---> [{service_name}][{tool_name}][{tool_args}]",
                )
            )
            if self._service_manager.check_is_agent_service(service_name):  # agent service
                res = await self._agent_tools.tools_routing(service_name, tool_name, tool_args)
            else:  # mcp service
                res = await self._service_manager.tools_routing(service_name, tool_name, tool_args)
            await self._memory_manager.add_robot_call_back_text_message(res, tool_call_id)
            await self._agent_tools.flush_pending_context_injections()

        return res

    async def response(self) -> OpenAIResponseMsg:
        self._transition_state(ActAgentState.RESPONSE)
        res: OpenAIResponseMsg = await self._llm_client.sync_chat(
            OpenAISendMsg(contexts=self.current_contexts, tools_list=self.available_tools)
        )
        self._current_total_tokens = res.total_tokens
        await self._memory_manager.add_agent_message_type(msg=res.first_choice.message)
        return res

    async def compress_memory(self, messages: OpenAIResponseMsg):
        if messages.need_compress:
            self._transition_state(ActAgentState.COMPRESS)
            await self._memory_manager.add_compress_request_message()
            res: OpenAIResponseMsg = await self._llm_client.sync_chat(
                OpenAISendMsg(contexts=self.current_contexts, tools_list=self.available_tools)
            )
            self._current_total_tokens = res.total_tokens
            await self._memory_manager.add_agent_message_type(msg=res.first_choice.message)
            await self._memory_manager.compress_current_memory()

        self._transition_state(ActAgentState.READY)

    async def reset(self):
        pass

    async def run_once(self, user_input: str) -> TextParam | None:
        res: OpenAIResponseMsg = await self.chat(user_input)
        logger.info(f"[😼 ]{res.first_choice.message}")
        while res.has_tool_call:
            tool_res: TextParam = await self.act(res)
            self._agent_card_ref.display_deque.append(
                InteractionPackage(
                    content_type="analyze_response",
                    agent_id=self._agent_card_ref.agent_id,
                    content=f"...........{tool_res}",
                )
            )
            logger.info(f"[🤖 ]{tool_res}")
            res = await self.response()
            logger.info(f"[😼 ]{res.first_choice.message}")
        await self.compress_memory(res)
        return res.first_choice.message.content  # type: ignore
