from .runtime_memory_tree.runtime_memory import RuntimeMemoryTree
from agent_demo.types.agent_types import (
    ServiceRegister,
    AssistantMessageType,
    CompressPolicy,
    RobotImgMessageType,
    TextParam,
    BaseAgentCard,
)
from typing_extensions import Literal
import logging

logger = logging.getLogger(__name__)


# 基本都是内存操作，异步收益不高，套壳就好
class MemoryManager(RuntimeMemoryTree):
    def __init__(
        self,
        agent_card: BaseAgentCard,
    ):
        super().__init__(agent_card=agent_card)
        self._memory_tree_static: str = ""

    # ========= 属性方法 =========
    @property
    def current_contexts(self) -> list[dict]:  # 先不上记忆缓存了，直接实时聚合，这里不是性能瓶颈
        # 在生成上下文前，先清理可能存在的孤立 tool 消息（作为最后一道防线）
        self.current_task_node._cleanup_orphaned_tool_messages()

        sys_contexts: list[dict] = []
        chat_contexts: list[dict] = []
        img_cnt: int = 0
        # 收集系统上下文（老到新）
        if self.self_know.Contexts is not None:
            sys_contexts.append(self.self_know.Contexts.to_openai_format())
        if self.knowledge_graph_caching_block.Contexts is not None:
            sys_contexts.append(self.knowledge_graph_caching_block.Contexts.to_openai_format())
        if self.service_registry_block.Contexts is not None:
            sys_contexts.append(self.service_registry_block.Contexts.to_openai_format())
        if self.task_template_block.Contexts is not None:
            sys_contexts.append(self.task_template_block.Contexts.to_openai_format())
        if self.task_session_block.Contexts is not None:
            sys_contexts.append(self.task_session_block.Contexts.to_openai_format())
        if self.current_task_node.task_sys_context is not None:
            sys_contexts.append(self.current_task_node.task_sys_context.to_openai_format())
        # 收集对话上下文
        for context in reversed(self.current_task_node.contexts):  # 从新到老遍历
            if isinstance(context.content, RobotImgMessageType):
                img_cnt += 1
                chat_contexts.append(context.to_openai_format(hide_image=(img_cnt > self._agent_card.img_threshold)))
            else:
                chat_contexts.append(context.to_openai_format())
        # 最后整体反转为老到新的顺序
        chat_contexts.reverse()
        # 拼接上下文：系统上下文（老到新） + 对话上下文（老到新）
        return sys_contexts + chat_contexts

    # ========= 初始化方法 =========
    async def init_memory(self):
        logger.info("[Init][Memory][Start]")
        await self.init_memory_tree()
        await self.init_default_task()
        logger.info("[Init][Memory][Done]")

    async def init_memory_tree(self):
        self._init_memory_tree()

    async def init_default_task(self):
        self._init_default_task_session_block()

    async def create_task(self, task_brief: str, assistant_guidance: str):
        self.create_new_task_node(
            task_brief=task_brief,
            assistant_guidance=assistant_guidance,
        )

    # ========= 更新记忆 =========
    async def update_service_registry(self, services_list: list[ServiceRegister]):
        services_list_str = ""
        for service in services_list:
            services_list_str += service.to_service_registry_block_prompt()
        self.update_service_registry_block_content(services_list_str)
        logger.info(f"[Update][{self.service_registry_block.base_info.name_en}][Done]")
        logger.debug(f"[Update][{self.service_registry_block.Contexts}][Done]")

    async def add_user_str_message(self, msg: str):
        self._add_user_str_message(msg)

    async def add_robot_img_message(
        self,
        img_frame_id: int,
        base64_str: str,
        img_type: Literal["jpeg", "png", "jpg"] = "jpeg",
        detail: Literal["auto", "low", "high"] = "auto",
    ):
        self._add_robot_img_message(img_frame_id=img_frame_id, base64_str=base64_str, img_type=img_type, detail=detail)

    async def add_agent_message_type(self, msg: AssistantMessageType):
        self._add_agent_message_type(msg)

    async def add_robot_call_back_text_message(self, msg: TextParam, tool_call_id: str):
        self._add_robot_call_back_text_message(msg, tool_call_id)

    async def add_compress_request_message(self):
        await self.add_user_str_message(
            self._agent_card.agent_memory_prompt.get("DEFAULT_COMPRESS_REQUEST_TEMPLATE", "")
        )

    # ========== 压缩记忆 ==========
    async def compress_current_memory(self):
        if self._agent_card.compress_policy is CompressPolicy.DISCARD_OLDEST:
            await self.compress_policy_discard_oldest()
        elif self._agent_card.compress_policy is CompressPolicy.COMPRESS_ALL:
            await self.compress_policy_compress_all()
        else:
            raise ValueError(f"Unsupported compress policy: {self._agent_card.compress_policy}")

    async def compress_policy_discard_oldest(self, drop_n: int = 2):
        total = len(self.current_task_node.contexts)
        if total == 0 or drop_n <= 0:
            return
        actual_drop = min(drop_n, total)
        start_index = self.current_task_node.contexts[0].index
        stop_index = self.current_task_node.contexts[actual_drop - 1].index
        self.current_task_node.compress_policy_discard_oldest(actual_drop)
        logger.info(
            f"---[{CompressPolicy.DISCARD_OLDEST}({self._agent_card.compress_cnt})]---[start_index({start_index})]---[stop_index({stop_index})]---"
        )

    async def compress_policy_compress_all(self):
        # 迁移当前上下文
        start_index = self.current_task_node.contexts[0].index
        stop_index = self.current_task_node.contexts[-2].index
        self.current_task_node.compress_policy_compress(1)
        # 重建记忆缓存
        self._agent_card.compress_cnt += 1
        logger.info(
            f"---[{CompressPolicy.COMPRESS_ALL}({self._agent_card.compress_cnt})]---[start_index({start_index})]---[stop_index({stop_index})]---"
        )
        logger.info(self.current_task_node.contexts)

    # ========== 优雅退出 ==========
    async def shutdown(self) -> None:
        return

    async def terminate(self) -> None:
        await self.shutdown()
