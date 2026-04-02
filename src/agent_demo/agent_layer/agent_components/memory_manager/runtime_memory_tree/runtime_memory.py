from agent_demo.types.interaction_types import InteractionPackage
from abc import ABC
import logging
from typing import Optional
from typing_extensions import Literal
from agent_demo.types.agent_types import (
    RuntimeMemory,
    MemoryName,
    SelfKnowledgeArea,
    KnowledgeGraphCachingBlock,
    ServerRegistryBlock,
    TaskTemplateBlock,
    LongTermMemoryArea,
    TaskNode,
    TaskSessionBlock,
    ShortTermMemoryArea,
    SysContent,
    ChatContent,
    SystemMessageType,
    SystemDynamicMessageType,
    UserMessageType,
    AssistantMessageType,
    ToolMessageType,
    RobotImgMessageType,
    TextParam,
    BaseAgentCard,
)
from dataclasses import dataclass

logger = logging.getLogger(name=__name__)


@dataclass
class RuntimeMemoryNode:
    key: Optional[
        RuntimeMemory
        | SelfKnowledgeArea
        | KnowledgeGraphCachingBlock
        | ServerRegistryBlock
        | TaskTemplateBlock
        | LongTermMemoryArea
        | TaskNode
        | TaskSessionBlock
        | ShortTermMemoryArea
    ] = None
    children: Optional[list["RuntimeMemoryNode"]] = None


class RuntimeMemoryTree(ABC):
    def __init__(self, agent_card: BaseAgentCard) -> None:
        self._agent_card: BaseAgentCard = agent_card
        # 默认任务
        self._current_task_node: RuntimeMemoryNode = RuntimeMemoryNode()
        # 长期记忆区
        self._knowledge_graph_caching_block: RuntimeMemoryNode = RuntimeMemoryNode(key=KnowledgeGraphCachingBlock())
        self._service_registry_block: RuntimeMemoryNode = RuntimeMemoryNode(key=ServerRegistryBlock())
        self._task_template_block: RuntimeMemoryNode = RuntimeMemoryNode(key=TaskTemplateBlock())
        self._long_memory_area = RuntimeMemoryNode(
            key=LongTermMemoryArea(),
            children=[self._knowledge_graph_caching_block, self._service_registry_block, self._task_template_block],
        )
        # 短期记忆区
        self._task_session_block: RuntimeMemoryNode = RuntimeMemoryNode(key=TaskSessionBlock())
        self._short_memory_area = RuntimeMemoryNode(key=ShortTermMemoryArea(), children=[self._task_session_block])
        # 自我认知区
        self._self_know_area: RuntimeMemoryNode = RuntimeMemoryNode(key=SelfKnowledgeArea())
        # 运行时记忆区
        self._root: RuntimeMemoryNode = RuntimeMemoryNode(
            key=RuntimeMemory(), children=[self._self_know_area, self._long_memory_area, self._short_memory_area]
        )

    # ========= 属性 =========
    @property
    def auto_index(self) -> int:
        index: int = self._agent_card.root_index
        self._agent_card.root_index += 1
        return index

    @property
    def self_know(self) -> SelfKnowledgeArea:
        return self._self_know_area.key  # type: ignore

    @property
    def long_memory(self) -> LongTermMemoryArea:
        return self._long_memory_area.key  # type: ignore

    @property
    def knowledge_graph_caching_block(self) -> KnowledgeGraphCachingBlock:
        return self._knowledge_graph_caching_block.key  # type: ignore

    @property
    def service_registry_block(self) -> ServerRegistryBlock:
        return self._service_registry_block.key  # type: ignore

    @property
    def task_template_block(self) -> TaskTemplateBlock:
        return self._task_template_block.key  # type: ignore

    @property
    def task_session_block(self) -> TaskSessionBlock:
        return self._task_session_block.key  # type: ignore

    @property
    def short_memory(self) -> ShortTermMemoryArea:
        return self._short_memory_area.key  # type: ignore

    @property
    def current_task_node(self) -> TaskNode:
        return self._current_task_node.key  # type: ignore

    # ========= 初始化方法 =========
    def _init_memory_tree(self):
        self._init_self_memory_content()
        self._init_long_memory_content()
        self._init_short_memory_content()

    def _init_long_memory_content(self):
        self._init_knowledge_graph_caching_block()
        self._init_service_registry_block_content()
        self._init_task_template_block_content()
        logger.debug(f"[Init][{self.long_memory.base_info.name_en}][Done]")

    def _init_short_memory_content(self):
        self._init_task_session_block_content()
        logger.debug(f"[Init][{self.short_memory.base_info.name_en}][Done]")

    def _init_self_memory_content(self):
        self.update_self_memory_content()
        logger.debug(f"[Init][{self.self_know.base_info.name_en}][Done]")

    def _init_knowledge_graph_caching_block(self):
        self.update_knowledge_graph_caching_block()
        logger.debug(f"[Init][{self.knowledge_graph_caching_block.base_info.name_en}][Done]")

    def _init_service_registry_block_content(self):
        self.update_service_registry_block_content("")
        logger.debug(f"[Init][{self.service_registry_block.base_info.name_en}][Done]")

    def _init_task_template_block_content(self):
        self.update_task_template_block_content()
        logger.debug(f"[Init][{self.task_template_block.base_info.name_en}][Done]")

    def _init_task_session_block_content(self):
        self.update_task_session_block_content()
        logger.debug(f"[Init][{self.task_session_block.base_info.name_en}][Done]")

    def _init_default_task_session_block(self):
        # 创建默认任务节点
        self.create_new_task_node(
            task_brief=self._agent_card.agent_memory_prompt.get("DEFAULT_TASK_BRIEF_TEMPLATE", ""),
            assistant_guidance=self._agent_card.agent_memory_prompt.get("DEFAULT_ASSISTANT_GUIDANCE_TEMPLATE", ""),
        )
        logger.debug(f"[Init][{self.task_session_block.base_info.name_en}][Done]")
        logger.debug(f"[Init][{self.current_task_node.base_info.name_en}][Done]")

    # ========= 更新方法 =========
    def update_sys_content_for_static_block(
        self,
        block: (
            SelfKnowledgeArea | KnowledgeGraphCachingBlock | ServerRegistryBlock | TaskTemplateBlock | TaskSessionBlock
        ),
        template: str,
        **kwargs,
    ) -> SysContent:
        if template is None or template == "":
            raise ValueError("template should not be empty or ''.")

        if isinstance(block, SelfKnowledgeArea):
            content = SystemMessageType.text_param(text=template.format(**kwargs))
        else:
            content = SystemDynamicMessageType.text_param(text=template.format(**kwargs))

        # 根据不同的 block_type，动态地设置 Contexts 和追加到 _display_deque 中
        block.Contexts = SysContent(
            index=self.auto_index,
            content=content,
        )
        self._agent_card.display_deque.append(
            InteractionPackage(
                display_widget="left_log",
                content_type="sys_content",
                agent_id=self._agent_card.agent_id,
                content=block.Contexts.rich_table(),
            )
        )
        return block.Contexts

    def update_sys_context_for_current_task_node(
        self,
        template: str,
        **kwargs,
    ) -> SysContent:

        content = SystemDynamicMessageType.text_param(text=template.format(**kwargs))

        self.current_task_node.task_sys_context = SysContent(
            index=self.auto_index,
            content=content,
        )
        self._agent_card.display_deque.append(
            InteractionPackage(
                display_widget="left_log",
                content_type="sys_content",
                agent_id=self._agent_card.agent_id,
                content=self.current_task_node.task_sys_context.rich_table(),
            )
        )
        return self.current_task_node.task_sys_context

    def update_self_memory_content(self) -> SysContent:
        template = self._agent_card.agent_memory_prompt.get("SELF_KNOWLEDGE_TEMPLATE", "")
        if template is None or template == "":
            raise ValueError("template should not be empty or ''.")

        ext = self._agent_card.agent_memory_prompt.get("SELF_KNOWLEDGE_EXTENSION", "")
        kwargs = dict(
            memory_tree=self.get_full_tree(),
            prefix=self.self_know.base_info.prefix,
            emoji=self.self_know.base_info.emoji,
            name_cn=self.self_know.base_info.name_cn,
            updated_at=self.self_know.base_info.update_time(),
            sn_code=self._agent_card.config.client_name,
            ormcp_version=MemoryName.ORMCP_VERSION,
            self_knowledge_extension=ext,
        )
        text = template.format(**kwargs)

        block = self.self_know
        block.Contexts = SysContent(
            index=self.auto_index,
            content=SystemMessageType.text_param(text=text),
        )
        self._agent_card.display_deque.append(
            InteractionPackage(
                display_widget="left_log",
                content_type="sys_content",
                agent_id=self._agent_card.agent_id,
                content=block.Contexts.rich_table(),
            )
        )
        return block.Contexts

    def update_knowledge_graph_caching_block(self) -> SysContent:
        return self.update_sys_content_for_static_block(
            block=self.knowledge_graph_caching_block,
            template=self._agent_card.agent_memory_prompt.get("KNOWLEDGE_GRAPH_CACHING_TEMPLATE", ""),
            l_prefix=self.long_memory.base_info.prefix,
            l_emoji=self.long_memory.base_info.emoji,
            l_name_cn=self.long_memory.base_info.name_cn,
            prefix=self.knowledge_graph_caching_block.base_info.prefix,
            emoji=self.knowledge_graph_caching_block.base_info.emoji,
            name_cn=self.knowledge_graph_caching_block.base_info.name_cn,
            updated_at=self.knowledge_graph_caching_block.base_info.update_time(),
        )

    def update_service_registry_block_content(self, services_list_str: str) -> SysContent:
        return self.update_sys_content_for_static_block(
            block=self.service_registry_block,
            template=self._agent_card.agent_memory_prompt.get("SERVER_REGISTRY_TEMPLATE", ""),
            prefix=self.service_registry_block.base_info.prefix,
            emoji=self.service_registry_block.base_info.emoji,
            name_cn=self.service_registry_block.base_info.name_cn,
            updated_at=self.service_registry_block.base_info.update_time(),
            services_list=services_list_str,
        )

    def update_task_template_block_content(self) -> SysContent:
        return self.update_sys_content_for_static_block(
            block=self.task_template_block,
            template=self._agent_card.agent_memory_prompt.get("TASK_TEMPLATE_TEMPLATE", ""),
            prefix=self.task_template_block.base_info.prefix,
            emoji=self.task_template_block.base_info.emoji,
            name_cn=self.task_template_block.base_info.name_cn,
            updated_at=self.task_template_block.base_info.update_time(),
        )

    def update_task_session_block_content(self) -> SysContent:
        return self.update_sys_content_for_static_block(
            block=self.task_session_block,
            template=self._agent_card.agent_memory_prompt.get("TASK_SESSION_TEMPLATE", ""),
            s_prefix=self.short_memory.base_info.prefix,
            s_emoji=self.short_memory.base_info.emoji,
            s_name_cn=self.short_memory.base_info.name_cn,
            prefix=self.task_session_block.base_info.prefix,
            emoji=self.task_session_block.base_info.emoji,
            name_cn=self.task_session_block.base_info.name_cn,
            updated_at=self.task_session_block.base_info.update_time(),
        )

    def create_new_task_node(self, task_brief: str, assistant_guidance: str) -> SysContent:
        self._current_task_node = RuntimeMemoryNode(
            key=TaskNode(
                task_brief=task_brief,
                assistant_guidance=assistant_guidance,
            )
        )

        if self._task_session_block.children is None:
            self._task_session_block.children = []
        self._task_session_block.children.append(self._current_task_node)

        return self.update_sys_context_for_current_task_node(
            template=self._agent_card.agent_memory_prompt.get("TASK_NODE_START_TEMPLATE", ""),
            prefix=self.current_task_node.base_info.prefix,
            emoji=self.current_task_node.base_info.emoji,
            name_cn=self.current_task_node.base_info.name_cn,
            task_id=self.current_task_node.task_id,
            task_brief=self.current_task_node.task_brief,
            assistant_guidance=self.current_task_node.assistant_guidance,
            updated_at=self.current_task_node.base_info.update_time(),
        )

    # ========= 对话方法 =========
    def _add_user_str_message(self, msg: str) -> ChatContent:
        message: str = self._agent_card.agent_memory_prompt.get("USER_STR", "{user_msg}").format(user_msg=msg)
        content: UserMessageType = UserMessageType.text_param(text=message)
        text_content: ChatContent = ChatContent(
            index=self.auto_index,
            content=content,
        )
        self.current_task_node.contexts.append(text_content)
        self._agent_card.display_deque.append(
            InteractionPackage(
                display_widget="left_log",
                content_type="UserMessageType",
                agent_id=self._agent_card.agent_id,
                content=text_content.rich_table(),
            )
        )
        return text_content

    def _add_robot_img_message(
        self,
        img_frame_id: int,
        base64_str: str,
        img_type: Literal["jpeg", "png", "jpg"] = "jpeg",
        detail: Literal["auto", "low", "high"] = "auto",
    ) -> ChatContent:
        content: RobotImgMessageType = RobotImgMessageType.image_param(
            img_frame_id=img_frame_id,
            text=self._agent_card.agent_memory_prompt.get("IMG_STR", "该图片为第{frame_id}帧").format(
                frame_id=img_frame_id
            ),
            img_type=img_type,
            base64_str=base64_str,
            detail=detail,
        )
        text_content: ChatContent = ChatContent(
            index=self.auto_index,
            content=content,
        )
        self.current_task_node.contexts.append(text_content)
        self._agent_card.display_deque.append(
            InteractionPackage(
                display_widget="left_log",
                content_type="RobotImgMessageType",
                agent_id=self._agent_card.agent_id,
                content=text_content.rich_table(),
            )
        )
        return text_content

    def _add_agent_message_type(self, msg: AssistantMessageType) -> ChatContent:
        text_content: ChatContent = ChatContent(
            index=self.auto_index,
            content=msg,
        )
        self.current_task_node.contexts.append(text_content)
        self._agent_card.display_deque.append(
            InteractionPackage(
                display_widget="left_log",
                content_type="AssistantMessageType",
                agent_id=self._agent_card.agent_id,
                content=text_content.rich_table(),
            )
        )
        return text_content

    def _add_robot_call_back_text_message(self, msg: TextParam, tool_call_id: str) -> ChatContent:
        content: ToolMessageType = ToolMessageType(content=msg, tool_call_id=tool_call_id)
        text_content: ChatContent = ChatContent(
            index=self.auto_index,
            content=content,
        )
        self.current_task_node.contexts.append(text_content)
        self._agent_card.display_deque.append(
            InteractionPackage(
                display_widget="left_log",
                content_type="ToolMessageType",
                agent_id=self._agent_card.agent_id,
                content=text_content.rich_table(),
            )
        )
        return text_content

    # ========= 树状结构方法 =========
    def _generate_tree_lines(
        self, node: RuntimeMemoryNode, include_description: bool, is_last: bool = True
    ) -> list[str]:
        if node is None or node.key is None:
            return []
        # 安全提取 Prompt 相关字段
        base_info = getattr(node.key, "base_info", None)
        if base_info:
            prefix = getattr(base_info, "prefix", "")
            prefix = prefix.replace("#", ">")
            emoji = getattr(base_info, "emoji", "❓")
            name_cn = getattr(base_info, "name_cn", "未知")
            if include_description:
                describe_cn = getattr(base_info, "describe_cn", "")
            else:
                describe_cn = ""
            line = f"{prefix} [{emoji} {name_cn}] {describe_cn}"
        else:
            line = "[❓ 未知] 无描述"
        lines = [line]
        if node.children:
            # 构造下一层前缀
            for i, child in enumerate(node.children):
                is_child_last = i == len(node.children) - 1
                lines.extend(self._generate_tree_lines(child, include_description, is_child_last))

        return lines

    def get_simple_tree(self) -> str:
        lines: list[str] = self._generate_tree_lines(self._root, include_description=False)
        return "\n".join(lines)

    def get_full_tree(self) -> str:
        lines: list[str] = self._generate_tree_lines(self._root, include_description=True)
        return "\n".join(lines)
