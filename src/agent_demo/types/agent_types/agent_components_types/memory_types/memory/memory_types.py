from pydantic import Field
from rich.table import Table
from typing import Optional
import uuid
from ..base.base_memory_types import BaseArea, BaseBlock, BaseTaskNode, BaseContent
from ..base.base_memory_info import MemoryBaseinfo, MemoryName, BASE_MEMORY_INFO
from ..base.base_context_types import (
    AssistantMessageType,
    ToolMessageType,
    UserMessageType,
    RobotImgMessageType,
    SystemMessageType,
    SystemDynamicMessageType,
)


# ========== BaseContent拓展 =========
class SysContent(BaseContent):
    base_info: MemoryBaseinfo = Field(default=BASE_MEMORY_INFO[MemoryName.SYS_CONTENT])
    content: SystemMessageType | SystemDynamicMessageType


class CompressContent(BaseContent):
    base_info: MemoryBaseinfo = Field(default=BASE_MEMORY_INFO[MemoryName.COMPRESS_CONTENT])
    content: UserMessageType | AssistantMessageType | ToolMessageType | RobotImgMessageType

    def convert_compress_to_chat(self) -> 'ChatContent':
        # 创建 CompressContent 实例
        chat_content = ChatContent(
            index=self.index,
            base_info=BASE_MEMORY_INFO[MemoryName.CHAT_CONTENT],
            content=self.content,
        )
        return chat_content


class ChatContent(BaseContent):
    base_info: MemoryBaseinfo = Field(default=BASE_MEMORY_INFO[MemoryName.CHAT_CONTENT])
    content: UserMessageType | AssistantMessageType | ToolMessageType | RobotImgMessageType

    def convert_chat_to_compress(self) -> 'CompressContent':
        # 创建 CompressContent 实例
        compress_content = CompressContent(
            index=self.index,
            base_info=BASE_MEMORY_INFO[MemoryName.COMPRESS_CONTENT],
            content=self.content,
        )
        return compress_content


# ========== 运行时记忆区描述 ==========
class RuntimeMemory(BaseArea):
    base_info: MemoryBaseinfo = Field(default=BASE_MEMORY_INFO[MemoryName.RUNTIME_MEMORY])
    Contexts: Optional[SysContent] = Field(default=None)


# ========== 自我认知区描述 ==========
class SelfKnowledgeArea(BaseArea):
    base_info: MemoryBaseinfo = Field(default=BASE_MEMORY_INFO[MemoryName.SELF_KNOWLEDGE])
    Contexts: Optional[SysContent] = Field(default=None)


# ========== 长期记忆区描述 ==========
class KnowledgeGraphCachingBlock(BaseBlock):
    base_info: MemoryBaseinfo = Field(default=BASE_MEMORY_INFO[MemoryName.KNOWLEDGE_GRAPH_CACHING])
    Contexts: Optional[SysContent] = Field(default=None)


class ServerRegistryBlock(BaseBlock):
    base_info: MemoryBaseinfo = Field(default=BASE_MEMORY_INFO[MemoryName.SERVER_REGISTRY])
    Contexts: Optional[SysContent] = Field(default=None)


class TaskTemplateBlock(BaseBlock):
    base_info: MemoryBaseinfo = Field(default=BASE_MEMORY_INFO[MemoryName.TASK_TEMPLATE])
    Contexts: Optional[SysContent] = Field(default=None)


class LongTermMemoryArea(BaseArea):
    base_info: MemoryBaseinfo = Field(default=BASE_MEMORY_INFO[MemoryName.LONG_TERM_MEMORY])


# ========== 短期记忆区描述 ==========
class TaskSessionBlock(BaseBlock):
    base_info: MemoryBaseinfo = Field(default=BASE_MEMORY_INFO[MemoryName.TASK_SESSION])
    Contexts: Optional[SysContent] = Field(default=None)


class ShortTermMemoryArea(BaseArea):
    base_info: MemoryBaseinfo = Field(default=BASE_MEMORY_INFO[MemoryName.SHORT_TERM_MEMORY])


# ========== 任务节点描述 ==========
class TaskNode(BaseTaskNode):  # 和数据库之间的映射？
    base_info: MemoryBaseinfo = Field(default=BASE_MEMORY_INFO[MemoryName.TASK_NODE])
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    user_requirements: str = Field(default="")
    task_state: str = Field(default="")
    task_summary: str = Field(default="")
    task_brief: str = Field(default="")
    assistant_guidance: str = Field(default="")
    task_sys_context: Optional[SysContent] = Field(default=None)  # 一个任务只能一一个系统级提示词
    compress_contexts: list[CompressContent] = Field(default_factory=list)  # 压缩队列
    contexts: list[ChatContent] = Field(default_factory=list)  # 正在进行的对话

    def _cleanup_orphaned_tool_messages(self):
        """
        清理所有孤立的 tool 消息（包括尾部和中间位置）。
        tool 消息必须紧跟在包含 tool_calls 的 assistant 消息之后。
        遍历整个消息序列，删除所有没有对应 assistant(tool_calls) 的 tool 消息。
        """
        if not self.contexts:
            return

        # 收集所有需要保留的消息索引
        keep_indices = []
        i = 0
        while i < len(self.contexts):
            content = self.contexts[i].content

            # 如果是 assistant 消息且包含 tool_calls，保留它及其后续的 tool 消息
            if isinstance(content, AssistantMessageType) and len(content.tool_calls) > 0:
                # 保留这个 assistant 消息
                keep_indices.append(i)
                i += 1

                # 保留紧跟在后面的所有 tool 消息（直到遇到非 tool 消息）
                while i < len(self.contexts) and isinstance(self.contexts[i].content, ToolMessageType):
                    keep_indices.append(i)
                    i += 1
            # 如果是其他类型的消息（user, assistant without tool_calls, robot_img），直接保留
            elif not isinstance(content, ToolMessageType):
                keep_indices.append(i)
                i += 1
            else:
                # 这是孤立的 tool 消息（前面没有 assistant(tool_calls)），跳过不保留
                i += 1

        # 根据保留的索引重建 contexts
        if len(keep_indices) < len(self.contexts):
            self.contexts = [self.contexts[idx] for idx in keep_indices]

    def compress_policy_compress(self, keep_last_n: int = 1):
        if len(self.contexts) <= keep_last_n:
            return

        # 计算要保留的起始位置，确保不会切断 tool_calls 和 tool 的配对
        # 从后往前找到最后一个包含 tool_calls 的 assistant 消息
        keep_start_idx = len(self.contexts) - keep_last_n
        last_assistant_with_tool_calls_idx = -1

        # 从保留区域往前查找，看是否有 assistant(tool_calls) 在保留区域之前
        for i in range(keep_start_idx - 1, -1, -1):
            content = self.contexts[i].content
            if isinstance(content, AssistantMessageType) and len(content.tool_calls) > 0:
                last_assistant_with_tool_calls_idx = i
                break

        # 如果找到了 assistant(tool_calls) 在保留区域之前，需要调整保留起始位置
        # 确保包含完整的 assistant(tool_calls) + tool 配对
        if last_assistant_with_tool_calls_idx >= 0 and last_assistant_with_tool_calls_idx < keep_start_idx:
            # 找到这个 assistant 之后的所有 tool 消息
            tool_count = 0
            for i in range(last_assistant_with_tool_calls_idx + 1, len(self.contexts)):
                if isinstance(self.contexts[i].content, ToolMessageType):
                    tool_count += 1
                else:
                    break
            # 调整保留起始位置，包含完整的配对
            keep_start_idx = last_assistant_with_tool_calls_idx

        for chat_content in self.contexts[:keep_start_idx]:
            compress_content = chat_content.convert_chat_to_compress()
            self.compress_contexts.append(compress_content)
        self.contexts = self.contexts[keep_start_idx:]

        # 清理可能残留的孤立 tool 消息
        self._cleanup_orphaned_tool_messages()

    def compress_policy_discard_oldest(self, drop_n: int = 1):
        """
        丢弃最老的 n 条对话上下文，不迁移、不压缩，仅做清除。
        确保不会留下孤立的 tool 消息。
        """
        if drop_n <= 0:
            return
        original_len = len(self.contexts)
        if original_len <= drop_n:
            self.contexts = []
            return

        # 计算删除边界
        drop_end_idx = drop_n

        # 检查删除边界是否会切断 assistant(tool_calls) 和 tool 的配对
        # 如果删除边界正好在 assistant(tool_calls) 之后、tool 之前，需要调整
        if drop_end_idx < original_len:
            # 检查删除边界前一条是否是 assistant(tool_calls)
            prev_idx = drop_end_idx - 1
            if prev_idx >= 0:
                prev_content = self.contexts[prev_idx].content
                if isinstance(prev_content, AssistantMessageType) and len(prev_content.tool_calls) > 0:
                    # 前一条是 assistant(tool_calls)，检查下一条是否是 tool
                    if drop_end_idx < original_len:
                        next_content = self.contexts[drop_end_idx].content
                        if isinstance(next_content, ToolMessageType):
                            # 会切断配对，需要调整：要么一起删除，要么一起保留
                            # 选择一起删除（因为我们要删除最老的，所以删除整个配对更安全）
                            # 继续删除，直到遇到非 tool 消息
                            while drop_end_idx < original_len and isinstance(
                                self.contexts[drop_end_idx].content, ToolMessageType
                            ):
                                drop_end_idx += 1

        # 执行删除
        self.contexts = self.contexts[drop_end_idx:]

        # 清理可能残留的孤立 tool 消息（作为安全网）
        self._cleanup_orphaned_tool_messages()

    def __str__(self) -> str:
        return (
            f"\n"
            f"[{self.base_info.emoji}{self.base_info.name_cn}] {self.base_info.describe_cn}\n"
            f"[task_id] {self.task_id}\n"
            f"[user_requirements] {self.user_requirements}\n"
            f"[task_state] {self.task_state}\n"
            f"[task_sys_context] {self.task_sys_context}\n"
            f"[compress_contexts_len] {len(self.compress_contexts)}\n"
            f"[contexts_len] {len(self.contexts)}\n"
            f"[task_brief] \n{self.task_brief}\n"
            f"[assistant_guidance] \n{self.assistant_guidance}\n"
            f"[task_summary] {self.task_summary}\n"
        )

    def rich_table(self) -> Table:
        table = Table(show_lines=False, box=None, expand=False)
        table.add_column(self.base_info.emoji, style="cyan", no_wrap=True)
        table.add_column(str(self.task_id), style="magenta")
        table.add_row("task_state", self.task_state)
        table.add_row("compress_contexts_len", str(len(self.compress_contexts)))
        table.add_row("contexts_len", str(len(self.contexts)))
        table.add_row("task_brief", self.task_brief)
        table.add_row("assistant_guidance", self.assistant_guidance)
        table.add_row("task_summary", self.task_summary)
        table.add_row("user_requirements", self.user_requirements)
        return table
