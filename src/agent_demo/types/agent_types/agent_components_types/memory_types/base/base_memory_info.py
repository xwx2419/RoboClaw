from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ========== 顶层结构 ==========
# ORMCPRuntimeMemory（运行时记忆区）
# │── SelfKnowledgeArea（自我认知区，系统提示词）
#     └── SysContent（唯一不变的系统上下文）
# │── LongTermMemoryArea（长期记忆区）
# │   ├── ServerRegistryBlock（服务注册块）
#         └── SysContent（以用户消息形式存在的动态系统上下文）
# │   ├── KnowledgeGraphCachingBlock（知识图谱缓存块）
#         └── SysContent（以用户消息形式存在的动态系统上下文）
# │   └── TaskTemplateBlock（任务模板块）
#         └── SysContent（以用户消息形式存在的动态系统上下文）
# ├── ShortTermMemoryArea（短期记忆区）
# │   └── TaskSessionBlock（任务会话块，记录最近几个任务的上下文）
# │       ├── RobotTaskNode（按任务拆分的聊天节点）
# │       └── RobotTaskNode（按任务拆分的聊天节点）
# │           ├── SysContent（以用户消息形式存在的动态系统上下文）
# │           ├── CompressContent（已经被压缩的上下文，对模型不可见）
# │           └── ChatContext（聊天上下文）
class MemoryName:
    ORMCP_VERSION = "v0.1"
    RUNTIME_MEMORY = "runtime_memory"
    SELF_KNOWLEDGE = "self_knowledge_area"
    LONG_TERM_MEMORY = "long_term_memory_area"
    SERVER_REGISTRY = "service_registry_block"
    KNOWLEDGE_GRAPH_CACHING = "knowledge_graph_caching_block"
    TASK_TEMPLATE = "task_template_block"
    SHORT_TERM_MEMORY = "short_term_memory_area"
    TASK_SESSION = "task_session_block"
    TASK_NODE = "task_node"
    SYS_CONTENT = "sys_content"
    COMPRESS_CONTENT = "compress_content"
    CHAT_CONTENT = "chat_content"


class CompressPolicy:
    DISCARD_OLDEST = "discard_oldest"
    COMPRESS_ALL = "compress_all"


class MemoryBaseinfo(BaseModel):
    created_at: str = Field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    updated_at: str = Field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    ormcp_version: str = Field(default=MemoryName.ORMCP_VERSION)
    prefix: str
    emoji: str
    name_cn: str
    describe_cn: str
    name_en: str
    describe_en: Optional[str] = Field(default=None)

    def update_time(self) -> str:
        self.updated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        return self.updated_at


BASE_MEMORY_INFO: dict[str, MemoryBaseinfo] = {
    MemoryName.RUNTIME_MEMORY: MemoryBaseinfo(
        prefix="",
        emoji="🤖",
        name_cn="机器人记忆区",
        describe_cn="存储机器人运行时的上下文信息",
        name_en=MemoryName.RUNTIME_MEMORY,
    ),
    MemoryName.SELF_KNOWLEDGE: MemoryBaseinfo(
        prefix="#",
        emoji="🧠",
        name_cn="自我认知区",
        describe_cn="系统级提示词，用于设定语言模型角色",
        name_en=MemoryName.SELF_KNOWLEDGE,
    ),
    MemoryName.LONG_TERM_MEMORY: MemoryBaseinfo(
        prefix="#",
        emoji="🗃️",
        name_cn="长期记忆区",
        describe_cn="存储长期记忆信息，包括知识图谱、任务模板等",
        name_en=MemoryName.LONG_TERM_MEMORY,
    ),
    MemoryName.SERVER_REGISTRY: MemoryBaseinfo(
        prefix="##",
        emoji="🛠️",
        name_cn="服务注册块",
        describe_cn="注册与管理可调用的机器人服务和服务接口",
        name_en=MemoryName.SERVER_REGISTRY,
    ),
    MemoryName.KNOWLEDGE_GRAPH_CACHING: MemoryBaseinfo(
        prefix="##",
        emoji="📍",
        name_cn="知识图谱缓存块",
        describe_cn="缓存最近被调用的知识图谱的信息，提高查询效率",
        name_en=MemoryName.KNOWLEDGE_GRAPH_CACHING,
    ),
    MemoryName.TASK_TEMPLATE: MemoryBaseinfo(
        prefix="##",
        emoji="📋",
        name_cn="任务模板块",
        describe_cn="存储用户提供的任务模板，用于快速构建与迁移复杂任务",
        name_en=MemoryName.TASK_TEMPLATE,
    ),
    MemoryName.SHORT_TERM_MEMORY: MemoryBaseinfo(
        prefix="#",
        emoji="🕒",
        name_cn="短期记忆区",
        describe_cn="存储当前对话上下文和任务状态",
        name_en=MemoryName.SHORT_TERM_MEMORY,
    ),
    MemoryName.TASK_SESSION: MemoryBaseinfo(
        prefix="##",
        emoji="💬",
        name_cn="任务会话块",
        describe_cn="记录最近3~5个任务节点，每个节点里存储当前聊天的上下文",
        name_en=MemoryName.TASK_SESSION,
    ),
    MemoryName.TASK_NODE: MemoryBaseinfo(
        prefix="###",
        emoji="🧾",
        name_cn="任务节点",
        describe_cn="按任务拆分的聊天节点，支持多轮对话任务切换",
        name_en=MemoryName.TASK_NODE,
    ),
    MemoryName.SYS_CONTENT: MemoryBaseinfo(
        prefix="",
        emoji="🫡",
        name_cn="系统上下文",
        describe_cn="当前任务的系统上下文，包括自我认知、服务注册、任务模板等信息",
        name_en=MemoryName.SYS_CONTENT,
    ),
    MemoryName.COMPRESS_CONTENT: MemoryBaseinfo(
        prefix="",
        emoji="🧐",
        name_cn="压缩上下文",
        describe_cn="已经被压缩的上下文，对模型不可见，用户可以自己选择压缩策略去处理这些上下文",
        name_en=MemoryName.COMPRESS_CONTENT,
    ),
    MemoryName.CHAT_CONTENT: MemoryBaseinfo(
        prefix="",
        emoji="🤓",
        name_cn="聊天上下文",
        describe_cn="当前聊天的多模态上下文",
        name_en=MemoryName.CHAT_CONTENT,
    ),
}
