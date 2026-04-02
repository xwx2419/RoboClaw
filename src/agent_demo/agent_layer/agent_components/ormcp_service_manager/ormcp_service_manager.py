import logging
from agent_demo.types.interaction_types import InteractionPackage
from agent_demo.types.agent_types import ORMCPServiceConfig, ServiceRegister, TextParam, BaseAgentCard
from .ormcp_service_connection import ORMCPServiceConnection
from agent_demo.common.root_logger import table_to_str

logger = logging.getLogger(__name__)


class ORMCPServiceManager:
    def __init__(self, agent_card: BaseAgentCard):
        self._agent_card_ref: BaseAgentCard = agent_card
        self._ormcp_services_config_list: dict[str, ORMCPServiceConfig] = ORMCPServiceConfig.load_from_json(
            self._agent_card_ref.service_config_path
        )
        self._services_conn_dict: dict[str, ORMCPServiceConnection] = {}  # 管理各个服务器连接
        self._services_register_list: list[ServiceRegister] = []

    # ===== 只读属性 =====
    @property
    def activate_tools_list(self) -> list[dict]:
        return [
            tool.openai_format
            for service in self._services_register_list
            if service.is_activation
            for tool in service.tools_list
        ]

    @property
    def services_config_list(self) -> dict[str, ORMCPServiceConfig]:
        return self._ormcp_services_config_list

    @property
    def services_conn_dict(self) -> dict[str, ORMCPServiceConnection]:
        return self._services_conn_dict

    # ===== 核心方法 =====
    async def init_services(self):
        await self.show_services_config_as_table()
        await self.init_mcp_services()
        await self.init_mcp_register_list()

    async def init_mcp_services(self) -> None:
        for service_name, config in self._ormcp_services_config_list.items():
            if service_name not in self._services_conn_dict:
                conn = ORMCPServiceConnection(
                    service_name=service_name,
                    config=config,
                    display_deque=self._agent_card_ref.display_deque,
                )
                try:
                    await conn.init_stdio_client()
                    self._services_conn_dict[service_name] = conn
                except Exception as e:
                    logger.error(f"[Init][ORMCPservices] Failed to initialize service {service_name}: {e}")
        logger.debug("[Init][ORMCPservices][Done]")

    async def init_mcp_register_list(self) -> None:
        for service_name, conn in self._services_conn_dict.items():
            try:
                service_activation = await conn.list_tools()
                await self.registry_service(service_activation)
            except Exception as e:
                logger.error(f"[Init][mcp_register] Failed to list tools for service {service_name}: {e}")
        logger.debug("[Init][mcp_register][Done]")

    async def registry_service(self, service: ServiceRegister) -> None:
        self._services_register_list.append(service)
        logger.info(f"[Registry][new_service] ⚙️  <--- {service.service_name}")
        if not self._agent_card_ref.silence:
            table = service.rich_table()
            self._agent_card_ref.display_deque.append(
                InteractionPackage(
                    content_type="service_register",
                    agent_id=self._agent_card_ref.agent_id,
                    content=table,
                )
            )
            logger.debug(table_to_str(table))

    async def tools_routing(self, service_name: str, tool_name: str, tool_args: dict) -> TextParam:
        res: TextParam | None = await self._services_conn_dict[service_name].execute_tool(
            tool_name=tool_name, arguments=tool_args
        )
        if res is None:
            res = TextParam(text="unknown error")
        return res

    def check_is_agent_service(self, service_name: str) -> bool:  # 检查service_name是内部服务还是外部服务？
        for service in self._services_register_list:
            if service.service_name == service_name:
                return service.is_agent_service
        raise RuntimeError("could not find match service in services_register_list.")

    # ========= 重置方法 =========
    async def reset(self):
        pass

    # ===== 打印方法 =====
    async def show_services_config_as_table(self) -> None:
        if self._agent_card_ref.silence:
            return
        for service_name, config in self._ormcp_services_config_list.items():
            table = config.rich_table()
            table.title = f"{service_name}"  # 设置每个子表标题
            self._agent_card_ref.display_deque.append(
                InteractionPackage(
                    content_type="services_config",
                    agent_id=self._agent_card_ref.agent_id,
                    content=table,
                )
            )
            logger.debug(table_to_str(table))

    async def show_service_register_as_table(self) -> None:
        if self._agent_card_ref.silence:
            return
        for service in self._services_register_list:
            table = service.rich_table()
            self._agent_card_ref.display_deque.append(
                InteractionPackage(
                    content_type="service_register",
                    agent_id=self._agent_card_ref.agent_id,
                    content=table,
                )
            )
            logger.debug(table_to_str(table))

    # ========== 优雅退出 ==========
    async def shutdown(self) -> None:
        for service_name, conn in self._services_conn_dict.items():
            await conn.shutdown()
            logger.info(f"[Shutdown][{service_name}][Done]")
        self._services_conn_dict.clear()

    async def terminate(self) -> None:
        await self.shutdown()
