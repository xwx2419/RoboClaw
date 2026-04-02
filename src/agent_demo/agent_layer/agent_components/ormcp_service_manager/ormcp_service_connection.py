import logging
from agent_demo.types.agent_types import ORMCPServiceConfig, ServiceRegister, ORMCPTool, FunctionDef, TextParam
from mcp import ClientSession
import asyncio
from contextlib import AsyncExitStack
from mcp.client.stdio import stdio_client
from mcp.types import ListToolsResult, CallToolResult, TextContent, ImageContent, EmbeddedResource
from collections import deque
from agent_demo.types.interaction_types import InteractionPackage

logger = logging.getLogger(__name__)


class ORMCPServiceConnection:
    def __init__(
        self,
        service_name: str,
        config: ORMCPServiceConfig,
        display_deque: deque[InteractionPackage],
    ):
        self._display_deque: deque[InteractionPackage] = display_deque
        self._config: ORMCPServiceConfig = config
        self._session: ClientSession = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._service_register_table: ServiceRegister = ServiceRegister(
            service_name=service_name,
            description=config.description_cn,
            is_activation=config.need_activation,
            is_agent_service=False,
        )
        self._closed: bool = False

    # ===== 只读属性 =====
    @property
    def service_name(self) -> str:
        return self._service_register_table.service_name

    @property
    def description(self) -> str:
        return self._service_register_table.description

    @property
    def config(self) -> ORMCPServiceConfig:
        return self._config

    @property
    def session(self) -> ClientSession:
        return self._session

    @property
    def service_register_table(self) -> ServiceRegister:
        return self._service_register_table

    # ===== async with 支持 =====
    async def __aenter__(self):
        """
        支持 async with, 用于自动初始化客户端连接。
        """
        await self.init_stdio_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        支持 async with, 退出时自动清理资源。
        """
        await self.close()

    # ========== 优雅退出 ==========
    async def shutdown(self) -> None:
        await self.close()

    async def terminate(self) -> None:
        await self.shutdown()

    # ===== close =====
    async def close(self) -> None:
        """close Connection ,Clean up service resources."""
        if self._closed:
            return
        self._closed = True
        try:
            await self._exit_stack.aclose()
            self._session = None
        except asyncio.CancelledError:
            logging.warning(f"{self.service_name} Close cancelled, ignoring to allow graceful shutdown.")
        except Exception as e:
            logging.error(f"Error during cleanup of service {self.service_name}: {e}")

    # ===== 初始化 =====
    async def init_stdio_client(self) -> None:
        try:
            read, write = await self._init_stdio_client_inst()
            self._session = await self._init_client_session(read, write)
        except Exception as e:
            logging.error(f"Error initializing service {self.service_name}: {e}")
            await self.close()
            raise

    async def _init_stdio_client_inst(self) -> tuple:
        service_params = self._config.to_stdio_service_parameters()
        return await self._exit_stack.enter_async_context(stdio_client(service_params))

    async def _init_client_session(self, read, write) -> ClientSession:
        session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    # ===== 核心方法 =====
    async def list_tools(self) -> ServiceRegister:
        """
        列出所有可用的工具，并返回工具列表。
        """
        if not self._session:
            raise RuntimeError(f"service {self.service_name} not initialized")

        try:
            logger.debug(f"[{self.service_name}] Listing tools...")
            tools_response: ListToolsResult = await self._session.list_tools()
            logger.debug(f"[{self.service_name}] Tools listed successfully.")
            _tools = tools_response.tools
            for tool in _tools:
                m_tool = ORMCPTool(
                    service_name=self.service_name,
                    func_definition=FunctionDef(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.inputSchema,
                    ),
                )
                self._service_register_table.tools_list.append(m_tool)
        except Exception as e:
            logger.error(f"[{self.service_name}] Error listing tools: {e}", exc_info=True)
            raise

        return self._service_register_table

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict,
        retries: int = 3,
        delay: float = 1.0,
        timeout: float = 10.0,
        raise_on_failure: bool = False,
    ) -> TextParam | None:
        if not self._session:
            raise RuntimeError(f"service {self.service_name} not initialized")

        res: TextParam = TextParam(text="")
        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result: CallToolResult = await self.session.call_tool(tool_name, arguments)

                # 检查返回结果是否为错误
                if result.isError:
                    if raise_on_failure:
                        raise ValueError(f"Error executing tool {tool_name}: {result.content}")
                    else:
                        logging.error(f"Error executing tool {tool_name}: {result.content}")
                        res.text = f"Error executing tool {tool_name}: {result.content}"
                        return res

                # 检查返回结构类型是否为期望的类型
                if isinstance(result.content, list):
                    list_res: str = ""
                    for index, item in enumerate(result.content):
                        if isinstance(item, TextContent):
                            text_len = len(item.text)
                            text = item.text
                            if text_len > TextParam.max_text_len:
                                logger.warning(f"工具返回内容过长({text_len} 字符,已截断")
                                list_res = f"结果{index}:" + text[: TextParam.max_text_len] + " ...(Truncated)\n"
                            else:
                                list_res = f"结果{index}:" + text + "\n"
                            res.text += list_res
                        elif isinstance(item, ImageContent):
                            list_res = f"结果{index}:" + f"不支持的工具返回类型 {type(item)}: {tool_name}" + "\n"
                        elif isinstance(item, EmbeddedResource):
                            list_res = f"结果{index}:" + f"不支持的工具返回类型 {type(item)}: {tool_name}" + "\n"
                        else:
                            list_res = f"结果{index}:" + f"不支持的工具返回类型 ImageContent,tool: {tool_name}" + "\n"
                    return res
            except asyncio.CancelledError:
                raise
            except Exception as e:
                attempt += 1
                logging.warning(f"Error executing tool: {e}. Attempt {attempt} of {retries}.")
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise
