import sys
import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
import httpx
from httpx._models import Response
import logging

logger = logging.getLogger(name=__name__)

app = Server("gma_mcp_server")

gma_base_url = "http://localhost:"
gma_port = "8764"


@app.call_tool()
async def fetch_tool(name: str, arguments: dict[str, str]) -> list[types.TextContent]:
    result: list[types.TextContent] = []
    if name == "start_task":
        await start_task(result)
    elif name == "stop_task":
        await stop_task(result)
    else:
        result.append(types.TextContent(type="text", text="非法的工具名请求"))
    return result


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="start_task",
            description="启动DP模型，使其开始运行（当前模型只有抓取能力，即调用本接口可以启动模型进行物品抓取）",
            inputSchema={
                "type": "object",
                "required": [],
                "properties": {},
            },
        ),
        types.Tool(
            name="stop_task",
            description="停止DP模型，使其停止运行（当前模型只有抓取能力，即调用本接口可以停止模型进行物品抓取）",
            inputSchema={
                "type": "object",
                "required": [],
                "properties": {},
            },
        ),
    ]


### 工具实现 ###
async def start_task(result: list[types.TextContent]):
    """启动任务"""
    params = [{"target": 0.3, "object": 1, "st": 0, "end": 30, "everyn": 1}]
    await send_request_to_gma(result, "start_task", params)
    return None


async def stop_task(result: list[types.TextContent]):
    """停止任务"""
    await send_request_to_gma(result, "stop_task")
    return None


# 给GMA发http请求
async def send_request_to_gma(result: list[types.TextContent], command: str, params: list | None = None):
    url = gma_base_url + gma_port + "/control_robot"
    headers = {"Content-Type": "application/json"}
    request_data: dict = {"command": command}
    if params:
        request_data["params"] = params

    timeout = httpx.Timeout(timeout=0.1)  # 非常短的超时
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response_data: Response = await client.post(url, headers=headers, json=request_data)
            result.append(types.TextContent(type="text", text=response_data.json()))
        except (httpx.RequestError, httpx.ReadTimeout):
            logger.info("Fire-and-forget: 请求发送但不等待响应。超时可忽略。")
            result.append(types.TextContent(type="text", text="因为未知原因，请求已超时。"))
    return result


def main() -> int:
    async def arun():
        async with stdio_server() as streams:
            await app.run(streams[0], streams[1], app.create_initialization_options())

    anyio.run(arun)

    return 0


sys.exit(main())  # type: ignore[call-arg]
