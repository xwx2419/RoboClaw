from agent_demo.types.agent_types import ChatAPIConfig, BaseAgentCard, TextParam
from agent_demo.agent_layer.agent_components.agent_tools.local_skill_registry import LocalSkillRegistry
from agent_demo.agent_layer.agent_core import ActAgent
from agent_demo.common.msg_center import UdpDispatcher, UdpReceiver
from agent_demo.agent_layer.agent_prompt import ActAgentPrompt
from agent_demo.interaction_layer.local_skill_support import PreparedAgentMessage, prepare_agent_message

# from agent_demo.machine_layer.dataloader_a2d import DataLoaderA2D
import signal
import anyio
import aioconsole
import asyncio
from pathlib import Path
from agent_demo.common.root_logger import setup_root_logging
import traceback
import logging

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[4]

setup_root_logging(default_log_path="./applog/")


class UdpData:
    def __init__(
        self,
        send_config: dict,
        receive_config: dict,
        receive_cqueue: asyncio.Queue[str],
    ):
        self._dispatcher = UdpDispatcher(send_config)
        self._receiver = UdpReceiver(receive_config)
        self._receive_cqueue: asyncio.Queue[str] = receive_cqueue
        self._sequence_number = 0
        self.running = False

    async def start(self):
        await self._receiver.start()
        await self._dispatcher.start()
        self.running = True
        logger.info("udp started.")

    async def stop(self):
        await self._receiver.shutdown()
        await self._dispatcher.shutdown()
        logger.info("udp stopped.")

    async def send(self, send_msg: str):
        self._sequence_number += 1
        message = send_msg.encode('utf-8')
        logger.info(f"Sending: {message.decode()}")
        await self._dispatcher.dispatch(message)

    async def receive(self):
        data = await self._receiver.get_message()
        if data is None:
            return
        message = data.decode('utf-8')
        await self._receive_cqueue.put(message)
        logger.info(f"Received: {message}")

    async def run_forever(self):
        await self.start()
        try:
            while self.running:
                await anyio.sleep(0.1)
                await self.receive()
        except asyncio.CancelledError:
            logger.info("udp task cancelled.")
        finally:
            await self.stop()


class Session:
    def __init__(self):
        self._agent_card: BaseAgentCard = BaseAgentCard(
            silence=False,
            config=ChatAPIConfig.resolve_runtime_default(),
            service_config_path=str(REPO_ROOT / "src/agent_demo/config/ormcp_services.json"),
            skill_paths=[str(REPO_ROOT / "skills")],
            agent_memory_prompt=ActAgentPrompt.init_memory_prompt,
            # robot_dataloader=DataLoaderA2D(),
        )
        self.skill_registry = LocalSkillRegistry(
            configured_paths=self._agent_card.skill_paths,
            workspace_root=str(REPO_ROOT),
        )
        self.agent = ActAgent(agent_card=self._agent_card)
        self.running = False  # 记录事件循环是否正在运行

    def prepare_message_for_agent(self, message: str) -> PreparedAgentMessage:
        services = self.agent.service_manager._services_register_list if self.agent is not None else None
        return prepare_agent_message(message, self.skill_registry, services=services)

    # async def tts_inference(self, content: str) -> None:
    #     url = "http://127.0.0.1:8000/tts_inference"
    #     headers = {"Content-Type": "application/json"}
    #     data = {"content": content}

    #     timeout = httpx.Timeout(connect=1.0, read=0.1, write=1.0, pool=1.0)  # 非常短的超时

    #     async with httpx.AsyncClient(timeout=timeout) as client:
    #         try:
    #             await client.post(url, headers=headers, json=data)
    #         except (httpx.RequestError, httpx.ReadTimeout):
    #             logger.info("Fire-and-forget: 请求发送但不等待响应。超时可忽略。")

    async def init_session(self):
        await self.agent.init_agent()
        self.running = True

    async def run_forever(self, queue: asyncio.Queue[str], udp_inst: UdpData):
        try:
            while self.running:
                # 每次循环都等一下，给取消和其他任务机会调度
                await anyio.sleep(0.1)
                # 非阻塞地获取最新一条输入（如果没有就跳过）
                try:
                    message = queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue
                # 收到 exit 命令就退出
                if message.strip().lower() in ("exit", "quit"):
                    logger.info(f"user CMD({message}) Cancelled Session.")
                    self.running = False
                    break
                # 否则交给 run_once 处理
                prepared_message = self.prepare_message_for_agent(message)
                if prepared_message.error_message:
                    await udp_inst.send(prepared_message.error_message)
                    continue

                agent_message = prepared_message.message or message
                res: TextParam | None = await self.agent.run_once(agent_message)
                if res:
                    await udp_inst.send(res.text)
        except asyncio.CancelledError:
            logger.info("Session cancelled.")
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {type(e).__name__}: {e}")
            logger.error("Full traceback:\n" + traceback.format_exc())
        finally:
            await self.agent.shutdown()
            logger.info("[🚪] agent shutdown down")

    async def display_deque_forever(self):
        try:
            while self.running:
                # 每次循环都等一下，给取消和其他任务机会调度
                await anyio.sleep(0.1)
                if len(self._agent_card.display_deque) > 0:
                    oldest_data = self._agent_card.display_deque.popleft()  # 取出并删除最老数据
                    if oldest_data.display_widget == "left_log":
                        # logger.info(table_to_str(oldest_data.content))
                        pass
                    else:
                        # logger.info(table_to_str(oldest_data.content))
                        pass
        except asyncio.CancelledError:
            logger.info("display_deque_forever cancelled.")
        except Exception:
            logger.info("display_deque_forever cancelled.")


async def signal_watcher(cancel_scope: anyio.CancelScope):
    with anyio.open_signal_receiver(signal.SIGINT, signal.SIGTERM) as signals:
        async for sig in signals:
            logger.info(f"Received signal: {sig!s}, preparing to exit...")
            cancel_scope.cancel()
            break  # 可选，避免重复 cancel


async def input_watcher(queue: asyncio.Queue[str], agent: ActAgent):
    running = True
    _agent = agent
    try:
        while running:  # anyio.to_thread.run_sync 会自动调度到线程池，不会阻塞主事件循环
            await anyio.sleep(0.1)
            if _agent.ready_to_chat:
                logger.info(">>>>>>")
                line = await aioconsole.ainput("")
                await queue.put(line)
            else:
                await anyio.sleep(0.1)
    except asyncio.CancelledError:
        logger.info("Session cancelled.")


async def main():
    udp_config = {
        "udp": {"listen": {"host": "0.0.0.0", "port": 23333}, "send": {"host": "172.19.9.200", "port": 23334}}
    }
    queue: asyncio.Queue[str] = asyncio.Queue()
    udp_data = UdpData(udp_config["udp"]["send"], udp_config["udp"]["listen"], queue)
    session_inst: Session = Session()
    await session_inst.init_session()
    async with anyio.create_task_group() as tg:
        tg.start_soon(udp_data.run_forever)
        # 启动 Session，并给它一个 queue
        tg.start_soon(session_inst.run_forever, queue, udp_data)
        # 启动命令行输入监听
        tg.start_soon(input_watcher, queue, session_inst.agent)
        # 启动信号监听
        tg.start_soon(signal_watcher, tg.cancel_scope)
        # 等待一下，确保主要服务已经启动
        await anyio.sleep(1)
        # 循环打印 display_deque
        tg.start_soon(session_inst.display_deque_forever)

    # 任务组退出后，可以再等一会儿或做收尾
    await anyio.sleep(1)  # 再运行一段时间


if __name__ == "__main__":
    anyio.run(main)
