import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class UdpReceiver:
    def __init__(self, config: dict) -> None:
        self._config = config
        self._host = self._config["host"]
        self._port = self._config["port"]
        logger.info(f"[UdpReceiver] receiver host:{self._host}, port:{self._port}")
        self._transport = None
        self._data_queue = asyncio.Queue()  # 使用异步队列

    async def start(self):
        """初始化UDP接收器"""
        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: self._UdpProtocol(self._data_queue), local_addr=(self._host, self._port)
        )
        logger.info(f"[UdpReceiver] receiver start: host:{self._host}, port:{self._port}")

    async def shutdown(self):
        """关闭UDP接收器"""
        if self._transport:
            self._transport.close()
            self._transport = None
            self._data_queue.put_nowait(None)
            logger.info("[UdpReceiver] UDP receiver shutdown")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    async def get_message(self, latest=False) -> Any | None:
        """从队列中获取消息"""
        if not latest:
            return await self._data_queue.get()

        # 获取最新数据并清空队列
        result = None
        while not self._data_queue.empty():
            result = await self._data_queue.get_nowait()
        return result

    class _UdpProtocol(asyncio.DatagramProtocol):
        """内部协议类，用于处理UDP数据包"""

        _ctl_data = None
        _logging_interval = 500
        _index = 0

        def __init__(self, data_queue):
            self._data_queue = data_queue  # 接收队列

        def datagram_received(self, data, addr):
            """接收到UDP数据包时的回调"""
            if self._index % self._logging_interval == 0:
                logger.info(f"[UdpReceiver] Received len:{len(data)} from addr:{addr}")
                self._index = 0
            self._index += 1

            # 将数据放入队列
            self._data_queue.put_nowait(data[:])  # 使用浅拷贝避免数据被修改

        def error_received(self, exc):
            logger.error(f"[UdpReceiver] error received: {exc}")
