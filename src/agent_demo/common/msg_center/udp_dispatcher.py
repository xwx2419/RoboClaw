import asyncio
import logging

logger = logging.getLogger(__name__)


class UdpDispatcher:
    def __init__(self, config: dict):
        self._config = config
        self._host = self._config["host"]
        self._port = self._config["port"]
        logger.info(f"[UdpDispatcher] dispatch host:{self._host}, port:{self._port}")
        self._index = 0
        self._log_interval = 500
        self._transport: asyncio.DatagramTransport | None = None
        self._count = 0
        self._last_log_time = asyncio.get_event_loop().time()
        self._log_frequency = 1.0

    async def start(self):
        if self._transport is None:
            loop = asyncio.get_running_loop()
            self._transport, _ = await loop.create_datagram_endpoint(
                lambda: asyncio.DatagramProtocol(), remote_addr=(self._host, self._port)
            )
            logger.info(f"[UdpDispatcher] dispatch started: host:{self._host}, port:{self._port}")

    async def shutdown(self):
        logger.info("[UdpDispatcher] dispatch shutdown")
        if self._transport:
            self._transport.close()
            self._transport = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    async def dispatch(self, data):
        if self._transport is None:
            logger.error("[UdpDispatcher] transport is None")
            raise RuntimeError("UDP transport is not initialized")

        self._index += 1
        self._count += 1

        current_time = asyncio.get_event_loop().time()
        time_diff = current_time - self._last_log_time

        if time_diff >= self._log_frequency:
            logger.debug(f"[UdpDispatcher] datachannel receive freq: {self._count / time_diff:.2f} Hz")
            self._count = 0
            self._last_log_time = current_time

        if self._index % self._log_interval == 0:
            logger.info(f"[UdpDispatcher] deliver: {len(data)}")
            self._index = 0
        try:
            if not isinstance(data, bytes):
                data = bytes(data, "utf-8")
            self._transport.sendto(data)
        except Exception as e:
            logger.error(f"[UdpDispatcher] Failed to send message: {data} e: {e}")
