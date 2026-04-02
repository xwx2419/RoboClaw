from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from agent_demo.common.project_env import load_feishu_app_config
from agent_demo.interaction_layer.feishu_bot import (
    FeishuBotController,
    FeishuClient,
    extract_reply_target,
    normalize_incoming_feishu_text,
    should_handle_incoming_feishu_message,
)

logger = logging.getLogger(__name__)


async def _await_maybe(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _safe_json_loads(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_log_level(lark: Any, level_name: str) -> Any:
    return getattr(lark.LogLevel, level_name.upper(), lark.LogLevel.INFO)


def should_enable_long_connection() -> bool:
    settings = load_feishu_app_config()
    mode = settings.event_receiver
    return mode in {"", "long", "long_conn", "long_connection", "ws", "websocket"}


@dataclass(frozen=True)
class FeishuLongConnectionSettings:
    app_id: str
    app_secret: str
    api_base: str
    verification_token: str
    sdk_log_level: str

    @classmethod
    def from_env(cls) -> "FeishuLongConnectionSettings":
        config = load_feishu_app_config()
        return cls(
            app_id=config.app_id,
            app_secret=config.app_secret,
            api_base=config.api_base,
            verification_token=config.verification_token,
            sdk_log_level=config.sdk_log_level,
        )


class FeishuLongConnectionRuntime:
    def __init__(
        self,
        *,
        ws_client: Any,
        feishu_client: FeishuClient,
        worker: threading.Thread | None = None,
        task: asyncio.Task | None = None,
        disconnect_coro: Callable[[], Awaitable[Any]] | None = None,
    ):
        self._ws_client = ws_client
        self._worker = worker
        self._task = task
        self._feishu_client = feishu_client
        self._disconnect_coro = disconnect_coro

    @property
    def is_alive(self) -> bool:
        if self._task is not None:
            return not self._task.done()
        if self._worker is not None:
            return self._worker.is_alive()
        return False

    async def aclose(self) -> None:
        stop = getattr(self._ws_client, "stop", None)
        if callable(stop):
            try:
                if inspect.iscoroutinefunction(stop):
                    await stop()
                else:
                    stop_result = await asyncio.to_thread(stop)
                    await _await_maybe(stop_result)
            except Exception:
                logger.warning("[FeishuLongConnection] Failed to stop websocket client cleanly.", exc_info=True)
        elif self._disconnect_coro is not None:
            try:
                await self._disconnect_coro()
            except Exception:
                logger.warning("[FeishuLongConnection] Failed to disconnect websocket client cleanly.", exc_info=True)
        if self._task is not None and not self._task.done():
            self._task.cancel()
        if self._task is not None and not self._task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=5.0)
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning("[FeishuLongConnection] Long connection task did not stop cleanly.", exc_info=True)
        await self._feishu_client.aclose()


def start_feishu_long_connection(
    *,
    session_inst: Any,
    ui_state_machine: Any,
    user_input_processing: asyncio.Event,
    chat_history: list,
    format_bot_msg: Callable[[str], str] | None = None,
    build_status_text: Callable[[], str] | None = None,
    get_image_bytes: Callable[[], Awaitable[bytes | None]] | None = None,
    source_name: str = "app",
) -> FeishuLongConnectionRuntime | None:
    if not should_enable_long_connection():
        return None

    settings = FeishuLongConnectionSettings.from_env()
    if not settings.app_id or not settings.app_secret:
        logger.warning(
            "[FeishuLongConnection] Skipping %s startup because FEISHU_APP_ID/FEISHU_APP_SECRET are missing.",
            source_name,
        )
        return None

    try:
        import lark_oapi as lark
    except ImportError:
        logger.warning(
            "[FeishuLongConnection] lark_oapi is not installed; long connection is disabled for %s.",
            source_name,
        )
        return None

    loop = asyncio.get_running_loop()
    feishu_client = FeishuClient(
        app_id=settings.app_id,
        app_secret=settings.app_secret,
        api_base=settings.api_base,
    )
    controller = FeishuBotController(
        session_inst=session_inst,
        ui_state_machine=ui_state_machine,
        user_input_processing=user_input_processing,
        chat_history=chat_history,
        format_bot_msg=format_bot_msg,
        build_status_text=build_status_text,
        get_image_bytes=get_image_bytes,
        feishu_client=feishu_client,
    )

    def _submit(coro: Awaitable[None], description: str) -> None:
        future = asyncio.run_coroutine_threadsafe(coro, loop)

        def _log_result(done: Any) -> None:
            try:
                done.result()
            except Exception:
                logger.exception("[FeishuLongConnection] Async %s failed.", description)

        future.add_done_callback(_log_result)

    def _handle_message_event(data: Any) -> None:
        try:
            payload = json.loads(lark.JSON.marshal(data))
            header = payload.get("header") or {}
            event = payload.get("event") or {}
            message = event.get("message") or {}
            reply_target = extract_reply_target(event=event, message=message)
            event_id = (header.get("event_id") or message.get("message_id") or "").strip()
            msg_type = (message.get("message_type") or "").strip().lower()

            if not reply_target.primary_id:
                return

            if msg_type != "text":
                _submit(
                    feishu_client.send_text_to_target(target=reply_target, text="目前仅支持 text 消息指令。"),
                    "non-text reply",
                )
                return

            content = _safe_json_loads(message.get("content") or "")
            raw_text = (content.get("text") or "").strip()
            if not should_handle_incoming_feishu_message(message=message, raw_text=raw_text):
                logger.info(
                    "[FeishuLongConnection] Ignore group message without mention. chat_id=%s message_id=%s",
                    reply_target.chat_id,
                    message.get("message_id"),
                )
                return

            text = normalize_incoming_feishu_text(raw_text=raw_text)
            if not text and raw_text:
                text = "help"
            if not text:
                return

            _submit(
                controller.handle_event(event_id=event_id, reply_target=reply_target, text=text),
                "message handling",
            )
        except Exception:
            logger.exception("[FeishuLongConnection] Failed to process inbound event payload.")

    event_handler = (
        lark.EventDispatcherHandler.builder(
            "",
            settings.verification_token,
            _resolve_log_level(lark, settings.sdk_log_level),
        )
        .register_p2_im_message_receive_v1(_handle_message_event)
        .build()
    )

    def _build_ws_client() -> Any:
        return lark.ws.Client(
            settings.app_id,
            settings.app_secret,
            event_handler=event_handler,
            log_level=_resolve_log_level(lark, settings.sdk_log_level),
        )

    start_method = getattr(lark.ws.Client, "start", None)
    has_legacy_async_primitives = callable(start_method) and not inspect.iscoroutinefunction(start_method)
    has_legacy_async_primitives = has_legacy_async_primitives and hasattr(lark.ws.Client, "_connect")
    has_legacy_async_primitives = has_legacy_async_primitives and hasattr(lark.ws.Client, "_disconnect")

    if has_legacy_async_primitives:
        from lark_oapi.ws import client as lark_ws_client_module

        lark_ws_client_module.loop = loop
        ws_client = _build_ws_client()

        async def _run_client_legacy() -> None:
            logger.info("[FeishuLongConnection] Starting long connection for %s.", source_name)
            try:
                await ws_client._connect()
                loop.create_task(ws_client._ping_loop())
                await asyncio.Future()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[FeishuLongConnection] Long connection stopped unexpectedly for %s.", source_name)

        task = asyncio.create_task(_run_client_legacy(), name=f"feishu-long-connection-{source_name}")
        return FeishuLongConnectionRuntime(
            ws_client=ws_client,
            task=task,
            feishu_client=feishu_client,
            disconnect_coro=ws_client._disconnect,
        )

    async def _run_client_async() -> None:
        logger.info("[FeishuLongConnection] Starting long connection for %s.", source_name)
        try:
            ws_client = _build_ws_client()
            await ws_client.start()
        except Exception:
            logger.exception("[FeishuLongConnection] Long connection stopped unexpectedly for %s.", source_name)

    def _run_client_sync() -> None:
        logger.info("[FeishuLongConnection] Starting long connection for %s.", source_name)
        try:
            ws_client = _build_ws_client()
            start_result = ws_client.start()
            if inspect.isawaitable(start_result):
                asyncio.run(_await_maybe(start_result))
        except Exception:
            logger.exception("[FeishuLongConnection] Long connection stopped unexpectedly for %s.", source_name)

    if inspect.iscoroutinefunction(start_method):
        ws_client = _build_ws_client()

        async def _run_client_async_bound() -> None:
            logger.info("[FeishuLongConnection] Starting long connection for %s.", source_name)
            try:
                await ws_client.start()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[FeishuLongConnection] Long connection stopped unexpectedly for %s.", source_name)

        task = asyncio.create_task(_run_client_async_bound(), name=f"feishu-long-connection-{source_name}")
        return FeishuLongConnectionRuntime(ws_client=ws_client, task=task, feishu_client=feishu_client)

    worker = threading.Thread(
        target=_run_client_sync,
        name=f"feishu-long-connection-{source_name}",
        daemon=True,
    )
    worker.start()
    return FeishuLongConnectionRuntime(ws_client=_build_ws_client(), worker=worker, feishu_client=feishu_client)
