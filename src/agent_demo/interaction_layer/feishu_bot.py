from __future__ import annotations

import asyncio
import importlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

import httpx
from agent_demo.common.assistant_output import resolve_final_response_text

logger = logging.getLogger(__name__)
_FEISHU_AT_PATTERN = re.compile(r"<at\b[^>]*>.*?</at>", flags=re.IGNORECASE | re.DOTALL)


def _detect_image_upload_meta(image_bytes: bytes) -> tuple[str, str]:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "snapshot.jpg", "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "snapshot.png", "image/png"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "snapshot.gif", "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "snapshot.webp", "image/webp"
    if image_bytes.startswith(b"BM"):
        return "snapshot.bmp", "image/bmp"
    if image_bytes.startswith((b"II*\x00", b"MM\x00*")):
        return "snapshot.tiff", "image/tiff"
    if image_bytes.startswith(b"\x00\x00\x01\x00"):
        return "snapshot.ico", "image/x-icon"
    return "snapshot.jpg", "image/jpeg"


def _now_ts() -> float:
    return time.time()


def _truncate_text(text: str, max_len: int = 3500) -> str:
    if not text:
        return text
    if len(text) <= max_len:
        return text
    return text[: max_len - 40] + "\n...(内容过长已截断)..."


def normalize_incoming_feishu_text(*, raw_text: str) -> str:
    text = _FEISHU_AT_PATTERN.sub(" ", raw_text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def looks_like_feishu_command(*, raw_text: str) -> bool:
    normalized = normalize_incoming_feishu_text(raw_text=raw_text)
    if not normalized:
        return False
    if normalized.startswith("$"):
        return True
    if normalized.startswith("/") or normalized.startswith("／"):
        return True
    return bool(
        re.match(
            r"^(help|\?|帮助|status|state|状态|检查状态|stop|cancel|停止|终止|start|开始)(\s|$)",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def should_handle_incoming_feishu_message(*, message: dict[str, Any], raw_text: str) -> bool:
    chat_type = ((message.get("chat_type") or "") if isinstance(message, dict) else "").strip().lower()
    if chat_type != "group":
        return True

    mentions = message.get("mentions") if isinstance(message, dict) else None
    has_mentions = bool(mentions) if isinstance(mentions, list) else False
    has_at_tag = bool(_FEISHU_AT_PATTERN.search(raw_text or ""))
    return has_mentions or has_at_tag or looks_like_feishu_command(raw_text=raw_text)


def _build_streaming_assistant_content(assistant_text: str, status_text: str | None) -> str:
    normalized_text = assistant_text or ""
    normalized_status = (status_text or "").strip()

    if normalized_text and normalized_status:
        return f"{normalized_text}\n\n_Status: {normalized_status}_"
    if normalized_text:
        return normalized_text
    if normalized_status:
        return f"_{normalized_status}_"
    return "_Thinking..._"


class FeishuClient:
    """
    Minimal Feishu(Open Platform) client for:
    - tenant_access_token (internal app)
    - send text message (chat_id)
    - upload image + send image message (chat_id)
    """

    def __init__(
        self,
        *,
        app_id: str,
        app_secret: str,
        api_base: str = "https://open.feishu.cn/open-apis",
        timeout_s: float = 8.0,
    ):
        self._app_id = app_id
        self._app_secret = app_secret
        self._api_base = api_base.rstrip("/")
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))

        self._token_lock = asyncio.Lock()
        self._tenant_access_token: str | None = None
        self._token_expire_at: float = 0.0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get_tenant_access_token(self) -> str:
        async with self._token_lock:
            if self._tenant_access_token and _now_ts() < self._token_expire_at:
                return self._tenant_access_token

            url = f"{self._api_base}/auth/v3/tenant_access_token/internal"
            payload = {"app_id": self._app_id, "app_secret": self._app_secret}
            resp = await self._client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") != 0:
                raise RuntimeError(f"Failed to get tenant_access_token: {data}")

            token = data.get("tenant_access_token")
            expire = int(data.get("expire") or 0)
            if not token:
                raise RuntimeError(f"tenant_access_token missing in response: {data}")

            # Refresh a bit earlier than actual expire time.
            self._tenant_access_token = token
            self._token_expire_at = _now_ts() + max(expire - 60, 0)
            return token

    async def _post_im_message(
        self,
        *,
        receive_id_type: str,
        receive_id: str,
        msg_type: str,
        content: dict[str, Any],
    ) -> None:
        token = await self._get_tenant_access_token()
        url = f"{self._api_base}/im/v1/messages?receive_id_type={receive_id_type}"
        payload = {
            "receive_id": receive_id,
            "msg_type": msg_type,
            "content": json.dumps(content, ensure_ascii=False),
        }
        resp = await self._client.post(url, headers={"Authorization": f"Bearer {token}"}, json=payload)
        response_text = resp.text
        if resp.is_error:
            raise RuntimeError(
                f"Failed to send {msg_type} message via {receive_id_type}={receive_id}: "
                f"HTTP {resp.status_code} {response_text}"
            )
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Failed to send {msg_type} message via {receive_id_type}={receive_id}: {data}")

    async def send_text(self, *, chat_id: str, text: str) -> None:
        await self._post_im_message(
            receive_id_type="chat_id",
            receive_id=chat_id,
            msg_type="text",
            content={"text": _truncate_text(text)},
        )

    async def upload_image_bytes(self, *, image_bytes: bytes, filename: str = "snapshot.jpg") -> str:
        if not image_bytes:
            raise ValueError("image_bytes is empty")
        token = await self._get_tenant_access_token()
        url = f"{self._api_base}/im/v1/images"
        detected_filename, content_type = _detect_image_upload_meta(image_bytes)
        upload_filename = filename or detected_filename
        files = {"image": (upload_filename, image_bytes, content_type)}
        data = {"image_type": "message"}
        resp = await self._client.post(url, headers={"Authorization": f"Bearer {token}"}, data=data, files=files)
        logid = resp.headers.get("X-Tt-Logid", "")
        response_text = resp.text
        if resp.is_error:
            raise RuntimeError(
                f"Failed to upload image: HTTP {resp.status_code} {response_text}"
                + (f" (logid={logid})" if logid else "")
            )
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"Failed to upload image: {body}" + (f" (logid={logid})" if logid else ""))
        image_key = ((body.get("data") or {}) or {}).get("image_key")
        if not image_key:
            raise RuntimeError(f"image_key missing in upload response: {body}" + (f" (logid={logid})" if logid else ""))
        return image_key

    async def send_image(self, *, chat_id: str, image_bytes: bytes) -> None:
        image_key = await self.upload_image_bytes(image_bytes=image_bytes)
        await self._post_im_message(
            receive_id_type="chat_id",
            receive_id=chat_id,
            msg_type="image",
            content={"image_key": image_key},
        )

    async def send_text_to_target(self, *, target: "FeishuReplyTarget", text: str) -> None:
        last_error: Exception | None = None
        for receive_id_type, receive_id in target.iter_receive_ids():
            try:
                await self._post_im_message(
                    receive_id_type=receive_id_type,
                    receive_id=receive_id,
                    msg_type="text",
                    content={"text": _truncate_text(text)},
                )
                return
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "[FeishuBot] send_text failed via %s=%s: %s",
                    receive_id_type,
                    receive_id,
                    exc,
                )
        if last_error is not None:
            raise last_error
        raise RuntimeError("No available Feishu receive_id to send text message.")

    async def send_image_to_target(self, *, target: "FeishuReplyTarget", image_bytes: bytes) -> None:
        image_key = await self.upload_image_bytes(image_bytes=image_bytes)
        last_error: Exception | None = None
        for receive_id_type, receive_id in target.iter_receive_ids():
            try:
                await self._post_im_message(
                    receive_id_type=receive_id_type,
                    receive_id=receive_id,
                    msg_type="image",
                    content={"image_key": image_key},
                )
                return
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "[FeishuBot] send_image failed via %s=%s: %s",
                    receive_id_type,
                    receive_id,
                    exc,
                )
        if last_error is not None:
            raise last_error
        raise RuntimeError("No available Feishu receive_id to send image message.")


@dataclass(frozen=True)
class FeishuReplyTarget:
    chat_id: str = ""
    open_id: str = ""
    user_id: str = ""
    chat_type: str = ""

    def iter_receive_ids(self) -> Iterable[tuple[str, str]]:
        seen: set[tuple[str, str]] = set()
        receive_ids: list[tuple[str, str]] = [("chat_id", self.chat_id.strip())]
        if not self.is_group_chat or not self.chat_id.strip():
            receive_ids.extend(
                [
                    ("open_id", self.open_id.strip()),
                    ("user_id", self.user_id.strip()),
                ]
            )

        for receive_id_type, receive_id in receive_ids:
            if not receive_id:
                continue
            key = (receive_id_type, receive_id)
            if key in seen:
                continue
            seen.add(key)
            yield key

    @property
    def primary_id(self) -> str:
        for _, receive_id in self.iter_receive_ids():
            return receive_id
        return ""

    @property
    def is_group_chat(self) -> bool:
        return self.chat_type.strip().lower() == "group"


def extract_reply_target(*, event: dict[str, Any], message: dict[str, Any]) -> FeishuReplyTarget:
    sender = event.get("sender") or {}
    sender_id = sender.get("sender_id") or {}
    return FeishuReplyTarget(
        chat_id=(message.get("chat_id") or "").strip(),
        open_id=(sender_id.get("open_id") or "").strip(),
        user_id=(sender_id.get("user_id") or "").strip(),
        chat_type=(message.get("chat_type") or "").strip().lower(),
    )


@dataclass(frozen=True)
class ParsedCommand:
    name: str
    arg: str = ""


_HELP_TEXT = """可用指令：
1) status / 状态 / 检查状态：查看当前任务状态
2) start <任务描述> / 开始 <任务描述>：启动任务（完成后自动回传结果）
3) stop / 停止：停止当前任务（尽力取消）
4) help / 帮助：查看本帮助
"""


def _parse_command(text: str) -> ParsedCommand:
    raw = (text or "").strip()
    if not raw:
        return ParsedCommand(name="help")

    # allow leading slash
    if raw.startswith("/") or raw.startswith("／"):
        raw = raw[1:].strip()

    if re.fullmatch(r"(help|\?|帮助)", raw, flags=re.IGNORECASE):
        return ParsedCommand(name="help")

    if re.fullmatch(r"(status|state|状态|检查状态)", raw, flags=re.IGNORECASE):
        return ParsedCommand(name="status")

    if re.fullmatch(r"(stop|cancel|停止|终止)", raw, flags=re.IGNORECASE):
        return ParsedCommand(name="stop")

    m = re.match(r"^(start|开始)\s+(.+)$", raw, flags=re.IGNORECASE)
    if m:
        return ParsedCommand(name="start", arg=m.group(2).strip())

    return ParsedCommand(name="chat", arg=raw)


class _EventIdDeduper:
    def __init__(self, *, max_size: int = 1024, ttl_s: int = 10 * 60):
        self._max_size = max_size
        self._ttl_s = ttl_s
        self._items: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def seen(self, event_id: str) -> bool:
        if not event_id:
            return False
        async with self._lock:
            now = _now_ts()
            # purge expired
            expired = [k for k, ts in self._items.items() if now - ts > self._ttl_s]
            for k in expired:
                self._items.pop(k, None)
            if event_id in self._items:
                return True
            if len(self._items) >= self._max_size:
                # drop oldest
                oldest_key = min(self._items.items(), key=lambda kv: kv[1])[0]
                self._items.pop(oldest_key, None)
            self._items[event_id] = now
            return False


class FeishuBotController:
    def __init__(
        self,
        *,
        session_inst: Any,
        ui_state_machine: Any,
        user_input_processing: asyncio.Event,
        chat_history: list,
        format_bot_msg: Callable[[str], str] | None = None,
        build_status_text: Callable[[], str] | None = None,
        get_image_bytes: Callable[[], Awaitable[bytes | None]] | None = None,
        feishu_client: FeishuClient,
    ):
        self._session_inst = session_inst
        self._ui_state_machine = ui_state_machine
        self._user_input_processing = user_input_processing
        self._chat_history = chat_history

        self._format_bot_msg = format_bot_msg or (lambda s: s)
        self._build_status_text = build_status_text or (lambda: "No status builder configured.")
        self._get_image_bytes = get_image_bytes or (lambda: asyncio.sleep(0, result=None))

        self._feishu = feishu_client
        self._deduper = _EventIdDeduper()

        self._task_lock = asyncio.Lock()
        self._active_task: asyncio.Task | None = None
        self._active_task_prompt: str | None = None
        self._active_task_started_at: float | None = None

    async def _send_ui_event(self, event_name: str) -> bool:
        event = getattr(self._ui_state_machine, event_name, None)
        if event is None:
            event = getattr(self._ui_state_machine.__class__, event_name, None)
        if event is None:
            try:
                module = importlib.import_module(self._ui_state_machine.__class__.__module__)
            except Exception:
                module = None
            ui_event_cls = getattr(module, "UIEvent", None) if module is not None else None
            if ui_event_cls is not None:
                event = getattr(ui_event_cls, event_name, None)
        if event is None or not hasattr(self._ui_state_machine, "send_event"):
            return False
        try:
            return bool(await self._ui_state_machine.send_event(event))
        except Exception:
            logger.debug("[FeishuBot] Failed to send UI event %s.", event_name, exc_info=True)
            return False

    async def _prepare_ui_for_user_input(self) -> None:
        if not hasattr(self._ui_state_machine, "send_event"):
            return
        ok = await self._send_ui_event("USER_INPUT")
        if ok or not hasattr(self._ui_state_machine, "wait_inference_done"):
            return
        await self._ui_state_machine.wait_inference_done()
        await self._send_ui_event("USER_INPUT")

    async def send_status(self, *, reply_target: FeishuReplyTarget) -> None:
        status_text = self._build_status_text()

        async with self._task_lock:
            if self._active_task and not self._active_task.done():
                elapsed_s = int(_now_ts() - (self._active_task_started_at or _now_ts()))
                status_text += f"\n\n🟠 BotTask: RUNNING ({elapsed_s}s)\nPrompt: {self._active_task_prompt}"
            else:
                status_text += "\n\n🟢 BotTask: IDLE"

        await self._feishu.send_text_to_target(target=reply_target, text=status_text)

    async def stop(self, *, reply_target: FeishuReplyTarget) -> None:
        async with self._task_lock:
            task = self._active_task
            if not task or task.done():
                await self._feishu.send_text_to_target(target=reply_target, text="当前没有正在运行的任务。")
                return

            try:
                # best-effort: set cancel flag if supported
                if hasattr(self._session_inst, "agent") and hasattr(self._session_inst.agent, "request_cancel"):
                    self._session_inst.agent.request_cancel()
            except Exception:
                pass

            task.cancel()
            await self._feishu.send_text_to_target(target=reply_target, text="已请求停止当前任务（尽力取消中）。")

    async def start(self, *, reply_target: FeishuReplyTarget, prompt: str) -> None:
        if not prompt.strip():
            await self._feishu.send_text_to_target(target=reply_target, text="用法：start <任务描述> / 开始 <任务描述>")
            return

        async with self._task_lock:
            if self._active_task and not self._active_task.done():
                await self._feishu.send_text_to_target(
                    target=reply_target, text="已有任务在运行中，请先 stop 或等待完成。"
                )
                return

            self._active_task_prompt = prompt
            self._active_task_started_at = _now_ts()
            self._active_task = asyncio.create_task(self._run_task_and_reply(reply_target=reply_target, prompt=prompt))
            self._active_task.add_done_callback(self._log_task_exception)

    def _log_task_exception(self, task: asyncio.Task) -> None:
        try:
            _ = task.result()
        except asyncio.CancelledError:
            logger.info("[FeishuBot] Task cancelled.")
        except Exception as e:
            logger.exception(f"[FeishuBot] Task error: {type(e).__name__}: {e}")

    async def handle_event(
        self,
        *,
        event_id: str,
        reply_target: FeishuReplyTarget,
        text: str,
    ) -> None:
        if await self._deduper.seen(event_id):
            logger.info(f"[FeishuBot] Skip duplicated event_id={event_id}")
            return

        cmd = _parse_command(text)
        logger.info(f"[FeishuBot] target={reply_target.primary_id} cmd={cmd.name} arg_len={len(cmd.arg)}")

        if cmd.name == "help":
            await self._feishu.send_text_to_target(target=reply_target, text=_HELP_TEXT)
            return

        if cmd.name == "status":
            await self.send_status(reply_target=reply_target)
            return

        if cmd.name == "stop":
            await self.stop(reply_target=reply_target)
            return

        if cmd.name == "start":
            await self.start(reply_target=reply_target, prompt=cmd.arg)
            return

        # default: treat as a chat/task prompt, run and reply.
        await self.start(reply_target=reply_target, prompt=cmd.arg)

    async def _run_task_and_reply(self, *, reply_target: FeishuReplyTarget, prompt: str) -> None:
        # block auto-inference while processing Feishu input
        self._user_input_processing.set()
        inference_started = False
        assistant_index: int | None = None
        try:
            # Try to transfer UI state to accept a new input (best-effort)
            try:
                await self._prepare_ui_for_user_input()
            except Exception:
                pass

            # Keep gradio UI chat in sync.
            try:
                self._chat_history.append({"role": "user", "content": prompt})
            except Exception:
                pass

            prepared_message = None
            prepare_message_for_agent = getattr(self._session_inst, "prepare_message_for_agent", None)
            if callable(prepare_message_for_agent):
                prepared_message = prepare_message_for_agent(prompt)
                if prepared_message.error_message:
                    try:
                        self._chat_history.append({"role": "assistant", "content": prepared_message.error_message})
                    except Exception:
                        pass
                    await self._feishu.send_text_to_target(target=reply_target, text=prepared_message.error_message)
                    return

            inference_started = await self._send_ui_event("INFER_START")
            assistant_text = ""
            status_text = prepared_message.status_message if prepared_message else "Thinking..."
            final_text = ""
            agent_prompt = prepared_message.message if prepared_message and prepared_message.message else prompt

            try:
                self._chat_history.append({"role": "assistant", "content": "_Thinking..._"})
                assistant_index = len(self._chat_history) - 1
            except Exception:
                assistant_index = None

            run_once_stream = getattr(self._session_inst, "run_once_stream", None)
            if callable(run_once_stream):
                async for event in run_once_stream(agent_prompt):
                    event_type = event.get("type")
                    if event_type == "status":
                        status_text = event.get("text", status_text) or status_text
                    elif event_type == "text_delta":
                        assistant_text += event.get("delta", "")
                    elif event_type == "final":
                        final_text = event.get("text", "") or ""

                    if assistant_index is not None:
                        self._chat_history[assistant_index]["content"] = _build_streaming_assistant_content(
                            assistant_text,
                            status_text,
                        )
            else:
                final_text = await self._session_inst.run_once(agent_prompt)

            final_text = resolve_final_response_text(final_text, assistant_text)
            formatted = self._format_bot_msg(final_text)

            try:
                if assistant_index is not None:
                    self._chat_history[assistant_index]["content"] = formatted
                else:
                    self._chat_history.append({"role": "assistant", "content": formatted})
            except Exception:
                pass

            if inference_started:
                await self._send_ui_event("INFER_SUCCESS")
            try:
                await self._feishu.send_text_to_target(target=reply_target, text=formatted)
            except Exception as exc:
                logger.warning("[FeishuBot] Failed to send task result: %s", exc, exc_info=True)
        except asyncio.CancelledError:
            if inference_started:
                await self._send_ui_event("INFER_FAIL")
            if assistant_index is not None:
                try:
                    self._chat_history[assistant_index]["content"] = "🛑 任务已取消。"
                except Exception:
                    pass
            try:
                await self._feishu.send_text_to_target(target=reply_target, text="🛑 任务已取消。")
            except Exception as exc:
                logger.warning("[FeishuBot] Failed to send cancellation reply: %s", exc, exc_info=True)
            raise
        except Exception as e:
            if inference_started:
                await self._send_ui_event("INFER_FAIL")
            if assistant_index is not None:
                try:
                    if assistant_text:
                        self._chat_history[assistant_index]["content"] = f"{assistant_text}\n\n_推理失败: {e}_"
                    else:
                        self._chat_history[assistant_index]["content"] = f"推理失败: {e}"
                except Exception:
                    pass
            try:
                await self._feishu.send_text_to_target(
                    target=reply_target,
                    text=f"❌ 任务执行失败：{type(e).__name__}: {e}",
                )
            except Exception as reply_exc:
                logger.warning("[FeishuBot] Failed to send error reply: %s", reply_exc, exc_info=True)
        finally:
            self._user_input_processing.clear()
            async with self._task_lock:
                if self._active_task and self._active_task.done():
                    self._active_task = None
