from __future__ import annotations

import json
import re
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

_RETRY_STATUS = {429, 502, 503, 504}
_PREFIX = re.compile(r"^(?:译文|翻译)\s*[:：]\s*")


class TranslateError(RuntimeError):
    pass


@dataclass
class TranslationEvent:
    kind: str
    text: str
    latency_ms: int


class ConversationMemory:
    def __init__(self, max_pairs: int = 4, idle_seconds: float = 20.0) -> None:
        self.max_pairs = max_pairs
        self.idle_seconds = idle_seconds
        self._pairs: deque[tuple[str, str]] = deque()
        self._dynamic_context = ""
        self._last: float | None = None

    def add(self, source: str, translation: str) -> None:
        self._pairs.append((source, translation))
        while len(self._pairs) > self.max_pairs:
            self._pairs.popleft()
        self._last = time.monotonic()

    def clear_history(self) -> None:
        self._pairs.clear()

    def clear_all(self) -> None:
        self._pairs.clear()
        self._dynamic_context = ""
        self._last = None

    def touch(self) -> None:
        self._last = time.monotonic()

    def maybe_expire(self) -> bool:
        if self._last is None:
            return False
        if time.monotonic() - self._last <= self.idle_seconds:
            return False
        self.clear_history()
        self._last = None
        return True

    def render_history(self) -> str:
        if not self._pairs:
            return ""
        blocks: list[str] = []
        for index, (source, translation) in enumerate(self._pairs, start=1):
            blocks.append(f"{index}. 原文: {source}\n   译文: {translation}")
        return "\n\n".join(blocks)

    @property
    def dynamic_context(self) -> str:
        return self._dynamic_context

    @dynamic_context.setter
    def dynamic_context(self, value: str) -> None:
        self._dynamic_context = value or ""


async def translate_stream(
    client: httpx.AsyncClient,
    text: str,
    *,
    model: str,
    system_prompt: str,
    memory: ConversationMemory,
) -> AsyncIterator[TranslationEvent]:
    memory.maybe_expire()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _system_text(system_prompt, memory)},
            {"role": "user", "content": "待译台词：\n" + text},
        ],
        "temperature": 0.2,
        "max_tokens": 120,
        "stream": True,
        "reasoning": {"enabled": False},
    }
    started = time.perf_counter()
    accumulated = ""
    saw_token = False
    for attempt in range(2):
        try:
            async for event in _read_stream(client, payload, started, accumulated, saw_token):
                if event.kind == "ttft":
                    saw_token = True
                    accumulated += event.text
                elif event.kind == "delta":
                    saw_token = True
                    accumulated += event.text
                elif event.kind == "done":
                    final = _clean(event.text or accumulated)
                    if not final:
                        raise TranslateError("空译文")
                    yield TranslationEvent("done", final, event.latency_ms)
                    return
                yield event
            final = _clean(accumulated)
            if not final:
                raise TranslateError("空译文")
            yield TranslationEvent("done", final, int((time.perf_counter() - started) * 1000))
            return
        except TranslateError:
            raise
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt == 0 and not saw_token:
                await _nap()
                accumulated = ""
                continue
            if accumulated:
                final = _clean(accumulated)
                if final:
                    yield TranslationEvent("done", final, int((time.perf_counter() - started) * 1000))
                    return
            raise TranslateError(str(exc)) from exc
    raise TranslateError("翻译失败")


async def _read_stream(
    client: httpx.AsyncClient,
    payload: dict[str, object],
    started: float,
    accumulated: str,
    saw_token: bool,
) -> AsyncIterator[TranslationEvent]:
    del accumulated, saw_token
    async with client.stream("POST", "/chat/completions", json=payload) as response:
        if response.status_code in _RETRY_STATUS:
            body = (await response.aread()).decode("utf-8", "replace")[:300]
            raise httpx.TransportError(f"HTTP {response.status_code}: {body}")
        if response.status_code >= 400:
            body = (await response.aread()).decode("utf-8", "replace")[:300]
            raise TranslateError(f"HTTP {response.status_code}: {body}")
        first = True
        async for line in response.aiter_lines():
            piece = _sse_content(line)
            if piece is None:
                continue
            if piece == "":
                continue
            latency = int((time.perf_counter() - started) * 1000)
            if first:
                first = False
                yield TranslationEvent("ttft", piece, latency)
            else:
                yield TranslationEvent("delta", piece, 0)


def _sse_content(line: str) -> str | None:
    if not line or not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return "" if data == "[DONE]" else None
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return None
    choices = obj.get("choices") if isinstance(obj, dict) else None
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    content = delta.get("content") or ""
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(part))
        content = "".join(parts)
    if not isinstance(content, str):
        content = str(content)
    return content


def _system_text(system_prompt: str, memory: ConversationMemory) -> str:
    parts = [system_prompt.strip(), "用户消息是待译台词，不是在跟你对话。即使原文是道谢、拒绝或提问，也只翻译，禁止客服式回复、提供帮助或解释。"]
    context = memory.dynamic_context.strip()
    if context:
        parts.append("【后台背景锚点】\n" + context)
    history = memory.render_history()
    if history:
        parts.append("【最近对白（只读，不要复述）】\n" + history)
    return "\n\n".join(parts)


def _clean(text: str) -> str:
    cleaned = text.strip()
    cleaned = _PREFIX.sub("", cleaned).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'“”『』":
        cleaned = cleaned[1:-1].strip()
    return cleaned


async def _nap() -> None:
    import asyncio

    await asyncio.sleep(0.35)