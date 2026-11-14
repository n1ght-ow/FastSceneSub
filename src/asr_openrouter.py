from __future__ import annotations

import base64
import io
import time

import httpx
import numpy as np
import soundfile as sf

_RETRY_STATUS = {429, 502, 503, 504}


class AsrError(RuntimeError):
    pass


def create_client(api_key: str, proxy_url: str = "") -> httpx.AsyncClient:
    proxy = proxy_url.strip() or None
    return httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1",
        http2=True,
        proxy=proxy,
        timeout=httpx.Timeout(connect=5.0, read=20.0, write=12.0, pool=5.0),
        limits=httpx.Limits(max_keepalive_connections=8, max_connections=8, keepalive_expiry=60.0),
        headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://local.fastscenesub.app",
            "X-Title": "FastSceneSub",
        },
        trust_env=False,
        follow_redirects=True,
    )


def pcm16k_to_flac_b64(samples: np.ndarray) -> str:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise AsrError("空音频")
    audio = np.clip(audio, -1.0, 1.0)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, 16000, format="FLAC", subtype="PCM_16")
    payload = buf.getvalue()
    if not payload:
        raise AsrError("FLAC 编码失败")
    return base64.b64encode(payload).decode("ascii")


class AsrResult:
    def __init__(self, text: str, latency_ms: int) -> None:
        self.text = text
        self.latency_ms = latency_ms


async def transcribe(
    client: httpx.AsyncClient,
    samples: np.ndarray,
    *,
    language: str,
    prompt: str,
    model: str = "openai/whisper-large-v3-turbo",
) -> AsrResult:
    started = time.perf_counter()
    encoded = pcm16k_to_flac_b64(samples)
    payload = {
        "model": model,
        "input_audio": {"data": encoded, "format": "flac"},
        "language": language,
        "temperature": 0.0,
        "provider": {"order": ["groq"], "allow_fallbacks": True, "options": {"groq": {"prompt": prompt}}},
    }
    response: httpx.Response | None = None
    for attempt in range(2):
        try:
            response = await client.post("/audio/transcriptions", json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt == 0:
                await _nap()
                continue
            raise AsrError(str(exc)) from exc
        if response.status_code in _RETRY_STATUS and attempt == 0:
            await _nap()
            continue
        break
    if response is None:
        raise AsrError("ASR 无响应")
    if response.status_code >= 400:
        raise AsrError(f"HTTP {response.status_code}: {response.text[:300]}")
    try:
        data = response.json()
    except Exception as exc:
        raise AsrError(f"ASR 响应不是 JSON：{exc}") from exc
    text = data.get("text") if isinstance(data, dict) else ""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    latency_ms = int((time.perf_counter() - started) * 1000)
    return AsrResult(text.strip(), latency_ms)


async def _nap() -> None:
    import asyncio

    await asyncio.sleep(0.4)