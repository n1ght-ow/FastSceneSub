from __future__ import annotations

import asyncio
from collections import deque
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication, QDialog

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QThread, QTimer, Signal

from src.asr_openrouter import AsrError, create_client, transcribe
from src.audio_capture import AudioCapture
from src.config import AppConfig, VAD_MODEL_PATH, ensure_vad_model, load_config, save_config
from src.context_profiler import ContextProfiler
from src.filters_and_cache import PhraseCache, is_assistant_reply, is_hallucination, normalize_text
from src.scene_engine import get_scene
from src.translator_llm import ConversationMemory, TranslateError, translate_stream
from src.ui_overlay import ControlBar, SettingsDialog, SubtitleWindow
from src.vad_chunker import VadChunker, VadParams



def _proxy_hint(proxy_url: str) -> str:
    port = _proxy_port(proxy_url)
    if port is not None and _port_open(port):
        return ""
    found = next((item for item in (7897, 7890, 7891, 10808, 1080) if _port_open(item)), None)
    if found is None:
        return "\n本机代理端口没有程序在听。先打开 Clash，再填写它的混合端口。"
    return f"\n{proxy_url or '当前代理'} 没有连上。检测到本机端口 {found} 可用，请改成 http://127.0.0.1:{found}"


def _proxy_port(proxy_url: str) -> int | None:
    text = (proxy_url or "").strip()
    if ":" not in text:
        return None
    tail = text.rsplit(":", 1)[-1].strip("/")
    return int(tail) if tail.isdigit() else None


def _port_open(port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(0.2)
    try:
        sock.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()



def _consume_task_result(task: asyncio.Task) -> None:
    if not task.done():
        return
    try:
        task.exception()
    except (asyncio.CancelledError, asyncio.InvalidStateError):
        return


class EngineThread(QThread):
    original_ready = Signal(str, int)
    translation_text = Signal(str)
    ttft_ready = Signal(int)
    status_changed = Signal(str)
    test_finished = Signal(str)
    listening_changed = Signal(bool)
    cleared = Signal()

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._api_key = config.openrouter_api_key
        self._proxy = config.proxy_url
        self._scene_id = "en_basketball"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._pcm_q: asyncio.Queue[np.ndarray | None] | None = None
        self._client = None
        self._client_token: tuple[str, str] | None = None
        self.capture = AudioCapture(self._on_audio, lambda text: self.status_changed.emit(text))
        self._chunker: VadChunker | None = None
        self._memory = ConversationMemory()
        self._cache = PhraseCache()
        self._profiler: ContextProfiler | None = None
        self._listening = False
        self._shutdown = False
        self._pending: deque[np.ndarray] = deque()
        self._pending_event: asyncio.Event | None = None
        self._translate_tasks: set[asyncio.Task[None]] = set()
        self._epoch = 0
        self._shown_gen = 0
        self._asr_client = None
        self._asr_tasks: list[asyncio.Task] = []
        self._last_loud = time.monotonic()
        self._last_original_key = ""

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._amain())
        finally:
            self._loop.close()
            self._loop = None

    def start_listening(self) -> None:
        self._submit(self._start_listening())

    def pause_listening(self) -> None:
        self._submit(self._pause_listening())

    def toggle_listening(self) -> None:
        self._submit(self._toggle_listening())

    def set_scene(self, scene_id: str) -> None:
        self._submit(self._set_scene(scene_id))

    def reset_show(self) -> None:
        self._submit(self._reset_show())

    def update_credentials(self, api_key: str, proxy_url: str) -> None:
        self._submit(self._update_credentials(api_key, proxy_url))

    def request_test(self, api_key: str, proxy_url: str) -> None:
        self._submit(self._request_test(api_key, proxy_url))

    def shutdown(self) -> None:
        self._submit(self._shutdown_loop())

    def _submit(self, coro):
        if not self._ready.wait(5) or self._loop is None:
            coro.close()
            return None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _amain(self) -> None:
        self._pcm_q = asyncio.Queue(maxsize=400)
        self._pending_event = asyncio.Event()
        self._ready.set()
        self.listening_changed.emit(False)
        worker = asyncio.create_task(self._asr_loop())
        try:
            while not self._shutdown:
                assert self._pcm_q is not None
                try:
                    item = await asyncio.wait_for(self._pcm_q.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                if item is None or not self._listening or self._chunker is None:
                    continue
                self._note_level(item)
                try:
                    pieces = self._chunker.push(item)
                except Exception as exc:
                    self.status_changed.emit(f"VAD 失败: {exc}")
                    continue
                for piece in pieces:
                    if not self._listening or self._shutdown:
                        break
                    # 识别和翻译都不再堵住采集，短句可以马上送出。
                    self._enqueue_piece(piece)
        finally:
            worker.cancel()
            self._cancel_asr()
            self._cancel_translates()
            tasks = [worker, *list(self._translate_tasks), *list(self._asr_tasks)]
            await asyncio.gather(*tasks, return_exceptions=True)
        self.capture.stop()
        await self._close_clients()

    def _enqueue_piece(self, audio: np.ndarray) -> None:
        now = time.monotonic()
        fresh: deque[tuple[np.ndarray, float]] = deque()
        for item in self._pending:
            if now - item[1] <= 2.0:
                fresh.append(item)
        fresh.append((np.ascontiguousarray(audio, dtype=np.float32), now))
        while len(fresh) > 2:
            fresh.popleft()
        self._pending = fresh
        event = self._pending_event
        if event is not None:
            event.set()

    def _note_level(self, samples: np.ndarray) -> None:
        rms = 0.0
        if samples.size:
            rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
        now = time.monotonic()
        if rms >= 0.015:
            self._last_loud = now
            return
        if now - self._last_loud < 0.45:
            return
        self._pending.clear()
        for task in self._asr_tasks[:-1]:
            if not task.done():
                task.cancel()

    def _fill_asr(self) -> None:
        client = self._asr_client or self._client
        if client is None or not self._listening:
            return
        now = time.monotonic()
        while self._pending and now - self._pending[0][1] > 2.0:
            self._pending.popleft()
        while len(self._asr_tasks) < 2 and self._pending and not self._shutdown:
            audio, created = self._pending.popleft()
            task = asyncio.create_task(self._transcribe_piece(audio, self._scene_id))
            task.add_done_callback(_consume_task_result)
            task.chunk_at = created  # type: ignore[attr-defined]
            task.epoch = self._epoch  # type: ignore[attr-defined]
            task.scene_id = self._scene_id  # type: ignore[attr-defined]
            self._asr_tasks.append(task)

    def _cancel_asr(self) -> None:
        tasks = list(self._asr_tasks)
        self._asr_tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()

    async def _transcribe_piece(self, audio: np.ndarray, scene_id: str):
        scene = get_scene(scene_id)
        client = self._asr_client or self._client
        if client is None:
            raise AsrError("ASR client missing")
        return await transcribe(
            client,
            audio,
            language=scene.language,
            prompt=scene.asr_prompt,
        )

    def _invalidate_pending(self) -> None:
        self._epoch += 1
        self._shown_gen += 1
        self._last_original_key = ""
        self._pending.clear()
        self._cancel_asr()
        self._cancel_translates()

    def _cancel_translates(self) -> None:
        for task in list(self._translate_tasks):
            task.cancel()

    async def _asr_loop(self) -> None:
        try:
            while not self._shutdown:
                self._fill_asr()
                if not self._asr_tasks:
                    event = self._pending_event
                    if event is None:
                        return
                    event.clear()
                    if self._pending or self._shutdown:
                        continue
                    await event.wait()
                    continue
                await asyncio.wait(set(self._asr_tasks), return_when=asyncio.FIRST_COMPLETED)
                while self._asr_tasks and self._asr_tasks[0].done():
                    task = self._asr_tasks.pop(0)
                    await self._finish_asr(task)
        except asyncio.CancelledError:
            self._cancel_asr()
            return

    async def _finish_asr(self, task: asyncio.Task) -> None:
        try:
            result = task.result()
        except asyncio.CancelledError:
            return
        except AsrError as exc:
            if getattr(task, "epoch", None) == self._epoch and self._listening:
                self.status_changed.emit(f"ASR 失败: {exc}")
            return
        except Exception:
            return
        epoch = getattr(task, "epoch", None)
        scene_id = str(getattr(task, "scene_id", self._scene_id))
        if self._shutdown or epoch != self._epoch or not self._listening:
            return
        if time.monotonic() - float(getattr(task, "chunk_at", 0.0)) > 2.6:
            return
        scene = get_scene(scene_id)
        if scene.scene_id != self._scene_id:
            return
        text = result.text.strip()
        if not text or is_hallucination(text):
            return
        key = normalize_text(text)
        if key and key == self._last_original_key:
            return
        self._last_original_key = key
        self._shown_gen += 1
        gen = self._shown_gen
        self.original_ready.emit(text, result.latency_ms)
        self._cancel_translates()
        follow = asyncio.create_task(self._translate_shown(gen, text))
        self._translate_tasks.add(follow)
        follow.add_done_callback(self._translate_tasks.discard)

    async def _translate_shown(self, gen: int, text: str) -> None:
        try:
            if gen != self._shown_gen or self._client is None:
                return
            scene = get_scene(self._scene_id)
            key = normalize_text(text)
            cached = self._cache.get(key)
            if cached:
                if gen != self._shown_gen or is_assistant_reply(cached):
                    if is_assistant_reply(cached):
                        self.cleared.emit()
                    return
                self.ttft_ready.emit(0)
                self.translation_text.emit(cached)
                self._memory.add(text, cached)
                if self._profiler is not None:
                    self._profiler.observe(text)
                return
            models = [scene.llm_model]
            if scene.fallback_model and scene.fallback_model not in models:
                models.append(scene.fallback_model)
            for model_index, model in enumerate(models):
                if gen != self._shown_gen or self._client is None:
                    return
                accumulated = ""
                final = ""
                refused = False
                try:
                    async for event in translate_stream(
                        self._client,
                        text,
                        model=model,
                        system_prompt=scene.system_prompt,
                        memory=self._memory,
                    ):
                        if gen != self._shown_gen:
                            return
                        joined = accumulated + (event.text or "")
                        if is_assistant_reply(joined) or is_assistant_reply(event.text):
                            refused = True
                            break
                        if event.kind == "ttft":
                            self.ttft_ready.emit(event.latency_ms)
                            accumulated += event.text
                            self.translation_text.emit(accumulated)
                        elif event.kind == "delta":
                            accumulated += event.text
                            self.translation_text.emit(accumulated)
                        elif event.kind == "done":
                            final = event.text or accumulated
                            self.translation_text.emit(final)
                except TranslateError:
                    if model_index + 1 < len(models):
                        continue
                    raise
                if refused or (final and is_assistant_reply(final)):
                    if model_index + 1 < len(models):
                        self.translation_text.emit("")
                        continue
                    if gen == self._shown_gen:
                        self.cleared.emit()
                    return
                if final and gen == self._shown_gen:
                    self._memory.add(text, final)
                    self._cache.put(key, final)
                    if self._profiler is not None:
                        self._profiler.observe(text)
                    return
        except asyncio.CancelledError:
            return
        except TranslateError as exc:
            if gen == self._shown_gen:
                self.status_changed.emit(f"翻译失败: {exc}")

    def _on_audio(self, samples: np.ndarray) -> None:
        loop = self._loop
        queue = self._pcm_q
        if loop is None or queue is None or not self._listening:
            return
        data = np.ascontiguousarray(samples, dtype=np.float32).copy()

        def _put() -> None:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                pass

        loop.call_soon_threadsafe(_put)

    async def _warmup(self) -> None:
        scene = get_scene(self._scene_id)
        payload = {
            "model": scene.llm_model,
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
            "reasoning": {"enabled": False},
        }
        for client in (self._client, self._asr_client):
            if client is None:
                continue
            try:
                await client.post("/chat/completions", json=payload)
            except Exception:
                continue

    async def _start_listening(self) -> None:
        if self._listening:
            return
        if not self._api_key.strip():
            self.status_changed.emit("请先在设置中填写 OpenRouter API Key")
            self.listening_changed.emit(False)
            return
        try:
            await asyncio.to_thread(ensure_vad_model, lambda text: self.status_changed.emit(text))
            scene = get_scene(self._scene_id)
            params = VadParams(scene.speech_threshold, scene.min_silence_ms, scene.max_speech_s)
            if self._chunker is None:
                self._chunker = await asyncio.to_thread(VadChunker, VAD_MODEL_PATH, params)
            else:
                self._chunker.set_params(params)
            await self._ensure_client()
            if self._profiler is not None:
                self._profiler.set_scene(scene)
            self.capture.start()
            self._listening = bool(self.capture.running)
            if self._listening:
                warm = asyncio.create_task(self._warmup())
                warm.add_done_callback(_consume_task_result)
            self.listening_changed.emit(self._listening)
            self.status_changed.emit("正在监听系统音频" if self._listening else "采集没有启动")
        except Exception as exc:
            self._listening = False
            self.listening_changed.emit(False)
            self.status_changed.emit(f"启动失败: {exc}")

    async def _pause_listening(self) -> None:
        self._invalidate_pending()
        self._listening = False
        self.capture.stop()
        if self._chunker is not None:
            self._chunker.reset()
        self.listening_changed.emit(False)
        self.status_changed.emit("已暂停")

    async def _toggle_listening(self) -> None:
        if self._listening:
            await self._pause_listening()
        else:
            await self._start_listening()

    async def _set_scene(self, scene_id: str) -> None:
        self._invalidate_pending()
        self._scene_id = scene_id
        scene = get_scene(scene_id)
        if self._chunker is not None:
            self._chunker.set_params(
                VadParams(scene.speech_threshold, scene.min_silence_ms, scene.max_speech_s)
            )
        self._memory.clear_all()
        self._cache.clear()
        if self._profiler is not None:
            self._profiler.set_scene(scene)
            self._profiler.reset()
        self.cleared.emit()
        self.status_changed.emit("已切换场景")

    async def _reset_show(self) -> None:
        self._invalidate_pending()
        self._memory.clear_all()
        self._cache.clear()
        if self._profiler is not None:
            self._profiler.reset()
        if self._chunker is not None:
            self._chunker.reset()
        self.cleared.emit()
        self.status_changed.emit("已重置")

    async def _update_credentials(self, api_key: str, proxy_url: str) -> None:
        self._api_key = api_key
        self._proxy = proxy_url
        await self._close_clients()
        if self._api_key.strip():
            await self._ensure_client()

    async def _close_clients(self) -> None:
        clients = [self._client, self._asr_client]
        self._client = None
        self._asr_client = None
        self._client_token = None
        for client in clients:
            if client is None:
                continue
            try:
                await client.aclose()
            except Exception:
                pass

    async def _ensure_client(self) -> None:
        token = (self._api_key, self._proxy)
        if self._client is not None and self._asr_client is not None and self._client_token == token:
            return
        await self._close_clients()
        self._client = create_client(self._api_key, self._proxy)
        self._asr_client = create_client(self._api_key, self._proxy)
        self._client_token = token
        if self._profiler is None:
            self._profiler = ContextProfiler(self._client, self._memory)
            self._profiler.set_scene(get_scene(self._scene_id))
        else:
            self._profiler.set_client(self._client)

    async def _request_test(self, api_key: str, proxy_url: str) -> None:
        client = None
        try:
            if not api_key.strip():
                self.test_finished.emit("缺少 API Key")
                return
            client = create_client(api_key, proxy_url)
            scene = get_scene(self._scene_id)
            samples = np.zeros(8000, dtype=np.float32)
            started = time.perf_counter()
            try:
                result = await transcribe(
                    client,
                    samples,
                    language=scene.language,
                    prompt=scene.asr_prompt,
                )
                stt = f"STT {result.latency_ms} ms，文本：{result.text or '（空）'}"
            except Exception as exc:
                elapsed = int((time.perf_counter() - started) * 1000)
                stt = f"STT 失败（{elapsed} ms）：{exc}{_proxy_hint(proxy_url)}"
            started = time.perf_counter()
            try:
                response = await client.post(
                    "/chat/completions",
                    json={
                        "model": scene.llm_model,
                        "messages": [{"role": "user", "content": "只回复：连通成功"}],
                        "temperature": 0.2,
                        "max_tokens": 20,
                        "stream": False,
                        "reasoning": {"enabled": False},
                    },
                )
                elapsed = int((time.perf_counter() - started) * 1000)
                if response.status_code >= 400:
                    llm = f"翻译接口失败 {response.status_code}（{elapsed} ms）：{response.text[:200]}"
                else:
                    content = response.json()["choices"][0]["message"].get("content") or ""
                    llm = f"翻译 {elapsed} ms，文本：{str(content).strip() or '（空）'}"
            except Exception as exc:
                elapsed = int((time.perf_counter() - started) * 1000)
                llm = f"翻译接口失败（{elapsed} ms）：{exc}{_proxy_hint(proxy_url)}"
            self.test_finished.emit(stt + "\n" + llm)
        except Exception as exc:
            self.test_finished.emit(f"测试失败：{exc}")
        finally:
            if client is not None:
                await client.aclose()

    async def _shutdown_loop(self) -> None:
        self._shutdown = True
        self._listening = False
        self.capture.stop()
        if self._pcm_q is not None:
            try:
                self._pcm_q.put_nowait(None)
            except asyncio.QueueFull:
                pass


class App:
    def __init__(self) -> None:
        self.config = load_config()
        self.bar = ControlBar()
        self.subtitles = SubtitleWindow()
        self.engine = EngineThread(self.config)
        self._asr_ms: int | None = None
        self._ttft_ms: int | None = None
        self._hidden = False
        self._resume_listen = False
        self._dialog: SettingsDialog | None = None
        self.bar.scene_changed.connect(self._on_scene)
        self.bar.listen_toggled.connect(self.engine.toggle_listening)
        self.bar.reset_requested.connect(self.engine.reset_show)
        self.bar.lock_toggled.connect(self._on_lock)
        self.bar.settings_requested.connect(self._open_settings)
        self.bar.boss_pressed.connect(self._boss)
        self.bar.close_requested.connect(self.quit)
        self.engine.original_ready.connect(self._on_original)
        self.engine.translation_text.connect(self.subtitles.set_translation)
        self.engine.ttft_ready.connect(self._on_ttft)
        self.engine.status_changed.connect(self.bar.set_status)
        self.engine.listening_changed.connect(self.bar.set_listening)
        self.engine.cleared.connect(self.subtitles.clear_text)
        self.engine.test_finished.connect(self._on_test_finished)
        self._level_timer = QTimer()
        self._level_timer.timeout.connect(self._poll_level)
        self._level_timer.start(250)
        self.engine.start()

    def _poll_level(self) -> None:
        if not self.bar.is_listening():
            self.bar.set_level(None)
            return
        self.bar.set_level(self.engine.capture.level_peak)

    def show(self) -> None:
        self.subtitles.show()
        self.bar.show()
        self.bar.raise_()
        self.bar.activateWindow()

    def _on_scene(self, scene_id: str) -> None:
        self.bar.set_scene(scene_id)
        self.subtitles.clear_text()
        self._asr_ms = None
        self._ttft_ms = None
        self.bar.set_latency(None, None)
        self.engine.set_scene(scene_id)

    def _on_lock(self, locked: bool) -> None:
        self.subtitles.set_click_through(locked)
        self.bar.set_locked(locked)
        self.bar.raise_()
        self.bar.activateWindow()

    def _on_original(self, text: str, latency_ms: int) -> None:
        self.subtitles.set_original(text)
        self._asr_ms = latency_ms
        self._ttft_ms = None
        self.bar.set_latency(self._asr_ms, self._ttft_ms)

    def _on_ttft(self, latency_ms: int) -> None:
        self._ttft_ms = latency_ms
        self.bar.set_latency(self._asr_ms, self._ttft_ms)

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.config, self.bar)
        self._dialog = dialog
        dialog.test_requested.connect(self.engine.request_test)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            key, proxy = dialog.values()
            self.config.openrouter_api_key = key
            self.config.proxy_url = proxy
            try:
                save_config(self.config)
                self.engine.update_credentials(key, proxy)
                self.bar.set_status("设置已保存")
            except Exception as exc:
                self.bar.set_status(f"保存失败: {exc}")
        self._dialog = None

    def _on_test_finished(self, text: str) -> None:
        if self._dialog is not None and self._dialog.isVisible():
            self._dialog.show_test_result(text)

    def _boss(self) -> None:
        if not self._hidden:
            self._resume_listen = self.bar.is_listening()
            if self._resume_listen:
                self.engine.pause_listening()
            self.bar.hide()
            self.subtitles.hide()
            self._hidden = True
            return
        self.bar.show()
        self.subtitles.show()
        self._hidden = False
        if self._resume_listen:
            self.engine.start_listening()

    def quit(self) -> None:
        self.bar.unregister()
        self.engine.shutdown()
        if not self.engine.wait(4000):
            self.engine.terminate()
            self.engine.wait(1000)
        QApplication.quit()


def main() -> int:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = App()
    app.aboutToQuit.connect(window.bar.unregister)
    window.show()
    return app.exec()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
