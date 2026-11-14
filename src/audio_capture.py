from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

_TARGET_RATE = 16000


class AudioCapture:
    def __init__(
        self,
        on_audio: Callable[[np.ndarray], None],
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self._on_audio = on_audio
        self._on_status = on_status
        self._running = False
        self._stop = threading.Event()
        self._life_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._clock_lock = threading.Lock()
        self._emit_lock = threading.Lock()
        self._pa: Any = None
        self._pyaudio: Any = None
        self._soxr: Any = None
        self._stream: Any = None
        self._resampler: Any = None
        self._in_rate: int | None = None
        self._channels = 1
        self._format: int | None = None
        self._rate = _TARGET_RATE
        self._device_key: tuple[int, str, int] | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None
        self._last_real = time.monotonic()
        self._injected_samples = 0
        self._gain = 1.0
        self._quiet_chunks = 0
        self.raw_peak = 0.0
        self.level_peak = 0.0
        self._held_peak = 0.0
        self._held_until = 0.0

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        with self._life_lock:
            if self._running:
                return
            self._stop.clear()
            try:
                self._pyaudio, self._soxr = _load_libraries()
                self._pa = self._pyaudio.PyAudio()
                device = self._select_loopback(self._pa)
                self._running = True
                self._open_and_start(device)
                with self._clock_lock:
                    self._last_real = time.monotonic()
                    self._injected_samples = 0
                self._ensure_thread("_heartbeat_thread", self._heartbeat_loop, "audio-heartbeat")
                self._ensure_thread("_monitor_thread", self._monitor_loop, "audio-monitor")
            except Exception as exc:
                self._running = False
                self._stop.set()
                self._shutdown_audio()
                self._status(f"音频采集启动失败：{exc}")

    def stop(self) -> None:
        with self._life_lock:
            self._running = False
            self._stop.set()
            self._shutdown_audio()
        self._join_worker(self._heartbeat_thread)
        self._join_worker(self._monitor_thread)

    def _ensure_thread(self, attr: str, target: Callable[[], None], name: str) -> None:
        current = getattr(self, attr)
        if isinstance(current, threading.Thread) and current.is_alive():
            return
        thread = threading.Thread(target=target, name=name, daemon=True)
        setattr(self, attr, thread)
        thread.start()

    def _join_worker(self, thread: threading.Thread | None) -> None:
        if thread is None or not thread.is_alive() or thread is threading.current_thread():
            return
        thread.join(timeout=2.0)

    def _shutdown_audio(self) -> None:
        self._close_stream()
        pa = self._pa
        self._pa = None
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass
        self._resampler = None
        self._in_rate = None

    def _close_stream(self) -> None:
        with self._io_lock:
            stream = self._stream
            self._stream = None
        if stream is None:
            return
        try:
            stream.stop_stream()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            return

    def _open_and_start(self, device: dict[str, Any]) -> None:
        stream, rate, key, channels, fmt = self._build_stream(device)
        with self._io_lock:
            self._channels = channels
            self._format = fmt
            self._rate = rate
            self._stream = stream
            if self._resampler is None or self._in_rate != rate:
                self._resampler = self._soxr.ResampleStream(
                    float(rate),
                    float(_TARGET_RATE),
                    1,
                    dtype="float32",
                    quality="HQ",
                )
                self._in_rate = rate
        try:
            stream.start_stream()
        except Exception:
            with self._io_lock:
                if self._stream is stream:
                    self._stream = None
            try:
                stream.close()
            except Exception:
                pass
            raise
        self._device_key = key

    def _build_stream(self, device: dict[str, Any]) -> tuple[Any, int, tuple[int, str, int], int, int]:
        channels = int(device.get("maxInputChannels") or 0)
        if channels <= 0:
            channels = int(device.get("maxOutputChannels") or 0)
        channels = max(1, channels)
        try:
            rate = int(device["defaultSampleRate"])
            index = int(device["index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"loopback 设备信息无效：{exc}") from exc
        if rate <= 0:
            raise RuntimeError(f"设备采样率无效：{rate}")
        frames = max(256, int(rate * 0.02))
        name = str(device.get("name", ""))
        pa = self._pyaudio
        common: dict[str, Any] = {
            "channels": channels,
            "rate": rate,
            "input": True,
            "input_device_index": index,
            "frames_per_buffer": frames,
            "stream_callback": self._on_callback,
            "start": False,
        }
        try:
            stream = self._pa.open(format=pa.paFloat32, **common)
            fmt = int(pa.paFloat32)
        except Exception:
            stream = self._pa.open(format=pa.paInt16, **common)
            fmt = int(pa.paInt16)
        return stream, rate, (index, name, rate), channels, fmt

    def _select_loopback(self, pa: Any) -> dict[str, Any]:
        getter = getattr(pa, "get_default_wasapi_loopback", None)
        if callable(getter):
            try:
                info = getter()
                if isinstance(info, dict) and "index" in info:
                    return info
            except Exception:
                pass
        default_name = ""
        try:
            default_out = pa.get_default_output_device_info()
            if isinstance(default_out, dict):
                default_name = str(default_out.get("name", ""))
        except Exception:
            default_name = ""
        generator = getattr(pa, "get_loopback_device_info_generator", None)
        if not callable(generator):
            raise RuntimeError("当前 PyAudio 不支持 WASAPI loopback")
        loopbacks: list[dict[str, Any]] = []
        try:
            for info in generator():
                if isinstance(info, dict):
                    loopbacks.append(info)
        except Exception as exc:
            raise RuntimeError(f"枚举 loopback 设备失败：{exc}") from exc
        if not loopbacks:
            raise RuntimeError("没有可用的 WASAPI loopback 设备")
        if default_name:
            for info in loopbacks:
                if default_name in str(info.get("name", "")):
                    return info
        return loopbacks[0]

    def _device_identity(self, device: dict[str, Any]) -> tuple[int, str, int]:
        return (
            int(device["index"]),
            str(device.get("name", "")),
            int(device["defaultSampleRate"]),
        )

    def _on_callback(
        self,
        in_data: bytes | None,
        frame_count: int,
        time_info: dict[str, Any],
        status: int,
    ) -> tuple[None, int]:
        del frame_count, time_info, status
        pa = self._pyaudio
        complete = int(pa.paComplete) if pa is not None else 1
        cont = int(pa.paContinue) if pa is not None else 0
        if self._stop.is_set() or not self._running:
            return (None, complete)
        try:
            samples = self._decode(in_data)
            if samples.size:
                with self._clock_lock:
                    self._last_real = time.monotonic()
                    self._injected_samples = 0
                self._resample_and_emit(samples)
        except Exception as exc:
            self._status(f"音频回调异常：{exc}")
        if self._stop.is_set() or not self._running:
            return (None, complete)
        return (None, cont)

    def _decode(self, in_data: bytes | None) -> np.ndarray:
        if not in_data:
            return np.zeros(0, dtype=np.float32)
        # 立刻复制，PortAudio 会复用这块缓冲。
        raw = bytes(in_data)
        pa = self._pyaudio
        if pa is not None and self._format == pa.paFloat32:
            audio = np.frombuffer(raw, dtype=np.float32).copy()
        else:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            audio *= np.float32(1.0 / 32768.0)
        channels = self._channels
        if channels > 1:
            usable = audio.size - (audio.size % channels)
            if usable <= 0:
                return np.zeros(0, dtype=np.float32)
            audio = audio[:usable].reshape(-1, channels).mean(axis=1, dtype=np.float32)
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _resample_and_emit(self, mono: np.ndarray) -> None:
        with self._io_lock:
            rate = self._rate
            if self._resampler is None or self._in_rate != rate:
                if self._soxr is None:
                    raise RuntimeError("soxr 不可用")
                # 只在采样率变化时重建，避免滤镜状态被无故清掉。
                self._resampler = self._soxr.ResampleStream(
                    float(rate),
                    float(_TARGET_RATE),
                    1,
                    dtype="float32",
                    quality="HQ",
                )
                self._in_rate = rate
            resampler = self._resampler
        chunk = np.ascontiguousarray(mono, dtype=np.float32).reshape(-1)
        output = resampler.resample_chunk(chunk)
        audio = np.asarray(output, dtype=np.float32).reshape(-1)
        self._emit(audio)

    def _emit(self, samples: np.ndarray) -> None:
        audio = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).reshape(-1)).copy()
        if audio.size == 0:
            return
        self.raw_peak = float(np.max(np.abs(audio)))
        audio = self._normalize(audio)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        now = time.monotonic()
        if peak >= self._held_peak or now >= self._held_until:
            self._held_peak = peak
            self._held_until = now + 0.4
        self.level_peak = self._held_peak
        try:
            with self._emit_lock:
                if not self._running:
                    return
                self._on_audio(audio)
        except Exception as exc:
            self._status(f"音频回调异常：{exc}")

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        # 系统音量很低时，原始峰值只有 0.01 左右，VAD 会当成静音丢掉。
        rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)) + 1e-12)
        if rms < 0.0008:
            self._quiet_chunks += 1
            if self._quiet_chunks > 25:
                self._gain = max(1.0, self._gain * 0.97)
            if self._gain <= 1.05:
                return audio
        else:
            self._quiet_chunks = 0
            desired = min(24.0, 0.055 / rms)
            self._gain = (0.7 * self._gain) + (0.3 * desired)
        return np.clip(audio * np.float32(self._gain), -0.98, 0.98).astype(np.float32, copy=False)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(0.05):
            if not self._running:
                continue
            try:
                self._inject_silence_if_stalled()
            except Exception as exc:
                self._status(f"静音心跳异常：{exc}")

    def _inject_silence_if_stalled(self) -> None:
        with self._clock_lock:
            if not self._running:
                return
            gap = time.monotonic() - self._last_real
            if gap <= 0.120:
                return
            due = int(gap * _TARGET_RATE) - self._injected_samples
            count = min(_TARGET_RATE, due)
            if count <= 0:
                return
            self._injected_samples += count
        # 系统静音时 WASAPI 回调会停。直接补 16kHz 零，不经过重采样器。
        self._emit(np.zeros(count, dtype=np.float32))

    def _monitor_loop(self) -> None:
        while not self._stop.wait(1.0):
            if not self._running:
                continue
            try:
                with self._life_lock:
                    if not self._running or self._pa is None:
                        continue
                    device = self._select_loopback(self._pa)
                    if self._device_identity(device) == self._device_key:
                        continue
                    self._status("播放设备已切换，正在重连")
                    self._close_stream()
                    self._open_and_start(device)
            except Exception as exc:
                self._status(f"播放设备重连失败：{exc}")

    def _status(self, message: str) -> None:
        if self._on_status is None:
            return
        try:
            self._on_status(message)
        except Exception:
            return


def _load_libraries() -> tuple[Any, Any]:
    import pyaudiowpatch as pyaudio
    import soxr

    return pyaudio, soxr
