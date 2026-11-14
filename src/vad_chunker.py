from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort


@dataclass
class VadParams:
    speech_threshold: float
    min_silence_ms: int
    max_speech_s: float


class _PreRoll:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._data = np.zeros(capacity, dtype=np.float32)
        self._write = 0
        self._filled = 0

    def append(self, samples: np.ndarray) -> None:
        incoming = np.asarray(samples, dtype=np.float32).reshape(-1)
        count = int(incoming.size)
        if count <= 0:
            return
        if count >= self.capacity:
            self._data[:] = incoming[-self.capacity :]
            self._write = 0
            self._filled = self.capacity
            return
        first = min(count, self.capacity - self._write)
        self._data[self._write : self._write + first] = incoming[:first]
        rest = count - first
        if rest:
            self._data[:rest] = incoming[first:]
        self._write = (self._write + count) % self.capacity
        self._filled = min(self.capacity, self._filled + count)

    def snapshot(self) -> np.ndarray:
        if self._filled <= 0:
            return np.zeros(0, dtype=np.float32)
        if self._filled < self.capacity:
            start = self._write - self._filled
            if start >= 0:
                return self._data[start : self._write].copy()
        return np.concatenate((self._data[self._write :], self._data[: self._write])).copy()


class VadChunker:
    FRAME: int = 512
    SAMPLE_RATE: int = 16000
    MIN_SPEECH_SAMPLES: int = 5600
    OVERLAP_SAMPLES: int = 3200
    LOW_RMS: float = 0.008
    PRE_ROLL_SAMPLES: int = 4000
    TAIL_SILENCE_SAMPLES: int = 1600
    MIN_PEAK: float = 0.01

    def __init__(self, model_path: str | Path, params: VadParams) -> None:
        self.params = params
        self._init_model(Path(model_path))
        self.reset()

    def _init_model(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise RuntimeError(f"VAD 模型不存在：{model_path}")
        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(f"加载 VAD 模型失败：{exc}") from exc
        inputs = list(self._session.get_inputs())
        outputs = list(self._session.get_outputs())
        by_name = {item.name: item for item in inputs}
        if "input" not in by_name:
            raise RuntimeError("VAD 模型缺少 input")
        self._input_meta = by_name
        self._output_names = [item.name for item in outputs]
        audio_shape = list(by_name["input"].shape or [])
        last_dim = audio_shape[-1] if audio_shape else None
        self._use_context = not (isinstance(last_dim, int) and last_dim == 512)
        self._audio_name = "input"
        self._sr_name: str | None = "sr" if "sr" in by_name else None
        state_names: list[str] = []
        for name in ("state", "h", "h0", "c", "c0"):
            if name in by_name:
                state_names.append(name)
        for item in inputs:
            if item.name not in ("input", "sr") and item.name not in state_names:
                state_names.append(item.name)
        self._states: dict[str, np.ndarray] = {
            name: self._state_zeros(by_name[name]) for name in state_names
        }
        self._context = np.zeros((1, 64), dtype=np.float32)

    def _state_zeros(self, meta: object) -> np.ndarray:
        shape = list(getattr(meta, "shape", []) or [])
        last = shape[-1] if shape else None
        hidden = 64 if isinstance(last, (int, np.integer)) and int(last) == 64 else 128
        concrete: list[int] = []
        for dim in shape:
            if isinstance(dim, (int, np.integer)) and int(dim) > 0:
                concrete.append(int(dim))
            else:
                concrete = []
                break
        if len(concrete) == 3:
            return np.zeros(concrete, dtype=np.float32)
        return np.zeros((2, 1, hidden), dtype=np.float32)

    def _speech_prob(self, frame: np.ndarray) -> float:
        samples = np.asarray(frame, dtype=np.float32).reshape(-1)
        if int(samples.size) != self.FRAME:
            raise RuntimeError(f"VAD 帧长度应为 {self.FRAME}，输入形状 {tuple(int(n) for n in samples.shape)}")
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if self._use_context:
            context = np.asarray(self._context, dtype=np.float32).reshape(1, -1)
            audio = np.concatenate((context, samples.reshape(1, -1)), axis=1)
        else:
            audio = samples.reshape(1, -1)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        feeds: dict[str, np.ndarray] = {self._audio_name: audio}
        if self._sr_name is not None:
            feeds[self._sr_name] = np.array(self.SAMPLE_RATE, dtype=np.int64)
        for name, value in self._states.items():
            feeds[name] = value
        try:
            outputs = self._session.run(None, feeds)
        except Exception as exc:
            shape = tuple(int(dim) for dim in audio.shape)
            raise RuntimeError(f"VAD 推理失败，输入形状 {shape}：{exc}") from exc
        self._apply_state_outputs(list(outputs))
        tail = audio[:, -64:] if audio.shape[-1] >= 64 else audio
        self._context = np.zeros((1, 64), dtype=np.float32)
        self._context[:, -tail.shape[-1] :] = np.asarray(tail, dtype=np.float32)
        flat = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if flat.size == 0:
            shape = tuple(int(dim) for dim in audio.shape)
            raise RuntimeError(f"VAD 输出为空，输入形状 {shape}")
        prob = float(flat[0])
        if prob < 0.0:
            return 0.0
        if prob > 1.0:
            return 1.0
        return prob

    def _apply_state_outputs(self, outputs: list[object]) -> None:
        index = {name: pos for pos, name in enumerate(self._output_names)}
        used: set[str] = set()
        named = {
            "hn": ("h", "h0"),
            "cn": ("c", "c0"),
            "stateN": ("state",),
        }
        for out_name, targets in named.items():
            idx = index.get(out_name)
            if idx is None or idx >= len(outputs):
                continue
            for target in targets:
                if target in self._states:
                    self._states[target] = np.array(outputs[idx], dtype=np.float32, copy=True)
                    used.add(target)
                    break
        if "h" in self._states and "c" in self._states and len(outputs) >= 3:
            if "h" not in used:
                self._states["h"] = np.array(outputs[1], dtype=np.float32, copy=True)
                used.add("h")
            if "c" not in used:
                self._states["c"] = np.array(outputs[2], dtype=np.float32, copy=True)
                used.add("c")
        if len(outputs) >= 2:
            for target in ("state", "h", "h0"):
                if target in self._states and target not in used:
                    self._states[target] = np.array(outputs[1], dtype=np.float32, copy=True)
                    break

    def set_params(self, params: VadParams) -> None:
        self.params = params
        # 只丢掉当前半句，LSTM 和 pre-roll 继续留着。
        self._reset_utterance()

    def reset(self) -> None:
        self._remainder = np.zeros(0, dtype=np.float32)
        self._pre_roll = _PreRoll(self.PRE_ROLL_SAMPLES)
        self._reset_utterance()
        states = getattr(self, "_states", None)
        if isinstance(states, dict):
            for key, value in list(states.items()):
                if isinstance(value, np.ndarray):
                    states[key] = np.zeros_like(value)
        context = getattr(self, "_context", None)
        if isinstance(context, np.ndarray):
            self._context = np.zeros_like(context)

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        incoming = np.asarray(samples, dtype=np.float32).reshape(-1)
        if incoming.size == 0 and self._remainder.size == 0:
            return []
        if self._remainder.size:
            data = np.concatenate((self._remainder, incoming))
        else:
            data = np.ascontiguousarray(incoming, dtype=np.float32)
        total = int(data.size)
        usable = total - (total % self.FRAME)
        if usable == 0:
            self._remainder = np.ascontiguousarray(data, dtype=np.float32).copy()
            return []
        if usable == total:
            self._remainder = np.zeros(0, dtype=np.float32)
        else:
            self._remainder = np.ascontiguousarray(data[usable:], dtype=np.float32).copy()
        emitted: list[np.ndarray] = []
        for offset in range(0, usable, self.FRAME):
            frame = np.ascontiguousarray(data[offset : offset + self.FRAME], dtype=np.float32).copy()
            emitted.extend(self._consume(frame))
        return emitted

    def flush(self) -> list[np.ndarray]:
        if not getattr(self, "_in_speech", False):
            return []
        audio = self._trim_tail(self._collect())
        chunk = self._emit_if_valid(audio)
        self._reset_utterance()
        if chunk is None:
            return []
        return [chunk]

    def _consume(self, frame: np.ndarray) -> list[np.ndarray]:
        prob = float(self._speech_prob(frame))
        emitted: list[np.ndarray] = []
        threshold = float(self.params.speech_threshold)
        if not self._in_speech:
            if prob >= threshold:
                # 句首只带本帧之前的 pre-roll。
                pre = self._pre_roll.snapshot()
                self._speech = []
                if pre.size:
                    self._speech.append(pre)
                self._speech.append(frame.copy())
                self._speech_samples = int(pre.size + frame.size)
                self._silence_samples = 0
                self._in_speech = True
                emitted.extend(self._close_if_needed())
        else:
            self._speech.append(frame.copy())
            self._speech_samples += int(frame.size)
            if prob < threshold:
                self._silence_samples += self.FRAME
            else:
                self._silence_samples = 0
            emitted.extend(self._close_if_needed())
        self._pre_roll.append(frame)
        return emitted

    def _close_if_needed(self) -> list[np.ndarray]:
        if not self._in_speech:
            return []
        silence_limit = int(self.params.min_silence_ms * self.SAMPLE_RATE / 1000)
        if self._silence_samples > 0 and self._silence_samples >= silence_limit:
            audio = self._trim_tail(self._collect())
            chunk = self._emit_if_valid(audio)
            self._reset_utterance()
            return [] if chunk is None else [chunk]
        audio = self._collect()
        self._speech_samples = int(audio.size)
        max_samples = int(float(self.params.max_speech_s) * self.SAMPLE_RATE)
        if self._speech_samples < max_samples or audio.size <= self.OVERLAP_SAMPLES:
            return []
        head = audio[: audio.size - self.OVERLAP_SAMPLES]
        tail = np.ascontiguousarray(audio[-self.OVERLAP_SAMPLES :], dtype=np.float32).copy()
        self._speech = [tail]
        self._speech_samples = int(tail.size)
        self._silence_samples = 0
        self._in_speech = True
        chunk = self._emit_if_valid(head)
        return [] if chunk is None else [chunk]

    def _trim_tail(self, audio: np.ndarray) -> np.ndarray:
        extra = self._silence_samples - self.TAIL_SILENCE_SAMPLES
        if extra > 0 and int(audio.size) > extra:
            return np.ascontiguousarray(audio[:-extra], dtype=np.float32)
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def _collect(self) -> np.ndarray:
        parts = getattr(self, "_speech", [])
        if not parts:
            return np.zeros(0, dtype=np.float32)
        if len(parts) == 1:
            return np.asarray(parts[0], dtype=np.float32).reshape(-1)
        arrays = [np.asarray(part, dtype=np.float32).reshape(-1) for part in parts]
        return np.concatenate(arrays)

    def _emit_if_valid(self, audio: np.ndarray) -> np.ndarray | None:
        data = np.asarray(audio, dtype=np.float32).reshape(-1)
        if int(data.size) < self.MIN_SPEECH_SAMPLES:
            return None
        data = np.ascontiguousarray(data, dtype=np.float32)
        peak = float(np.max(np.abs(data)))
        if peak < self.MIN_PEAK:
            return None
        rms = float(np.sqrt(np.mean(data * data)))
        if rms < self.LOW_RMS:
            return None
        return data.copy()

    def _reset_utterance(self) -> None:
        self._in_speech = False
        self._speech = []
        self._silence_samples = 0
        self._speech_samples = 0
