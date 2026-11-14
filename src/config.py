from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


def _runtime_dirs() -> tuple[Path, Path]:
    if getattr(sys, "frozen", False):
        writable = Path(sys.executable).resolve().parent
        bundle = Path(getattr(sys, "_MEIPASS", writable))
        return writable, bundle
    root = Path(__file__).resolve().parent.parent
    return root, root


ROOT, BUNDLE_ROOT = _runtime_dirs()
CONFIG_PATH: Path = ROOT / "config.json"
ASSETS_DIR: Path = ROOT / "assets"
VAD_MODEL_PATH: Path = ASSETS_DIR / "silero_vad.onnx"
BUNDLED_VAD_PATH: Path = BUNDLE_ROOT / "assets" / "silero_vad.onnx"
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
VAD_URLS: list[str] = [
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad_16k_op15.onnx",
]


@dataclass
class AppConfig:
    openrouter_api_key: str = ""
    proxy_url: str = ""


def load_config() -> AppConfig:
    try:
        if not CONFIG_PATH.is_file():
            return AppConfig()
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return AppConfig()
        return AppConfig(
            openrouter_api_key=_as_text(data.get("openrouter_api_key", "")),
            proxy_url=_as_text(data.get("proxy_url", "")),
        )
    except Exception:
        return AppConfig()


def save_config(config: AppConfig) -> None:
    payload = {
        "openrouter_api_key": _as_text(config.openrouter_api_key),
        "proxy_url": _as_text(config.proxy_url),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".config-",
        suffix=".json",
        dir=str(CONFIG_PATH.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, CONFIG_PATH)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def ensure_vad_model(progress: Callable[[str], None] | None = None) -> Path:
    if _model_ready(VAD_MODEL_PATH):
        return VAD_MODEL_PATH
    if _install_bundled_vad():
        _notify(progress, "VAD 模型已就绪")
        return VAD_MODEL_PATH
    _notify(progress, "正在下载 VAD 模型…")
    try:
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"无法创建模型目录：{exc}") from exc
    last_error = "未知错误"
    for url in VAD_URLS:
        tmp_path: Path | None = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=".vad-",
                suffix=".onnx",
                dir=str(ASSETS_DIR),
            )
            os.close(fd)
            tmp_path = Path(tmp_name)
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "FastSceneSub"},
            )
            with urllib.request.urlopen(request, timeout=60) as response, tmp_path.open("wb") as out:
                while True:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            size = tmp_path.stat().st_size
            if size <= 100_000:
                raise RuntimeError(f"文件过小（{size} 字节）")
            os.replace(tmp_path, VAD_MODEL_PATH)
            tmp_path = None
            _notify(progress, "VAD 模型已就绪")
            return VAD_MODEL_PATH
        except Exception as exc:
            last_error = str(exc)
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
    raise RuntimeError(f"下载 VAD 模型失败：{last_error}")



def _model_ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 100_000
    except OSError:
        return False


def _install_bundled_vad() -> bool:
    source = BUNDLED_VAD_PATH
    if not _model_ready(source):
        return False
    try:
        if source.resolve() == VAD_MODEL_PATH.resolve() and _model_ready(VAD_MODEL_PATH):
            return True
    except OSError:
        return False
    try:
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    tmp_path: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=".vad-",
            suffix=".onnx",
            dir=str(ASSETS_DIR),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        shutil.copyfile(source, tmp_path)
        if not _model_ready(tmp_path):
            return False
        os.replace(tmp_path, VAD_MODEL_PATH)
        tmp_path = None
        return _model_ready(VAD_MODEL_PATH)
    except OSError:
        return False
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _notify(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is None:
        return
    try:
        progress(message)
    except Exception:
        return
