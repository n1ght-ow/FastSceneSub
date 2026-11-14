from __future__ import annotations

import ctypes
from ctypes import wintypes

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.config import AppConfig
from src.scene_engine import SCENE_ORDER, get_scene

HOTKEY_ID = 0x5A11
MOD_ALT = 0x0001
VK_H = 0x48
WM_HOTKEY = 0x0312
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_NOACTIVATE = 0x08000000
HWND_TOPMOST = -1
SWP_FLAGS = 0x0001 | 0x0002 | 0x0004 | 0x0020
SWP_SHOW = 0x0001 | 0x0002 | 0x0040

_USER32_READY = False


def _user32():
    global _USER32_READY
    user32 = ctypes.windll.user32
    if not _USER32_READY:
        if hasattr(user32, "GetWindowLongPtrW"):
            user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
            user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
            user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
        user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
        user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        _USER32_READY = True
    return user32


def _window_styles() -> Qt.WindowType:
    return Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool


class ControlBar(QWidget):
    scene_changed = Signal(str)
    listen_toggled = Signal()
    reset_requested = Signal()
    lock_toggled = Signal(bool)
    settings_requested = Signal()
    boss_pressed = Signal()
    close_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._listening = False
        self._locked = False
        self._placed = False
        self._drag: QPoint | None = None
        self._hotkey_on = False
        self._hotkey_hwnd = 0
        self._buttons: dict[str, QPushButton] = {}
        self.setObjectName("controlBar")
        self.setWindowTitle("FastSceneSub")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet(
            """
            QWidget#controlBar {
                background: transparent;
            }
            QPushButton {
                color: #dddddd;
                background: rgba(255, 255, 255, 18);
                border: none;
                border-radius: 8px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QPushButton:checked {
                background: #f2f2f2;
                color: #111111;
                border-radius: 8px;
            }
            QLabel#latency { color: #9BE7A8; font-family: Consolas, monospace; font-size: 11px; }
            QLabel#status { color: #9a9a9a; font-size: 11px; }
            """
        )
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(4)
        group = QButtonGroup(self)
        group.setExclusive(True)
        for scene_id in SCENE_ORDER:
            button = QPushButton(get_scene(scene_id).label)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, sid=scene_id: self.scene_changed.emit(sid))
            group.addButton(button)
            self._buttons[scene_id] = button
            row.addWidget(button)
        self._buttons["en_basketball"].setChecked(True)
        self.listen_button = QPushButton("▶ 开始监听")
        self.listen_button.clicked.connect(self.listen_toggled.emit)
        self.reset_button = QPushButton("🔄 新片重置")
        self.reset_button.clicked.connect(self.reset_requested.emit)
        self.lock_button = QPushButton("🔒 锁定字幕穿透")
        self.lock_button.setCheckable(True)
        self.lock_button.toggled.connect(self.lock_toggled.emit)
        self.settings_button = QPushButton("⚙ 设置")
        self.settings_button.clicked.connect(self.settings_requested.emit)
        self.close_button = QPushButton("×")
        self.close_button.clicked.connect(self.close_requested.emit)
        self._asr_ms = None
        self._ttft_ms = None
        self._level_pct = None
        self.latency = QLabel("ASR: --- ms | 翻译首字: --- ms | 音量 --")
        self.latency.setObjectName("latency")
        self.status = QLabel("就绪")
        self.status.setObjectName("status")
        self.status.setMaximumWidth(220)
        for widget in (
            self.listen_button,
            self.reset_button,
            self.lock_button,
            self.settings_button,
            self.latency,
            self.status,
            self.close_button,
        ):
            row.addWidget(widget)
        self.adjustSize()

    def is_listening(self) -> bool:
        return self._listening

    def set_listening(self, listening: bool) -> None:
        self._listening = listening
        self.listen_button.setText("⏸ 暂停" if listening else "▶ 开始监听")

    def set_latency(self, asr_ms: int | None, ttft_ms: int | None) -> None:
        self._asr_ms = asr_ms
        self._ttft_ms = ttft_ms
        self._refresh_latency()

    def set_level(self, peak: float | None) -> None:
        self._level_pct = None if peak is None else min(99, int(min(1.0, max(0.0, peak)) * 100))
        self._refresh_latency()

    def _refresh_latency(self) -> None:
        asr = "---" if self._asr_ms is None else str(int(self._asr_ms))
        ttft = "---" if self._ttft_ms is None else str(int(self._ttft_ms))
        level = "--" if self._level_pct is None else f"{self._level_pct}%"
        self.latency.setText(f"ASR: {asr} ms | 翻译首字: {ttft} ms | 音量 {level}")

    def set_status(self, text: str) -> None:
        self.status.setText(text)
        self.status.setToolTip(text)

    def set_scene(self, scene_id: str) -> None:
        button = self._buttons.get(scene_id)
        if button is None:
            return
        button.blockSignals(True)
        button.setChecked(True)
        button.blockSignals(False)

    def set_locked(self, locked: bool) -> None:
        self._locked = locked
        self.lock_button.blockSignals(True)
        self.lock_button.setChecked(locked)
        self.lock_button.blockSignals(False)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if not self._placed:
            self._placed = True
            screen = self.screen() or self.windowHandle().screen()
            area = screen.availableGeometry()
            self.adjustSize()
            x = area.x() + max(8, (area.width() - self.width()) // 2)
            self.move(x, area.y() + 12)
        self._register_hotkey()
        QTimer.singleShot(0, self._enforce_clickable)

    def _enforce_clickable(self) -> None:
        hwnd = int(self.winId())
        user32 = _user32()
        if hasattr(user32, "GetWindowLongPtrW"):
            style = int(user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE) or 0)
            style = (style | WS_EX_LAYERED) & ~WS_EX_TRANSPARENT & ~WS_EX_NOACTIVATE
            user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style)
        user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_SHOW)
        self.raise_()
        self.activateWindow()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor(255, 255, 255, 36), 1))
        painter.setBrush(QColor(16, 16, 18, 242))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 12, 12)
        painter.end()

    def _register_hotkey(self) -> None:
        if self._hotkey_on:
            return
        hwnd = int(self.winId())
        user32 = _user32()
        user32.UnregisterHotKey(hwnd, HOTKEY_ID)
        if user32.RegisterHotKey(hwnd, HOTKEY_ID, MOD_ALT, VK_H):
            self._hotkey_on = True
            self._hotkey_hwnd = hwnd
        else:
            self.set_status("热键注册失败")

    def unregister(self) -> None:
        if not self._hotkey_on:
            return
        _user32().UnregisterHotKey(self._hotkey_hwnd, HOTKEY_ID)
        self._hotkey_on = False

    def nativeEvent(self, eventType, message):  # type: ignore[override]
        if not _is_windows_msg(eventType):
            return super().nativeEvent(eventType, message)
        try:
            msg = wintypes.MSG.from_address(int(message))
        except Exception:
            return super().nativeEvent(eventType, message)
        if msg.message == WM_HOTKEY and int(msg.wParam) == HOTKEY_ID:
            self.boss_pressed.emit()
            return True, 0
        return super().nativeEvent(eventType, message)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        child = self.childAt(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton and not isinstance(child, QPushButton):
            self._drag = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._drag is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        self._drag = None
        super().mouseReleaseEvent(event)


class SubtitleWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._original = ""
        self._translation = ""
        self._click_through = False
        self._placed = False
        self._drag: QPoint | None = None
        self.setWindowTitle("FastSceneSub 字幕")
        self.setWindowFlags(_window_styles() | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setFixedHeight(104)

    def set_original(self, text: str) -> None:
        self._original = text
        self.update()

    def set_translation(self, text: str) -> None:
        self._translation = text
        self.update()

    def clear_text(self) -> None:
        self._original = ""
        self._translation = ""
        self.update()

    def set_click_through(self, enabled: bool) -> None:
        self._click_through = enabled
        hwnd = int(self.winId())
        user32 = _user32()
        if hasattr(user32, "GetWindowLongPtrW"):
            style = int(user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE) or 0)
            if enabled:
                style |= WS_EX_LAYERED | WS_EX_TRANSPARENT
            else:
                style = (style | WS_EX_LAYERED) & ~WS_EX_TRANSPARENT
            user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style)
        else:
            style = int(user32.GetWindowLongW(hwnd, GWL_EXSTYLE) or 0)
            if enabled:
                style |= WS_EX_LAYERED | WS_EX_TRANSPARENT
            else:
                style = (style | WS_EX_LAYERED) & ~WS_EX_TRANSPARENT
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, SWP_FLAGS)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if self._placed:
            return
        self._placed = True
        screen = self.screen() or self.windowHandle().screen()
        area = screen.availableGeometry()
        width = min(1200, max(640, int(area.width() * 0.78)))
        self.setFixedWidth(width)
        x = area.x() + (area.width() - width) // 2
        y = area.bottom() - 72 - self.height()
        self.move(x, y)

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        width = self.width() - 24
        if width < 20:
            painter.end()
            return
        font1 = QFont("Microsoft YaHei UI")
        font1.setPixelSize(14)
        painter.setFont(font1)
        metrics1 = painter.fontMetrics()
        line1 = metrics1.elidedText(self._original, Qt.TextElideMode.ElideRight, width)
        x1 = 12 + max(0, (width - metrics1.horizontalAdvance(line1)) // 2)
        y1 = 10 + metrics1.ascent()
        if line1:
            painter.setPen(QColor(0, 0, 0, 180))
            painter.drawText(x1 + 1, y1 + 1, line1)
            painter.setPen(QColor(208, 208, 208))
            painter.drawText(x1, y1, line1)
        font2 = QFont("Microsoft YaHei UI")
        font2.setPixelSize(30)
        font2.setBold(True)
        metrics2 = QFontMetrics(font2)
        line2 = metrics2.elidedText(self._translation, Qt.TextElideMode.ElideRight, width)
        if line2:
            x2 = 12 + max(0, (width - metrics2.horizontalAdvance(line2)) // 2)
            y2 = y1 + metrics1.descent() + 12 + metrics2.ascent()
            path = QPainterPath()
            path.addText(float(x2), float(y2), font2, line2)
            pen = QPen(QColor(0, 0, 0, 230))
            pen.setWidthF(3.0)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.strokePath(path, pen)
            painter.fillPath(path, QColor(255, 255, 255))
        painter.end()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self._click_through:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._click_through or self._drag is None:
            return
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        self._drag = None
        super().mouseReleaseEvent(event)


class SettingsDialog(QDialog):
    test_requested = Signal(str, str)

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("FastSceneSub 设置")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setModal(True)
        self.key_edit = QLineEdit(config.openrouter_api_key)
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("sk-or-...")
        self.proxy_edit = QLineEdit(config.proxy_url)
        self.proxy_edit.setPlaceholderText("http://127.0.0.1:7897 或 socks5://127.0.0.1:7897")
        form = QFormLayout()
        form.addRow("OpenRouter API Key", self.key_edit)
        form.addRow("本地代理地址", self.proxy_edit)
        self.result = QLabel("")
        self.result.setWordWrap(True)
        self.result.setMinimumWidth(420)
        test = QPushButton("测试 STT 与翻译连通性")
        save = QPushButton("保存")
        cancel = QPushButton("取消")
        test.clicked.connect(self._emit_test)
        save.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addWidget(test)
        buttons.addStretch(1)
        buttons.addWidget(save)
        buttons.addWidget(cancel)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.result)
        layout.addLayout(buttons)

    def _emit_test(self) -> None:
        key, proxy = self.values()
        self.result.setText("正在测试…")
        self.test_requested.emit(key, proxy)

    def values(self) -> tuple[str, str]:
        return self.key_edit.text().strip(), self.proxy_edit.text().strip()

    def show_test_result(self, text: str) -> None:
        self.result.setText(text)


def _is_windows_msg(eventType: object) -> bool:
    if eventType in (b"windows_generic_MSG", "windows_generic_MSG"):
        return True
    try:
        raw = bytes(eventType)  # type: ignore[arg-type]
    except Exception:
        return False
    return raw == b"windows_generic_MSG"