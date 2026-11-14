from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict

_ELONG = set("ー〜～—–-‐−")
_HALLUCINATIONS = (
    "thank you for watching",
    "thanks for watching",
    "subtitles by",
    "subtitle by",
    "please subscribe",
    "subscribe to",
    "субтитры сделал",
    "редактор субтитров",
    "продолжение следует",
    "подписывайтесь",
    "구독과 좋아요",
    "시청해 주셔서 감사합니다",
    "ご視聴ありがとうございました",
    "ご視聴ありがとう",
    "チャンネル登録",
    "字幕作成",
    "ご覧いただきありがとうございます",
    "ご清聴ありがとうございました",
    "字幕制作",
)

class PhraseCache:
    def __init__(self, maxsize: int = 256) -> None:
        self.maxsize = maxsize
        self._items: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> str | None:
        value = self._items.get(key)
        if value is None:
            return None
        self._items.move_to_end(key)
        return value

    def put(self, key: str, value: str) -> None:
        if key in self._items:
            self._items.move_to_end(key)
        self._items[key] = value
        while len(self._items) > self.maxsize:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()


def normalize_text(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    kept: list[str] = []
    for char in folded:
        if char.isspace() or char in _ELONG:
            continue
        if unicodedata.category(char).startswith("P"):
            continue
        kept.append(char)
    return "".join(kept)


_EXACT_HALLUCINATIONS = frozenset(
    normalize_text(item)
    for item in (
        "ありがとう",
        "ありがとうございます",
        "ありがとうございました",
        "どうもありがとう",
        "どうもありがとうございます",
        "どうもありがとうございました",
        "thank you",
        "thanks",
        "thank you very much",
    )
)

_ASSISTANT_REPLY = (
    "如果您有任何",
    "如果你有任何",
    "请随时告诉我",
    "请随时问我",
    "感谢您的感谢",
    "感谢你的感谢",
    "有什么可以帮",
    "作为人工智能",
    "我无法提供",
)



def is_hallucination(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return True
    normalized = normalize_text(raw)
    if not normalized:
        return True
    if normalized in _EXACT_HALLUCINATIONS:
        return True
    for phrase in _HALLUCINATIONS:
        folded = normalize_text(phrase)
        if folded and folded in normalized and len(normalized) <= len(folded) + 8:
            return True
    return False


def is_assistant_reply(text: str) -> bool:
    folded = normalize_text(text)
    if not folded:
        return False
    return any(normalize_text(phrase) in folded for phrase in _ASSISTANT_REPLY)


