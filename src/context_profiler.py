from __future__ import annotations

import asyncio

import httpx

from src.scene_engine import ScenePreset
from src.translator_llm import ConversationMemory


class ContextProfiler:
    def __init__(self, client: httpx.AsyncClient, memory: ConversationMemory) -> None:
        self._client = client
        self._memory = memory
        self._scene: ScenePreset | None = None
        self._pending: list[str] = []
        self._task: asyncio.Task[None] | None = None
        self._generation = 0

    def set_client(self, client: httpx.AsyncClient) -> None:
        self._client = client

    def set_scene(self, scene: ScenePreset) -> None:
        self._scene = scene
        self._generation += 1
        self._pending.clear()

    def reset(self) -> None:
        self._generation += 1
        self._pending.clear()
        self._memory.dynamic_context = ""

    def observe(self, original: str) -> None:
        try:
            text = (original or "").strip()
            if not text or self._scene is None or self._client is None:
                return
            self._pending.append(text)
            running = self._task is not None and not self._task.done()
            if len(self._pending) < 8 or running:
                return
            batch = self._pending[:8]
            del self._pending[:8]
            scene = self._scene
            generation = self._generation
            self._task = asyncio.create_task(self._run(batch, generation, scene))
        except Exception:
            return

    async def _run(self, batch: list[str], generation: int, scene: ScenePreset) -> None:
        try:
            payload = {
                "model": scene.profiler_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你只输出不超过50个汉字的背景锚点，不要翻译，不要引号。",
                    },
                    {"role": "user", "content": _user_prompt(scene.scene_id, batch)},
                ],
                "temperature": 0.2,
                "max_tokens": 80,
                "stream": False,
                "reasoning": {"enabled": False},
            }
            response = await self._client.post("/chat/completions", json=payload)
            if response.status_code >= 400 or generation != self._generation:
                return
            data = response.json()
            content = data["choices"][0]["message"].get("content") or ""
            if not isinstance(content, str):
                content = str(content)
            line = " ".join(content.replace("\n", " ").split()).strip("\"'“”")
            if len(line) > 50:
                line = line[:50]
            if generation == self._generation and line:
                self._memory.dynamic_context = line
        except Exception:
            return


def _user_prompt(scene_id: str, batch: list[str]) -> str:
    if scene_id in {"en_basketball", "en_football"}:
        guide = "根据下面的新台词，用不超过50个汉字更新背景锚点：当前对阵双方球队，以及刚出现的球员英文名到标准中文译名的映射。不要逐句翻译。"
    elif scene_id in {"ru_vlog", "ko_vlog"}:
        guide = "根据下面的新台词，用不超过50个汉字归纳博主当前所处场景和正在讨论的话题。不要逐句翻译。"
    else:
        guide = "根据下面的新台词，用不超过50个汉字归纳当前场景和正在讨论的话题。不要逐句翻译。"
    return guide + "\n" + "\n".join(batch)