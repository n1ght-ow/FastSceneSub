from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenePreset:
    scene_id: str
    label: str
    language: str
    speech_threshold: float
    min_silence_ms: int
    max_speech_s: float
    asr_prompt: str
    llm_model: str
    profiler_model: str
    system_prompt: str
    fallback_model: str = ""


_FLASH = "google/gemini-2.5-flash"
_FAST = "openai/gpt-4.1-mini"

SCENES: dict[str, ScenePreset] = {
    "en_basketball": ScenePreset(
        scene_id="en_basketball",
        label="🏀 英文篮球",
        language="en",
        speech_threshold=0.55,
        min_silence_ms=150,
        max_speech_s=1.25,
        asr_prompt="NBA live commentary vocabulary: pick and roll, transition, step-back three, isolation, alley-oop, turnover, foul trouble, shot clock, fast break, and-one, double team, rebound, assist.",
        llm_model=_FAST,
        profiler_model=_FLASH,
        system_prompt=(
            "你是资深中文篮球解说字幕译员。把英文解说译成短促有力、术语地道的中文口播，像现场解说，不要写成书面翻译。"
            "术语按这个口径：pick and roll=挡拆，transition=转换进攻，step-back three=后撤步三分，isolation=单打，"
            "alley-oop=空接或隔扣（看动作），turnover=失误，foul trouble=犯规麻烦，shot clock=进攻时限。"
            "严禁生硬直译，严禁解释、备注或双语对照，只输出最终中文译文。"
        ),
    ),
    "en_football": ScenePreset(
        scene_id="en_football",
        label="⚽ 英文足球",
        language="en",
        speech_threshold=0.55,
        min_silence_ms=150,
        max_speech_s=1.25,
        asr_prompt="Soccer live commentary vocabulary: through ball, offside trap, counter-attack, clinical finish, clean sheet, stoppage time, VAR check, box-to-box, penalty, set piece, hat-trick.",
        llm_model=_FAST,
        profiler_model=_FLASH,
        system_prompt=(
            "你是资深英超与欧冠中文解说字幕译员。译文要干脆利落。"
            "术语按这个口径：through ball=直塞，offside trap=反越位，counter-attack=快速反击，"
            "clinical finish=推射远角或冷静破门（看动作），clean sheet=零封，stoppage time=补时，"
            "VAR check=VAR，box-to-box=攻防两端。严禁解释和双语对照，只输出最终中文译文。"
        ),
    ),
    "ru_vlog": ScenePreset(
        scene_id="ru_vlog",
        label="🇷🇺 俄语Vlog",
        language="ru",
        speech_threshold=0.50,
        min_silence_ms=240,
        max_speech_s=2.2,
        asr_prompt="Разговорный влог: короче, вообще, жесть, типа, влог.",
        llm_model=_FLASH,
        profiler_model=_FLASH,
        system_prompt=(
            "你是精通俄语俚语和口头禅的年轻 Vlog 字幕组译者。короче、вообще、жесть、типа 这类口头禅要译成自然的中文语气，不要逐词硬译。"
            "结合最近几句上下文，把跳切的口语接顺。译文接地气，只输出中文译文，不要解释。"
        ),
    ),
    "ko_vlog": ScenePreset(
        scene_id="ko_vlog",
        label="🇰🇷 韩语Vlog",
        language="ko",
        speech_threshold=0.50,
        min_silence_ms=240,
        max_speech_s=2.2,
        asr_prompt="일상 브이로그 어휘: 진짜, 대박, 오늘, 우리, 완전, 솔직히, 맞아.",
        llm_model=_FLASH,
        profiler_model=_FLASH,
        system_prompt=(
            "你是精通韩语日常缩略语、敬语关系和网感表达的韩综与 Vlog 字幕组译者。"
            "根据上下文精准补全韩语省略的主语（我、你、他、她），不要把敬语关系翻错。"
            "语气灵动自然，只输出中文译文，不要解释。"
        ),
    ),
}

SCENE_ORDER: list[str] = [
    "en_basketball",
    "en_football",
    "ru_vlog",
    "ko_vlog",
]


def get_scene(scene_id: str) -> ScenePreset:
    try:
        return SCENES[scene_id]
    except KeyError as exc:
        raise KeyError(scene_id) from exc