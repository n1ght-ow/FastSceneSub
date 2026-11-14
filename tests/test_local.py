from __future__ import annotations

import base64
import io
import sys
import time
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.filters_and_cache import (  # noqa: E402
    PhraseCache,
    is_assistant_reply,
    is_hallucination,
    normalize_text,
)
from src.scene_engine import SCENE_ORDER, get_scene  # noqa: E402
from src.translator_llm import ConversationMemory  # noqa: E402
from src.vad_chunker import VadChunker, VadParams  # noqa: E402
from src.asr_openrouter import pcm16k_to_flac_b64  # noqa: E402
import soundfile as sf  # noqa: E402


def scripted(params: VadParams, prob) -> VadChunker:
    chunker = VadChunker.__new__(VadChunker)
    chunker.params = params
    chunker.reset()
    chunker._speech_prob = prob
    return chunker


class LocalTests(unittest.TestCase):

    def test_runtime_dirs(self) -> None:
        import src.config as config

        writable, bundle = config._runtime_dirs()
        self.assertEqual(writable, ROOT)
        self.assertEqual(bundle, ROOT)
        self.assertEqual(config.CONFIG_PATH, ROOT / "config.json")
        with (
            unittest.mock.patch.object(config.sys, "frozen", True, create=True),
            unittest.mock.patch.object(config.sys, "executable", r"C:\Apps\FastSceneSub.exe"),
            unittest.mock.patch.object(config.sys, "_MEIPASS", r"C:\Temp\mei", create=True),
        ):
            frozen_writable, frozen_bundle = config._runtime_dirs()
        self.assertEqual(frozen_writable, Path(r"C:\Apps"))
        self.assertEqual(frozen_bundle, Path(r"C:\Temp\mei"))

    def test_scenes(self) -> None:
        self.assertEqual(
            SCENE_ORDER,
            ["en_basketball", "en_football", "ru_vlog", "ko_vlog"],
        )
        ball = get_scene("en_basketball")
        foot = get_scene("en_football")
        self.assertEqual((ball.min_silence_ms, ball.max_speech_s, ball.speech_threshold), (150, 1.25, 0.55))
        self.assertEqual((foot.min_silence_ms, foot.max_speech_s, foot.speech_threshold), (150, 1.25, 0.55))
        self.assertEqual(get_scene("ru_vlog").min_silence_ms, 240)
        self.assertEqual(get_scene("ko_vlog").max_speech_s, 2.2)
        self.assertEqual(ball.llm_model, "openai/gpt-4.1-mini")
        self.assertEqual(foot.llm_model, "openai/gpt-4.1-mini")
        self.assertIn("pick and roll", ball.asr_prompt)
        self.assertIn("through ball", foot.asr_prompt)
        self.assertIn("короче", get_scene("ru_vlog").asr_prompt)
        with self.assertRaises(KeyError):
            get_scene("missing")

    def test_filters(self) -> None:
        self.assertTrue(is_hallucination("  "))
        self.assertTrue(is_hallucination("Thank you for watching!"))
        self.assertTrue(is_hallucination("Subtitles by someone"))
        self.assertTrue(is_hallucination("Субтитры сделал Иван"))
        self.assertTrue(is_hallucination("ご視聴ありがとうございました"))
        self.assertTrue(is_hallucination("ありがとうございました"))
        self.assertTrue(is_hallucination("ありがとうございます！"))
        self.assertFalse(is_hallucination("ありがとう、先生、気持ちいいよ"))
        self.assertTrue(is_assistant_reply("感谢您的感谢。如果您有任何其他问题，请随时告诉我。"))
        self.assertFalse(is_assistant_reply("谢谢你，真的很舒服"))
        self.assertTrue(is_hallucination("구독과 좋아요"))
        self.assertFalse(is_hallucination("He said thank you for watching the game and then called a timeout play"))
        self.assertEqual(normalize_text("ちょっと待ってー！"), normalize_text("ちょっと待って"))

    def test_cache_and_memory(self) -> None:
        cache = PhraseCache(maxsize=2)
        cache.put("a", "1")
        cache.put("b", "2")
        self.assertEqual(cache.get("a"), "1")
        cache.put("c", "3")
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), "1")
        memory = ConversationMemory(max_pairs=2, idle_seconds=0.05)
        memory.dynamic_context = "锚点"
        memory.add("a", "甲")
        memory.add("b", "乙")
        memory.add("c", "丙")
        self.assertNotIn("原文: a", memory.render_history())
        self.assertIn("原文: c", memory.render_history())
        time.sleep(0.06)
        self.assertTrue(memory.maybe_expire())
        self.assertEqual(memory.render_history(), "")
        self.assertEqual(memory.dynamic_context, "锚点")
        self.assertFalse(memory.maybe_expire())

    def test_flac_roundtrip(self) -> None:
        samples = np.linspace(-0.4, 0.4, 8000, dtype=np.float32)
        encoded = pcm16k_to_flac_b64(samples)
        self.assertNotIn("data:", encoded)
        raw = base64.b64decode(encoded)
        audio, rate = sf.read(io.BytesIO(raw), always_2d=False)
        self.assertEqual(rate, 16000)
        self.assertEqual(np.asarray(audio).ndim, 1)
        self.assertGreater(len(audio), 7000)

    def test_vad_rules(self) -> None:
        def prob(frame: np.ndarray) -> float:
            return 0.9 if float(frame[0]) >= 0.5 else 0.1

        params = VadParams(speech_threshold=0.5, min_silence_ms=64, max_speech_s=10.0)
        chunker = scripted(params, prob)
        silence = np.zeros(512, dtype=np.float32)
        soft = np.full(512, 0.3, dtype=np.float32)
        loud = np.full(512, 0.8, dtype=np.float32)
        audio = np.concatenate([soft] * 4 + [loud] * 12 + [silence] * 2)
        emitted = chunker.push(audio)
        self.assertEqual(len(emitted), 1)
        self.assertAlmostEqual(float(emitted[0][0]), 0.3, places=3)
        self.assertAlmostEqual(float(emitted[0][2048]), 0.8, places=3)
        self.assertGreaterEqual(emitted[0].size, 5600)

        short = scripted(params, prob)
        self.assertEqual(short.push(np.concatenate([loud] * 5 + [silence] * 2)), [])

        quiet = scripted(params, lambda _frame: 0.99)
        self.assertEqual(quiet.push(np.concatenate([silence] * 14)), [])

        forced = scripted(VadParams(0.5, 5000, 1.0), lambda _frame: 0.99)
        pieces = forced.push(np.concatenate([loud] * 32))
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0].size, 16384 - 3200)
        self.assertEqual(forced._speech_samples, 3200)
        self.assertTrue(forced._in_speech)
        forced.set_params(VadParams(0.5, 5000, 1.0))
        self.assertFalse(forced._in_speech)


if __name__ == "__main__":
    unittest.main()