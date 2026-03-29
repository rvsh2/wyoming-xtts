import sys
import types
import unittest
from unittest.mock import patch

from xtts_wyoming.synthesizer import XttsSynthesizer


class _FakeTensor:
    def __init__(self, device="cpu", shape=(1, 3)):
        self.device = device
        self.shape = shape

    def __getitem__(self, item):
        return self


class _FakeInference:
    def __init__(self):
        self.calls = []

    def generate(self, gpt_inputs, **kwargs):
        self.calls.append((gpt_inputs, kwargs))
        return _FakeGenerated()


class _FakeGenerated:
    def __getitem__(self, item):
        return "generated"


class _FakeGPT:
    def __init__(self):
        self.start_audio_token = 10
        self.stop_audio_token = 20
        self.max_gen_mel_tokens = 30
        self.gpt_inference = _FakeInference()

    def compute_embeddings(self, cond_latents, text_inputs):
        return _FakeTensor(device="cuda", shape=(1, 4))

    def generate(self, cond_latents, text_inputs, **hf_generate_kwargs):
        return "original"


class SynthesizerPatchTests(unittest.TestCase):
    def test_attention_mask_patch_adds_default_mask(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.long = "fake-long"

        def ones_like(tensor, dtype=None, device=None):
            return {
                "shape": tensor.shape,
                "dtype": dtype,
                "device": device,
            }

        fake_torch.ones_like = ones_like

        fake_gpt_module = types.ModuleType("TTS.tts.layers.xtts.gpt")
        fake_gpt_module.GPT = _FakeGPT

        with patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "TTS": types.ModuleType("TTS"),
                "TTS.tts": types.ModuleType("TTS.tts"),
                "TTS.tts.layers": types.ModuleType("TTS.tts.layers"),
                "TTS.tts.layers.xtts": types.ModuleType("TTS.tts.layers.xtts"),
                "TTS.tts.layers.xtts.gpt": fake_gpt_module,
            },
        ):
            XttsSynthesizer._patch_xtts_attention_mask()

            gpt = _FakeGPT()
            result = gpt.generate("cond", "text")

            self.assertEqual(result, "generated")
            _, kwargs = gpt.gpt_inference.calls[0]
            self.assertIn("attention_mask", kwargs)
            self.assertEqual(kwargs["attention_mask"]["shape"], (1, 4))
            self.assertEqual(kwargs["attention_mask"]["dtype"], "fake-long")
            self.assertEqual(kwargs["attention_mask"]["device"], "cuda")

    def test_voice_choices_use_only_configured_aliases(self):
        synthesizer = XttsSynthesizer(
            voice_speed_aliases={"normal": 1.0, "fast": 1.15},
        )

        self.assertEqual(
            synthesizer.voice_choices_for("Ana Florence"),
            [
                ("Ana Florence(1.00x)", 1.0, "normal"),
                ("Ana Florence(1.15x)", 1.15, "fast"),
            ],
        )

    def test_resolve_voice_and_speed_matches_generated_variant_name(self):
        synthesizer = XttsSynthesizer(
            voice_speed_aliases={"normal": 1.0, "fast": 1.15},
        )

        self.assertEqual(
            synthesizer.resolve_voice_and_speed("Ana Florence(1.15x)", None),
            ("Ana Florence", 1.15, "fast"),
        )


if __name__ == "__main__":
    unittest.main()
