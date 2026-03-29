"""Shared XTTS-v2 synthesis runtime."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

from .speaker_store import SpeakerStore


LOGGER = logging.getLogger("xtts-wyoming.synthesizer")

SUPPORTED_LANGUAGES = {
    "ar",
    "cs",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "hu",
    "it",
    "ja",
    "ko",
    "nl",
    "pl",
    "pt",
    "ru",
    "tr",
    "zh-cn",
}


@dataclass
class SynthesisResult:
    audio: list[float]
    sample_rate: int
    language: str
    speaker: str | None
    processing_time: float

    def asdict(self) -> dict:
        payload = asdict(self)
        payload["audio_samples"] = int(len(self.audio))
        del payload["audio"]
        return payload


class XttsSynthesizer:
    def __init__(
        self,
        *,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        default_language: str = "en",
        default_voice: str | None = None,
        speaker_dir: str = "/data/speakers",
        model_dir: str | None = None,
        device: str | None = None,
        temperature: float = 0.65,
        top_k: int = 50,
        top_p: float = 0.8,
        speed: float = 1.0,
        voice_speed_aliases: Optional[dict[str, float]] = None,
        length_penalty: float = 1.0,
        repetition_penalty: float = 2.0,
        enable_text_splitting: bool = True,
    ) -> None:
        self.model_name = model_name
        self.default_language = self.resolve_language(default_language)
        self.default_voice = default_voice
        self.speaker_store = SpeakerStore(speaker_dir)
        self.model_dir = model_dir
        self.device = device or os.getenv("XTTS_DEVICE")
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.speed = speed
        self.voice_speed_aliases = {
            key.strip().lower(): float(value)
            for key, value in (voice_speed_aliases or {}).items()
            if key and float(value) > 0
        }
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.enable_text_splitting = enable_text_splitting
        self.backend = "coqui-tts"
        self._tts = None
        self._builtin_voice_names: list[str] = []

    @staticmethod
    def parse_voice_speed_aliases(raw_value: Optional[str]) -> dict[str, float]:
        aliases: dict[str, float] = {}
        if not raw_value:
            return aliases

        for item in raw_value.split(","):
            item = item.strip()
            if not item or "=" not in item:
                continue

            name, value = item.split("=", 1)
            name = name.strip().lower()
            if not name:
                continue

            try:
                speed = float(value.strip())
            except ValueError:
                continue

            if speed > 0:
                aliases[name] = speed

        return aliases

    @staticmethod
    def format_voice_speed(speed: float) -> str:
        return f"{speed:.2f}x"

    def _voice_choice_name(self, voice_name: str, speed: float) -> str:
        return f"{voice_name}({self.format_voice_speed(speed)})"

    def voice_choices_for(self, voice_name: str) -> list[tuple[str, Optional[float], Optional[str]]]:
        if not self.voice_speed_aliases:
            return [(voice_name, None, None)]

        choices = []
        for alias_name, alias_speed in self.voice_speed_aliases.items():
            choices.append((self._voice_choice_name(voice_name, alias_speed), alias_speed, alias_name))
        return choices

    def resolve_voice_and_speed(
        self,
        voice_name: Optional[str],
        speed: Optional[float],
    ) -> tuple[Optional[str], Optional[float], Optional[str]]:
        if speed is not None or not voice_name:
            return voice_name, speed, None

        for alias_name, alias_speed in self.voice_speed_aliases.items():
            suffix = f"({self.format_voice_speed(alias_speed)})"
            if voice_name.endswith(suffix):
                base_voice_name = voice_name[: -len(suffix)]
                if self._voice_choice_name(base_voice_name, alias_speed) == voice_name:
                    return base_voice_name, alias_speed, alias_name

        return voice_name, speed, None

    def resolve_language(self, language: Optional[str]) -> str:
        if not language:
            return self.default_language if hasattr(self, "default_language") else "en"

        resolved = language.strip().lower()
        if resolved not in SUPPORTED_LANGUAGES:
            LOGGER.warning(
                "Language '%s' not supported by XTTS-v2. Falling back to '%s'.",
                resolved,
                self.default_language if hasattr(self, "default_language") else "en",
            )
            return self.default_language if hasattr(self, "default_language") else "en"

        return resolved

    def is_loaded(self) -> bool:
        return self._tts is not None

    @staticmethod
    def _patch_transformers_for_xtts() -> None:
        """Expose generation helpers expected by XTTS on newer transformers releases."""
        # These imports stay local because they are only needed once the XTTS
        # backend is initialized, and keeping them lazy makes unit tests lighter.
        import transformers

        if not hasattr(transformers, "BeamSearchScorer"):
            from transformers.generation.beam_search import BeamSearchScorer, ConstrainedBeamSearchScorer

            transformers.BeamSearchScorer = BeamSearchScorer
            transformers.ConstrainedBeamSearchScorer = ConstrainedBeamSearchScorer

        if not hasattr(transformers, "DisjunctiveConstraint"):
            from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint

            transformers.DisjunctiveConstraint = DisjunctiveConstraint
            transformers.PhrasalConstraint = PhrasalConstraint

    @staticmethod
    def _patch_torch_load_for_xtts() -> None:
        """Restore torch.load behavior expected by Coqui TTS on PyTorch 2.6+."""
        # Torch is imported lazily for the same reason as the XTTS backend:
        # faster, lighter module import during tests and CLI parsing.
        import torch

        current_load = torch.load
        if getattr(current_load, "_xtts_patched", False):
            return

        def _compat_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return current_load(*args, **kwargs)

        _compat_load._xtts_patched = True  # type: ignore[attr-defined]
        torch.load = _compat_load

    @staticmethod
    def _patch_xtts_attention_mask() -> None:
        """Pass an explicit attention mask to suppress HF generation ambiguity warnings."""
        # XTTS internals are optional heavy dependencies, so keep them lazy.
        from TTS.tts.layers.xtts.gpt import GPT
        import torch

        current_generate = GPT.generate
        if getattr(current_generate, "_xtts_attention_patched", False):
            return

        def _generate_with_attention_mask(self, cond_latents, text_inputs, **hf_generate_kwargs):
            gpt_inputs = self.compute_embeddings(cond_latents, text_inputs)
            hf_generate_kwargs.setdefault(
                "attention_mask",
                torch.ones_like(gpt_inputs, dtype=torch.long, device=gpt_inputs.device),
            )
            gen = self.gpt_inference.generate(
                gpt_inputs,
                bos_token_id=self.start_audio_token,
                pad_token_id=self.stop_audio_token,
                eos_token_id=self.stop_audio_token,
                max_length=self.max_gen_mel_tokens + gpt_inputs.shape[-1],
                **hf_generate_kwargs,
            )
            if "return_dict_in_generate" in hf_generate_kwargs:
                return gen.sequences[:, gpt_inputs.shape[1] :], gen
            return gen[:, gpt_inputs.shape[1] :]

        _generate_with_attention_mask._xtts_attention_patched = True  # type: ignore[attr-defined]
        GPT.generate = _generate_with_attention_mask

    def load(self) -> None:
        if self._tts is not None:
            return

        start_time = time.time()
        os.environ["COQUI_TOS_AGREED"] = "1"
        self._patch_transformers_for_xtts()
        self._patch_torch_load_for_xtts()
        self._patch_xtts_attention_mask()

        # TTS import is intentionally lazy because it pulls in the full XTTS
        # stack and should only happen when the model is actually loaded.
        from TTS.api import TTS

        if self.model_dir:
            os.environ.setdefault("TTS_HOME", self.model_dir)

        use_gpu = (self.device or "").startswith("cuda")
        LOGGER.info("Loading XTTS model '%s' with gpu=%s", self.model_name, use_gpu)
        self._tts = TTS(self.model_name, gpu=use_gpu)
        self._builtin_voice_names = self._discover_builtin_voice_names()
        LOGGER.info("XTTS model loaded in %.1fs", time.time() - start_time)

    def _discover_builtin_voice_names(self) -> list[str]:
        if self._tts is None:
            return []

        speaker_manager = getattr(self._tts.synthesizer.tts_model, "speaker_manager", None)
        speakers = getattr(speaker_manager, "speakers", None)
        if not speakers:
            return []

        return sorted(str(name) for name in speakers.keys())

    def health_payload(self) -> dict:
        return {
            "status": "ok" if self.is_loaded() else "loading",
            "ready": self.is_loaded(),
            "model": self.model_name,
            "device": self.device,
            "default_language": self.default_language,
            "default_voice": self.default_voice,
            "speaker_profiles": self.speaker_store.profile_names(),
            "builtin_voices": self._builtin_voice_names,
            "backend": self.backend,
        }

    def builtin_voice_names(self) -> list[str]:
        self.load()
        return list(self._builtin_voice_names)

    def is_builtin_voice(self, voice_name: Optional[str]) -> bool:
        return bool(voice_name) and (voice_name in self._builtin_voice_names)

    def available_voices(self) -> list[str]:
        local_names = self.speaker_store.profile_names()
        builtin_names = self.builtin_voice_names()
        ordered = [*builtin_names, *local_names]
        if self.default_voice and self.default_voice not in ordered:
            ordered.insert(0, self.default_voice)

        seen: set[str] = set()
        voices: list[str] = []
        for name in ordered:
            for variant_name, _, _ in self.voice_choices_for(name):
                if variant_name in seen:
                    continue
                seen.add(variant_name)
                voices.append(variant_name)
        return voices

    def synthesize(
        self,
        text: str,
        *,
        language: Optional[str] = None,
        voice_name: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> SynthesisResult:
        if not text.strip():
            resolved_voice_name, _, _ = self.resolve_voice_and_speed(voice_name, speed)
            return SynthesisResult(
                audio=[],
                sample_rate=24000,
                language=self.resolve_language(language),
                speaker=resolved_voice_name or self.default_voice,
                processing_time=0.0,
            )

        self.load()
        assert self._tts is not None

        resolved_language = self.resolve_language(language)
        requested_voice, alias_speed, _ = self.resolve_voice_and_speed(voice_name, speed)
        requested_voice = requested_voice or self.default_voice
        builtin_voice = requested_voice if self.is_builtin_voice(requested_voice) else None
        profile = None if builtin_voice else self.speaker_store.get_profile(requested_voice, self.default_voice)
        speaker_wav = None if builtin_voice else SpeakerStore.wav_paths_from_profile(profile)
        start_time = time.time()
        resolved_speed = alias_speed if alias_speed is not None else speed
        if resolved_speed is None:
            resolved_speed = self.speed

        audio = self._tts.synthesizer.tts(
            text=text,
            speaker_name=builtin_voice,
            language_name=resolved_language,
            speaker_wav=speaker_wav,
            split_sentences=self.enable_text_splitting,
            temperature=self.temperature,
            speed=resolved_speed,
            length_penalty=self.length_penalty,
            repetition_penalty=self.repetition_penalty,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        if hasattr(audio, "tolist"):
            audio = audio.tolist()
        else:
            audio = [float(sample) for sample in audio]

        return SynthesisResult(
            audio=audio,
            sample_rate=24000,
            language=resolved_language,
            speaker=builtin_voice or (profile.name if profile else requested_voice),
            processing_time=round(time.time() - start_time, 2),
        )
