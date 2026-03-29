"""Wyoming event handler for XTTS-v2."""

from __future__ import annotations

import argparse
import asyncio
import math
from typing import Any, Optional

from .audio import float32_to_pcm16
from .synthesizer import XttsSynthesizer
from .text import SentenceChunker
from .wyoming_protocol import (
    AsyncEventHandler,
    AudioChunk,
    AudioStart,
    AudioStop,
    Describe,
    Error,
    Event,
    Info,
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

class XttsEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        synthesizer: XttsSynthesizer,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.synthesizer = synthesizer
        self.wyoming_info_event = wyoming_info.event()
        self._streaming = False
        self._voice_name: Optional[str] = None
        self._language: Optional[str] = None
        self._speed: Optional[float] = None
        self._chunker = SentenceChunker()
        self._lock = asyncio.Lock()

    async def handle_event(self, event: Event) -> bool:
        try:
            if Describe.is_type(event.type):
                await self.write_event(self.wyoming_info_event)
                return True

            if Synthesize.is_type(event.type):
                synthesize = Synthesize.from_event(event)
                if self._streaming:
                    return True
                return await self._handle_full_synthesize(synthesize)

            if self.cli_args.no_streaming:
                return True

            if SynthesizeStart.is_type(event.type):
                stream_start = SynthesizeStart.from_event(event)
                self._streaming = True
                self._chunker = SentenceChunker()
                self._voice_name = getattr(stream_start.voice, "name", None)
                self._language = getattr(stream_start.voice, "speaker", None) or self.cli_args.language
                self._speed = self._get_speed_override(getattr(stream_start, "context", None))
                return True

            if SynthesizeChunk.is_type(event.type):
                stream_chunk = SynthesizeChunk.from_event(event)
                for sentence in self._chunker.add_chunk(stream_chunk.text):
                    await self._emit_sentence(
                        sentence,
                        voice_name=self._voice_name,
                        language=self._language or self.cli_args.language,
                        speed=self._speed,
                    )
                return True

            if SynthesizeStop.is_type(event.type):
                remainder = self._chunker.finish()
                if remainder:
                    await self._emit_sentence(
                        remainder,
                        voice_name=self._voice_name,
                        language=self._language or self.cli_args.language,
                        speed=self._speed,
                    )
                await self.write_event(SynthesizeStopped().event())
                self._streaming = False
                self._voice_name = None
                self._language = None
                self._speed = None
                return True

            return True
        except Exception as err:  # pragma: no cover
            await self.write_event(Error(text=str(err), code=err.__class__.__name__).event())
            raise

    async def _handle_full_synthesize(self, synthesize: Synthesize) -> bool:
        voice_name = getattr(synthesize.voice, "name", None)
        speed = self._get_speed_override(getattr(synthesize, "context", None))
        sentences = self._chunker.add_chunk(synthesize.text)
        if not sentences:
            remainder = synthesize.text.strip()
            if remainder:
                sentences = [remainder]

        sent_any = False
        for index, sentence in enumerate(sentences):
            await self._emit_sentence(
                sentence,
                voice_name=voice_name,
                language=self.cli_args.language,
                speed=speed,
                send_start=(index == 0),
                send_stop=False,
            )
            sent_any = True

        remainder = self._chunker.finish()
        if remainder:
            await self._emit_sentence(
                remainder,
                voice_name=voice_name,
                language=self.cli_args.language,
                speed=speed,
                send_start=not sent_any,
                send_stop=True,
            )
        else:
            await self.write_event(AudioStop().event())

        return True

    async def _emit_sentence(
        self,
        sentence: str,
        *,
        voice_name: Optional[str],
        language: Optional[str],
        speed: Optional[float] = None,
        send_start: bool = True,
        send_stop: bool = True,
    ) -> None:
        async with self._lock:
            result = self.synthesizer.synthesize(
                sentence,
                language=language,
                voice_name=voice_name,
                speed=speed,
            )

        audio_bytes = float32_to_pcm16(result.audio)
        width = 2
        channels = 1
        bytes_per_sample = width * channels
        bytes_per_chunk = bytes_per_sample * self.cli_args.samples_per_chunk
        num_chunks = max(1, int(math.ceil(len(audio_bytes) / max(1, bytes_per_chunk))))

        if send_start:
            await self.write_event(
                AudioStart(rate=result.sample_rate, width=width, channels=channels).event()
            )

        for index in range(num_chunks):
            offset = index * bytes_per_chunk
            chunk = audio_bytes[offset : offset + bytes_per_chunk]
            await self.write_event(
                AudioChunk(
                    audio=chunk,
                    rate=result.sample_rate,
                    width=width,
                    channels=channels,
                ).event()
            )

        if send_stop:
            await self.write_event(AudioStop().event())

    @staticmethod
    def _get_speed_override(context: Optional[dict[str, Any]]) -> Optional[float]:
        if not isinstance(context, dict):
            return None

        candidates = (
            context.get("speed"),
            context.get("tts_speed"),
            (context.get("xtts") or {}).get("speed") if isinstance(context.get("xtts"), dict) else None,
        )

        for candidate in candidates:
            if candidate is None:
                continue
            try:
                value = float(candidate)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value

        return None
