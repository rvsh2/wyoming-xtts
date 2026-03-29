import asyncio
import unittest
from types import SimpleNamespace

from xtts_wyoming.handler import XttsEventHandler
from xtts_wyoming.wyoming_protocol import Event, Info, TtsProgram


class CollectingHandler(XttsEventHandler):
    def __init__(self, cli_args, synthesizer):
        super().__init__(Info(tts=[TtsProgram(name="xtts")]), cli_args, synthesizer)
        self.events = []

    async def write_event(self, event):
        self.events.append(event)


class HandlerTests(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.run(coro)

    def test_describe_returns_info(self):
        handler = CollectingHandler(
            cli_args=SimpleNamespace(no_streaming=False, samples_per_chuck=1024, samples_per_chunk=1024, language="pl"),
            synthesizer=SimpleNamespace(),
        )
        self.run_async(handler.handle_event(Event("describe", {})))
        self.assertEqual(handler.events[0].type, "describe")

    def test_synthesize_returns_audio_events(self):
        class FakeSynthesizer:
            def synthesize(self, text, *, language, voice_name, speed=None):
                return SimpleNamespace(
                    audio=[0.0, 0.25, -0.25, 0.1],
                    sample_rate=24000,
                    language=language,
                    speaker=voice_name,
                )

        handler = CollectingHandler(
            cli_args=SimpleNamespace(no_streaming=False, samples_per_chunk=2, language="pl"),
            synthesizer=FakeSynthesizer(),
        )

        self.run_async(handler.handle_event(Event("synthesize", {"text": "test one. test two."})))
        event_types = [event.type for event in handler.events]
        self.assertEqual(event_types[0], "audio-start")
        self.assertIn("audio-chunk", event_types)
        self.assertEqual(event_types[-1], "audio-stop")

    def test_streaming_stop_emits_synthesize_stopped(self):
        class FakeSynthesizer:
            def synthesize(self, text, *, language, voice_name, speed=None):
                return SimpleNamespace(
                    audio=[0.1, -0.1],
                    sample_rate=24000,
                    language=language,
                    speaker=voice_name,
                )

        handler = CollectingHandler(
            cli_args=SimpleNamespace(no_streaming=False, samples_per_chunk=8, language="pl"),
            synthesizer=FakeSynthesizer(),
        )

        self.run_async(handler.handle_event(Event("synthesize-start", {})))
        self.run_async(handler.handle_event(Event("synthesize-chunk", {"text": "Ala ma kota."})))
        self.run_async(handler.handle_event(Event("synthesize-stop", {})))

        self.assertEqual(handler.events[-1].type, "synthesize-stopped")

    def test_synthesize_reads_speed_from_context(self):
        calls = []

        class FakeSynthesizer:
            def synthesize(self, text, *, language, voice_name, speed=None):
                calls.append(
                    {
                        "text": text,
                        "language": language,
                        "voice_name": voice_name,
                        "speed": speed,
                    }
                )
                return SimpleNamespace(
                    audio=[0.1, -0.1],
                    sample_rate=24000,
                    language=language,
                    speaker=voice_name,
                )

        handler = CollectingHandler(
            cli_args=SimpleNamespace(no_streaming=False, samples_per_chunk=8, language="pl"),
            synthesizer=FakeSynthesizer(),
        )

        self.run_async(
            handler.handle_event(
                Event(
                    "synthesize",
                    {"text": "Ala ma kota.", "context": {"speed": 1.2}},
                )
            )
        )

        self.assertEqual(calls[0]["speed"], 1.2)


if __name__ == "__main__":
    unittest.main()
