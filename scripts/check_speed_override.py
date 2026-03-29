"""Small Wyoming client to verify per-request XTTS speed override."""

from __future__ import annotations

import argparse
import asyncio
import json

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize, SynthesizeVoice


async def _run(host: str, port: int, voice: str, text: str, speed: float) -> dict:
    client = AsyncTcpClient(host, port)
    await client.connect()
    await client.write_event(
        Synthesize(
            text=text,
            voice=SynthesizeVoice(name=voice),
            context={"speed": speed},
        ).event()
    )

    event_types: list[str] = []
    audio_bytes = 0
    rate = None

    while True:
        event = await client.read_event()
        if event is None:
            break

        event_types.append(event.type)
        if AudioStart.is_type(event.type):
            rate = AudioStart.from_event(event).rate
        elif AudioChunk.is_type(event.type):
            audio_bytes += len(AudioChunk.from_event(event).audio)
        elif AudioStop.is_type(event.type):
            break

    await client.disconnect()
    return {
        "voice": voice,
        "speed": speed,
        "rate": rate,
        "audio_bytes": audio_bytes,
        "event_types": event_types,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10201)
    parser.add_argument("--voice", default="Ana Florence")
    parser.add_argument("--text", default="To jest test predkosci.")
    parser.add_argument("--speed", type=float, default=1.35)
    args = parser.parse_args()

    result = asyncio.run(_run(args.host, args.port, args.voice, args.text, args.speed))
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
