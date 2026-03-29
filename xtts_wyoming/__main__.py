"""CLI entrypoint for Wyoming XTTS."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from functools import partial
from typing import Optional

from . import __version__
from .handler import XttsEventHandler
from .speaker_store import SpeakerStore
from .synthesizer import SUPPORTED_LANGUAGES, XttsSynthesizer
from .wyoming_protocol import (
    AsyncServer,
    Attribution,
    Info,
    TtsProgram,
    TtsVoice,
    WYOMING_AVAILABLE,
)


LOGGER = logging.getLogger("xtts-wyoming")


def _voice_entry(name: str, *, attribution_name: str, attribution_url: str) -> TtsVoice:
    return TtsVoice(
        name=name,
        description=name,
        attribution=Attribution(
            name=attribution_name,
            url=attribution_url,
        ),
        installed=True,
        version=None,
        languages=sorted(SUPPORTED_LANGUAGES),
    )


def build_info(args: argparse.Namespace, synthesizer: XttsSynthesizer) -> Info:
    store = SpeakerStore(args.speaker_dir)
    builtin_voice_names = synthesizer.builtin_voice_names()
    builtin_voices = []
    for voice_name in builtin_voice_names:
        for variant_name, _, _ in synthesizer.voice_choices_for(voice_name):
            builtin_voices.append(
                _voice_entry(
                    variant_name,
                    attribution_name="Coqui XTTS built-in",
                    attribution_url="https://huggingface.co/coqui/XTTS-v2",
                )
            )

    local_voices = []
    for profile in store.list_profiles():
        for variant_name, _, _ in synthesizer.voice_choices_for(profile.name):
            local_voices.append(
                _voice_entry(
                    variant_name,
                    attribution_name="Local speaker references",
                    attribution_url="https://github.com/coqui-ai/TTS",
                )
            )

    voices = [*builtin_voices]
    seen_names = {voice.name for voice in voices}
    for voice in local_voices:
        if voice.name in seen_names:
            continue
        voices.append(voice)
        seen_names.add(voice.name)

    return Info(
        tts=[
            TtsProgram(
                name="xtts",
                description="Wyoming protocol server backed by Coqui XTTS-v2",
                attribution=Attribution(
                    name="Coqui",
                    url="https://huggingface.co/coqui/XTTS-v2",
                ),
                installed=True,
                version=__version__,
                voices=voices,
                supports_synthesize_streaming=(not args.no_streaming),
            )
        ]
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wyoming XTTS-v2 server for Home Assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uri", default=os.getenv("WYOMING_URI", "tcp://0.0.0.0:10201"))
    parser.add_argument(
        "--model",
        default=os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"),
    )
    parser.add_argument("--voice", default=os.getenv("XTTS_DEFAULT_VOICE", "default"))
    parser.add_argument("--language", default=os.getenv("XTTS_DEFAULT_LANGUAGE", "pl"))
    parser.add_argument("--speaker-dir", default=os.getenv("XTTS_SPEAKER_DIR", "/data/speakers"))
    parser.add_argument("--model-dir", default=os.getenv("XTTS_MODEL_DIR", "/data/models"))
    parser.add_argument("--device", default=os.getenv("XTTS_DEVICE", "cuda"))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--samples-per-chunk", type=int, default=1024)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--http-host", default=os.getenv("HTTP_HOST"))
    parser.add_argument("--http-port", type=int, default=int(os.getenv("HTTP_PORT", "8180")))
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--voice-speed-presets",
        default=os.getenv("XTTS_VOICE_SPEED_PRESETS", "normal=1.0,fast=1.15"),
        help="Comma-separated voice aliases exposed in Home Assistant, e.g. normal=1.0,fast=1.15",
    )
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=2.0)
    parser.add_argument("--disable-text-splitting", action="store_true")
    return parser.parse_args(argv)


async def _serve_http_debug(
    synthesizer: XttsSynthesizer,
    *,
    host: str,
    port: int,
) -> None:
    # Keep HTTP debug dependencies lazy so the Wyoming server can start
    # without importing the full FastAPI/uvicorn stack unless requested.
    import uvicorn

    from . import server as http_server

    http_server.service = synthesizer
    config = uvicorn.Config(
        http_server.app,
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


async def serve(args: argparse.Namespace) -> None:
    if not WYOMING_AVAILABLE:
        raise RuntimeError("The 'wyoming' package is not installed. Install project dependencies first.")

    synthesizer = XttsSynthesizer(
        model_name=args.model,
        default_language=args.language,
        default_voice=args.voice,
        speaker_dir=args.speaker_dir,
        model_dir=args.model_dir,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        speed=args.speed,
        voice_speed_aliases=XttsSynthesizer.parse_voice_speed_aliases(args.voice_speed_presets),
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
        enable_text_splitting=(not args.disable_text_splitting),
    )
    SpeakerStore(args.speaker_dir).ensure_default_profile(args.voice)
    synthesizer.load()
    info = build_info(args, synthesizer)
    server = AsyncServer.from_uri(args.uri)

    LOGGER.info("Model: %s", args.model)
    LOGGER.info("Voice: %s", args.voice)
    LOGGER.info("Language: %s", args.language)
    LOGGER.info("URI: %s", args.uri)
    LOGGER.info("Speaker dir: %s", args.speaker_dir)
    tasks = [
        asyncio.create_task(server.run(partial(XttsEventHandler, info, args, synthesizer)))
    ]

    if args.http_host:
        LOGGER.info("HTTP debug: http://%s:%s", args.http_host, args.http_port)
        tasks.append(
            asyncio.create_task(
                _serve_http_debug(
                    synthesizer,
                    host=args.http_host,
                    port=args.http_port,
                )
            )
        )

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    for task in pending:
        task.cancel()

    for task in done:
        task.result()


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
