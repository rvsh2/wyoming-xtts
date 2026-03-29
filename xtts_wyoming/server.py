"""HTTP debug server for XTTS Wyoming."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .audio import pcm16_wav_bytes
from .synthesizer import XttsSynthesizer

INDEX_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "index.html"
service = XttsSynthesizer(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    default_language="pl",
    default_voice="default",
    speaker_dir="/data/speakers",
    model_dir="/data/models",
)


class SynthesisRequest(BaseModel):
    text: str
    language: str | None = None
    voice: str | None = None


def render_index() -> str:
    template = INDEX_TEMPLATE_PATH.read_text(encoding="utf-8")
    template = template.replace("__MODEL__", service.model_name)
    template = template.replace("__LANG__", service.default_language)
    template = template.replace("__VOICE__", service.default_voice or "none")
    return template


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="wyoming-xtts debug server", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return render_index()


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(service.health_payload())


@app.get("/voices")
async def voices() -> JSONResponse:
    return JSONResponse({"voices": service.available_voices()})


@app.post("/synthesize")
async def synthesize(request: SynthesisRequest) -> JSONResponse:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    result = service.synthesize(
        request.text,
        language=request.language,
        voice_name=request.voice,
    )
    wav_bytes = pcm16_wav_bytes(result.audio, sample_rate=result.sample_rate)
    return JSONResponse(
        {
            **result.asdict(),
            "wav_bytes": len(wav_bytes),
        }
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XTTS HTTP debug server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8180)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    uvicorn.run(app, host=args.host, port=args.port)
