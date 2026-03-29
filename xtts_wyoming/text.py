"""Text chunking helpers for streaming-style Wyoming TTS input."""

from __future__ import annotations

import re


SENTENCE_END_RE = re.compile(r"(.+?[.!?。！？])(?:\s+|$)", re.DOTALL)


class SentenceChunker:
    def __init__(self) -> None:
        self._buffer = ""

    def add_chunk(self, text: str) -> list[str]:
        self._buffer += text
        sentences: list[str] = []

        while True:
            match = SENTENCE_END_RE.match(self._buffer)
            if match is None:
                break
            sentence = match.group(1).strip()
            if sentence:
                sentences.append(sentence)
            self._buffer = self._buffer[match.end():]

        return sentences

    def finish(self) -> str:
        remainder = self._buffer.strip()
        self._buffer = ""
        return remainder
