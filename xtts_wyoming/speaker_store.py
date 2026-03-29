"""Speaker profile discovery for XTTS reference WAV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpeakerProfile:
    name: str
    wav_paths: list[Path]


class SpeakerStore:
    def __init__(self, speaker_dir: str | Path) -> None:
        self.speaker_dir = Path(speaker_dir)

    def list_profiles(self) -> list[SpeakerProfile]:
        if not self.speaker_dir.exists():
            return []

        profiles: list[SpeakerProfile] = []

        for wav_path in sorted(self.speaker_dir.glob("*.wav")):
            profiles.append(SpeakerProfile(name=wav_path.stem, wav_paths=[wav_path]))

        for child in sorted(self.speaker_dir.iterdir()):
            if not child.is_dir():
                continue
            wav_paths = sorted(child.glob("*.wav"))
            if wav_paths:
                profiles.append(SpeakerProfile(name=child.name, wav_paths=wav_paths))

        return profiles

    def get_profile(self, name: str | None, default_name: str | None = None) -> SpeakerProfile | None:
        requested = name or default_name
        if not requested:
            return None

        for profile in self.list_profiles():
            if profile.name == requested:
                return profile

        requested_path = Path(requested)
        if requested_path.exists() and requested_path.is_file() and requested_path.suffix.lower() == ".wav":
            return SpeakerProfile(name=requested_path.stem, wav_paths=[requested_path])

        if requested_path.exists() and requested_path.is_dir():
            wav_paths = sorted(requested_path.glob("*.wav"))
            if wav_paths:
                return SpeakerProfile(name=requested_path.name, wav_paths=wav_paths)

        return None

    def ensure_default_profile(self, default_name: str) -> None:
        if self.get_profile(default_name) is not None:
            return

        self.speaker_dir.mkdir(parents=True, exist_ok=True)
        placeholder = self.speaker_dir / f"{default_name}.txt"
        if not placeholder.exists():
            placeholder.write_text(
                "Add one or more .wav reference files to create a speaker profile.\n",
                encoding="utf-8",
            )

    def profile_names(self) -> list[str]:
        return [profile.name for profile in self.list_profiles()]

    @staticmethod
    def wav_paths_from_profile(profile: SpeakerProfile | None) -> list[str] | None:
        if profile is None:
            return None
        return [str(path) for path in profile.wav_paths]
