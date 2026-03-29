# wyoming-xtts

Wyoming TTS server for Home Assistant built on `coqui/XTTS-v2`.

## What It Does

- exposes Wyoming TTS on port `10201`
- exposes optional HTTP debug on port `8180`
- supports XTTS built-in voices
- supports local cloned voices from `data/speakers`
- publishes voice variants with speed in the name, for example `Zofia Kendirk(1.15x)`

## Project Layout

- `xtts_wyoming/` - Wyoming handler, XTTS runtime, audio helpers, HTTP debug
- `server.py` - HTTP debug entrypoint
- `compose.yml` - Docker runtime
- `tests/` - unit tests

## Docker

```bash
cp .env.example .env
docker compose up --build -d
```

Exposed ports:

- `10201` - Wyoming
- `8180` - HTTP debug

Persisted data:

- `./data/models` - XTTS model cache
- `./data/speakers` - local speaker WAV files

The container sets `COQUI_TOS_AGREED=1`. Use it only if you have reviewed and accepted the Coqui license terms.

## Voices

Built-in voices are published automatically.

Local voices are discovered from `data/speakers`:

- `data/speakers/mira.wav` -> `mira`
- `data/speakers/mira/*.wav` -> `mira`

Voice speed variants are exposed as separate Wyoming voices so Home Assistant can use them without a custom integration.

Default variants:

- `VoiceName(1.00x)`
- `VoiceName(1.15x)`

You can change them with `XTTS_VOICE_SPEED_PRESETS`, for example:

```env
XTTS_VOICE_SPEED_PRESETS=normal=1.0,fast=1.2
```

## Home Assistant

1. Start the container.
2. Add the `Wyoming Protocol` integration.
3. Point it to port `10201`.
4. Reload the integration if you change the published voice list.

## Local Run

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m xtts_wyoming \
  --uri tcp://0.0.0.0:10201 \
  --speaker-dir ./data/speakers \
  --model-dir ./data/models \
  --voice default \
  --language pl
```

Optional HTTP debug:

```bash
python server.py --host 0.0.0.0 --port 8180
```

## Tests

```bash
python -m unittest discover -s tests
```

## License

Unless stated otherwise, the code in this repository is intended to be distributed under the MIT License.

This repository also integrates the `coqui/XTTS-v2` model, which is licensed separately by Coqui.

The XTTS-v2 model itself is published by Coqui under the `Coqui Public Model License (CPML)`.
In practice, that means:

- non-commercial use is covered by the default model license
- commercial use of XTTS requires separate licensing from Coqui
- you should review the current XTTS model card and CPML terms before deployment

References:

- XTTS-v2 model card: `https://huggingface.co/coqui/XTTS-v2`
- CPML text: `https://coqui.ai/cpml.txt`
- Coqui XTTS FAQ: `https://coqui.ai/faq/`
