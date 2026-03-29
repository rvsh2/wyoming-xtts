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

Configuration is loaded from `.env` because `compose.yml` uses:

```yaml
env_file:
  - .env
```

If you want to change runtime options such as `XTTS_VOICE_SPEED_PRESETS`, edit `.env` and restart the container.

## Voices

There are two ways to get voices in Home Assistant.

Built-in XTTS voices:

- work out of the box
- do not require any files in `data/speakers`
- are the easiest option if you just want to start using TTS

Local cloned voices:

- are optional
- use your own reference WAV file or files
- are only needed if you want voice cloning

Local voices are discovered from `data/speakers`:

- `data/speakers/mira.wav` -> publishes voice `mira`
- `data/speakers/mira/*.wav` -> also publishes voice `mira`

What this means in practice:

- if you put one file at `data/speakers/mira.wav`, XTTS will use that file as the speaker reference
- if you create a folder `data/speakers/mira/` and put multiple `.wav` files inside it, XTTS will use all of them as references for the same cloned voice
- after adding or changing local voices, restart the container and reload the Wyoming integration in Home Assistant

If you do not want voice cloning, you can leave `data/speakers` empty and use only built-in XTTS voices.

Voice speed variants are exposed as separate Wyoming voices so Home Assistant can use them without a custom integration.

Default variants:

- `VoiceName(1.00x)`
- `VoiceName(1.15x)`

The value comes from `.env`:

```env
XTTS_VOICE_SPEED_PRESETS=normal=1.0,fast=1.15
```

To change it:

1. Edit `.env`
2. Change `XTTS_VOICE_SPEED_PRESETS`
3. Restart the container
4. Reload the Wyoming integration in Home Assistant so it fetches the new voice list

Example:

```env
XTTS_VOICE_SPEED_PRESETS=normal=1.0,fast=1.20
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
