# Reachy Mini GGUF Assistant

<p align="center">
  <a href="https://www.pollen-robotics.com/reachy-mini/"><img src="docs/images/reachy-icon.svg" alt="Reachy Mini Lite" height="180"/></a>
  &nbsp;&nbsp;&nbsp;<b>x</b>&nbsp;&nbsp;&nbsp;
  <a href="https://developer.nvidia.com/embedded/jetson-orin-nano"><img src="docs/images/jetson-family.png" alt="NVIDIA Jetson" height="180"/></a>
</p>

A voice and vision assistant for [Reachy Mini
Lite](https://www.pollen-robotics.com/reachy-mini/) on a Jetson Orin Nano,
running **two GGUF models on llama.cpp** and nothing else. No cloud, no API
keys, and no speech-to-text model — it hears you directly.

A fork of
[NVIDIA-AI-IOT/reachy-mini-jetson-assistant](https://github.com/NVIDIA-AI-IOT/reachy-mini-jetson-assistant)
— see [FORK.md](FORK.md) for the fork point and what changed.

## Install

```bash
git clone <this repo> reachy-mini-gguf-assistant
cd reachy-mini-gguf-assistant
./setup.sh
sudo -v && ./start.sh
```

That is the whole thing. `setup.sh` offers prebuilt JetPack 6 binaries with a
pinned sha256, so the usual 30–60 minute CUDA build becomes a minute's
download — pass `--build-from-source` if you would rather compile it yourself,
which is the honest choice if you do not want to run someone else's binaries.
It then pulls the two models, makes the virtualenv, and writes the paths it
found into `config/servers.local.json`.

`start.sh` brings up the language model, waits for it to be genuinely healthy,
then the speech model, warms both, and starts the robot. Open the address it
prints and talk.

### Bring the robot to life

```bash
# 1. plug the Reachy Mini into the Jetson over USB
./setup.sh                # re-run it: the robot step is separate and idempotent
# 2. log out and back in if it just added you to the dialout group
sudo -v && ./start.sh
# 3. open http://<jetson>:8090 and speak
```

Re-running `./setup.sh` with the robot attached is the whole robot install: the
SDK pin and `scipy` into the venv, the apt packages the daemon needs to run
(`libportaudio2` above all — without it the daemon exits at startup), both udev
vendor rules, and `dialout`. `./setup.sh --no-robot` skips it.

The first `start.sh` with the robot connected downloads Pollen's emotions
library (~172 files) before the first movement — once, then never again. If the
robot appears on USB with no audio interface, unplug it and plug it back in;
that revision loses the race on first enumeration.

The `sudo -v` is not for the app. On a Jetson the page cache has to be freed
immediately before each CUDA allocation, or the load dies with
`NvMapMemAllocInternalTagged error 12` while `free -m` still shows gigabytes
"available". `start.sh` does that for you — if sudo does not stop to ask for a
password.

## What it does

Speak. Reachy hears you — the microphone audio goes to the model *as audio*, so
nothing has to be transcribed first — looks at you, and answers out loud while
moving. All of it visible in a browser: live camera, the conversation, system
telemetry.

```
[Mic] → [Silero VAD] ─────────────────┐
[USB Camera] → [Frame ring buffer] ───┼→ [Gemma 4 E2B] → [Pocket TTS] → [Speaker + Robot]
                                      └→ [Web UI over WebSocket]
```

### Two models instead of three

| Upstream ran | This runs |
|---|---|
| faster-whisper (STT) | nothing — Gemma hears the raw audio, and writes it down itself afterwards for the browser |
| Cosmos-Reason2-2B (VLM) | **Gemma 4 E2B** — hears, sees and thinks, one `mmproj` for both image and audio |
| Kokoro ONNX (TTS) | **[EryriLabs Pocket TTS](https://huggingface.co/EryriLabs/pocket-tts-GGUF)** on a warm `llama-tts-server` |

Dropping transcription is not only about the second it costs. Whisper turns a
question into its best guess at words and throws the rest away; the model then
answers the guess. Here it gets the audio.

What that costs is the transcript in the browser, which shows *🎤 spoken* and
how long you spoke for, because nothing wrote the words down. So once the reply
is finished and spoken, Gemma is asked one extra question — what did that clip
say? — and the chip becomes the words. It happens after the answer, never
before it and never alongside it, so nothing is slowed down; and it yields
immediately to the next thing you say, because `llama-server` runs one request
at a time and the conversation matters more than the caption. Talking again
therefore costs you the transcript, not the speed.
`pipeline.transcribe_after_reply: false` turns it off.

What upstream built and this fork keeps: Silero VAD, YuNet face tracking, the
100 Hz MovementManager, the official Pollen speaking movements, the camera ring
buffer, the web UI, the Reachy SDK glue.

## Status — read this before you clone

Honest position, as of this commit:

- **Phase 1, the voice loop: verified on real hardware.** `setup.sh` (prebuilt
  fast path) and `start.sh` were run end to end on a Jetson Orin Nano Super
  8 GB: both models load in ~27 s, a text question through the CLI answers with
  a **794 ms first token at 12.5 tok/s**, a spoken WAV through the fork's own
  `generate_stream(audio_b64=...)` answers correctly with a **1.19 s first
  token**, and the speech client returns real 24 kHz audio. 83 tests pass.
  Only a live microphone and speaker remain untested in this phase - the board
  was driven over SSH.
- **Phase 2, vision: verified on real hardware.** Real USB-camera frames
  through the fork's own module on the Orin: "what do you see" answered
  sensibly from a live frame (2.2 s); a **spoken** question plus a frame in one
  message - the full ears-plus-eyes turn - answered correctly (1.6 s); and the
  distractor test passed: with a frame attached and a maths question asked, the
  model answered "12" and said nothing about the room (1.0 s). The
  ignore-the-image-unless-asked instruction holds under the Q2 quant, so
  unified mode is the shipped path. `pipeline.mode: "split"` remains available
  as an escape hatch, but nothing so far needs it.
- **Phase 4, the robot: verified on real hardware.** A Reachy Mini Lite on the
  Orin Nano Super, driven through `run_web_vision_chat.py`. All 9 motors
  (IDs 10–18) initialise; wake, sleep and gestures were exercised with
  upstream's `scripts/test_reachy_movement.py`. Running: YuNet face detection,
  the 100 Hz head controller, face tracking at 15 Hz, the official Pollen
  speaking movements (the emotions library — 172 files — downloads on the first
  run), WebRTC echo cancellation across robot-mic → Jetson-audio, and the web UI
  on `:8090` reporting "Ready — speak anytime!".
  Two things about this robot revision are worth knowing before you buy into the
  wiring: the microphone array enumerates under the *camera's* USB name, and
  **there is no USB speaker at all** — speech leaves by the Jetson's own audio
  output. Both are already the defaults in `config/settings.yaml`, and
  [SETUP.md](SETUP.md#newer-hardware-revisions) explains the rest.
- **Writing down what you said: code complete, to be measured on the board.**
  After each reply the clip goes back to Gemma to be transcribed, and the
  browser's *🎤 spoken* chip becomes the words. It is written to yield: the
  request is only made once the reply has been given and spoken, a new
  utterance stops it being made at all, and one already in flight has its
  connection closed under it so the single `llama-server` slot goes straight
  back to the conversation. What exists is 20 tests against a stub server
  covering the request, the skip and the hang-up, and the same mechanism
  running against a real llama-server in the sibling
  [jetson-voice-assistant](https://github.com/dwain-barnes/jetson-voice-assistant).
  What does *not* exist is a single measurement on this board: how long the
  second pass takes on an Orin, and whether a turn is ever slowed down by the
  previous turn's transcript being hung up on. Until those numbers exist, treat
  it as unproven on hardware.
- **Barge-in is still future work (Phase 5).** You cannot yet interrupt the
  robot mid-sentence; it finishes speaking, then listens. The echo cancellation
  needed for that is running, but the pipeline does not act on speech detected
  during playback.
- **No latency table yet.** The numbers that belong here have to be measured on
  an Orin Nano, not inferred from a desktop.

## Modes

| Mode | Entry point | What it is |
|---|---|---|
| **Web Vision Chat** | `python3 run_web_vision_chat.py` | camera + voice + browser UI on `:8090` — what `start.sh` runs |
| **Vision Chat** | `python3 run_vision_chat.py` | the same, terminal only |
| **Voice Chat** | `python3 run_voice_chat.py` | no camera |
| **Text Chat** | `python3 main.py chat -t` | typing, no mic or speaker |
| **CLI** | `python3 main.py ask "..."` | one question, one answer |

## Stack

| Piece | What | Where it runs |
|---|---|---|
| Thinking, hearing, seeing | Gemma 4 E2B GGUF (`UD-Q2_K_XL`) + `mmproj-F16` | `llama-server`, GPU, port 8080 |
| Speech | Pocket TTS GGUF + a reference voice | `llama-tts-server`, GPU, port 8100 |
| VAD | Silero (ONNX) | CPU |
| Face detection | YuNet via OpenCV | CPU |
| Robot | Reachy Mini SDK + Pollen recorded moves | USB |
| Web UI | FastAPI + WebSocket | port 8090 |

Roughly 4.0 GiB of GPU and 6.8 GiB in total on an 8 GB Orin Nano, leaving about
0.6 GiB of headroom. That is why `contextSize` is 2048, and why raising it is
the first thing that will break this: the KV cache comes out of the same pool
as the weights.

## Configuration

Two files, doing different jobs:

- **`config/settings.yaml`** — how the assistant behaves. Edit freely.
- **`config/servers.local.json`** — where the binaries and models are, and
  which ports. Written by `setup.sh`; edit it if you move things.

| Section | Controls |
|---|---|
| `pipeline` | `unified` (audio straight to the model) or `split` (STT + a separate VLM); whether to run Whisper purely to show the words in the UI; whether to ask the model itself for those words after the reply |
| `llm` | server URL, token budget, temperature, system prompts |
| `tts` | speech server URL, voice name, runaway cap |
| `stt` | Whisper settings — split mode only |
| `audio` | devices, echo cancellation |
| `vad` | Silero threshold, silence duration, utterance filters |
| `vision` | camera resolution, capture FPS, frames per query, prompt, few-shot |
| `reachy` | connection, face tracking, speaking movements |
| `web` | UI FPS, host, port |
| `rag` | off — a third model does not fit next to the other two |

The ports have to agree across the two files: `llm.base_url` ↔ `llmPort`,
`tts.base_url` ↔ `ttsPort`, `web.port` ↔ `webUiPort`.

### Speech on the CPU

`"ttsOnGpu": false` in `config/servers.local.json` moves Pocket TTS to the CPU
and hands the whole GPU to Gemma. It costs about 19 seconds a sentence on an
Orin Nano — fine for testing, not for conversation. You want it if you move to
a bigger quant, which is the other lever: `./setup.sh --quant UD-Q4_K_XL
--force` answers better and stops leaving room for speech on the GPU.

## When something else owns the GPU

Many Jetsons boot with a GPU service already running (NanoOWL,
`jetson-inference`, a stray container). Stop it for the session:

```bash
systemctl list-units --type=service --state=running | grep -i 'nano\|owl\|jetson'
sudo systemctl stop <service>
```

Do not `disable` it — that is your own setup, and it should come back on the
next boot. Do not `pkill` it either: services restart on failure, and
containers keep holding GPU memory after the visible process dies.

## Project structure

```
reachy-mini-gguf-assistant/
├── app/
│   ├── pipeline.py           # audio I/O, VAD, utterance → WAV, sentence-paced playback
│   ├── tts_client.py         # llama-tts-server client (new in this fork)
│   ├── sentence_split.py     # sentence boundaries for a token stream (new in this fork)
│   ├── llm.py                # chat client: text + audio + images
│   ├── stt.py                # faster-whisper — split mode only
│   ├── after_transcript.py   # the words, asked for after the reply is out
│   ├── config.py             # typed config + YAML loader
│   ├── camera.py             # USB webcam ring buffer
│   ├── face_detector.py      # YuNet
│   ├── face_tracker.py       # 15 Hz visual tracking
│   ├── movement_manager.py   # single-writer 100 Hz motion controller
│   ├── speaking_movements.py # official Pollen gestures, synced to speech
│   ├── vision_capture.py     # stable-frame capture
│   ├── reachy.py             # robot connection and daemon
│   ├── web.py                # FastAPI + WebSocket
│   ├── monitor.py            # CPU/GPU/RAM
│   ├── rag.py                # optional retrieval
│   ├── audio.py              # PulseAudio / ALSA helpers
│   └── cli.py                # Typer CLI
├── config/
│   ├── settings.yaml         # behaviour
│   ├── servers.json          # template for the launcher
│   └── servers.local.json    # written by setup.sh (gitignored)
├── legacy/                   # upstream's Docker launchers, retired
├── setup.sh                  # one-time install
├── start.sh                  # start everything, in the right order
└── tests/                    # 83 tests, no hardware needed
```

## Tests

```bash
python3 -m pytest tests/
```

No robot, GPU or microphone required: both servers are stubbed with a real HTTP
server on localhost.

## Credit

- **[NVIDIA-AI-IOT](https://github.com/NVIDIA-AI-IOT/reachy-mini-jetson-assistant)**
  wrote the assistant this forks: the pipeline, the face tracking, the motion
  controller, the web UI. Apache-2.0.
- **[Pollen Robotics](https://www.pollen-robotics.com/reachy-mini/)** make
  Reachy Mini, and the movement library the gestures come from.
- **[Kyutai](https://huggingface.co/kyutai/tts-voices)** for the TTS voice work,
  and **[EryriLabs](https://huggingface.co/EryriLabs/pocket-tts-GGUF)** for the
  Pocket TTS GGUF conversion. CC-BY-4.0 — if you publish audio from this,
  credit them.
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** for both
  servers, and
  **[llama-tts-server](https://github.com/dwain-barnes/llama-tts-server)** for
  the patch that keeps the speech model warm between sentences.

Full list in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). Gemma is under
Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms), not an OSI
licence.

## Reachy Mini resources

| Resource | Link |
|----------|------|
| Getting started | [huggingface.co/docs/reachy_mini](https://huggingface.co/docs/reachy_mini/index) |
| Reachy Mini Lite setup | [Lite guide](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini_lite/get_started) |
| Python SDK | [SDK reference](https://huggingface.co/docs/reachy_mini/SDK/readme) |
| Examples | [github.com/pollen-robotics/reachy_mini/examples](https://github.com/pollen-robotics/reachy_mini/tree/main/examples) |
| Discord | [Community](https://discord.gg/Y7FgMqHsub) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), inherited from upstream, including the
Developer Certificate of Origin sign-off.

## Licence

Apache-2.0, same as upstream — see [LICENSE](LICENSE). Every file NVIDIA wrote
keeps its copyright header; every file this fork changed says so underneath.
