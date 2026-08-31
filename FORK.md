# Fork point

This repository is a fork of
[NVIDIA-AI-IOT/reachy-mini-jetson-assistant](https://github.com/NVIDIA-AI-IOT/reachy-mini-jetson-assistant)
(Apache-2.0).

| | |
|---|---|
| Forked from | `NVIDIA-AI-IOT/reachy-mini-jetson-assistant` |
| Fork point commit | `4a029df` — *Merge pull request #5 from NVIDIA-AI-IOT/feat/movements* |
| Fork name | `reachy-mini-gguf-assistant` |
| Licence | Apache-2.0 (unchanged) |

Everything before this commit is upstream's work, unmodified. Everything after
it is this fork.

## Why fork

Upstream runs three models: faster-whisper for speech-to-text, Cosmos-Reason2
for vision-language, and Kokoro ONNX for speech. On an 8 GB Orin Nano they have
to share the GPU with the robot's own workload, and the speech-to-text stage
adds latency and transcription errors before the model has seen anything.

This fork replaces all three with two GGUF models served by llama.cpp:

| Upstream | Here |
|---|---|
| faster-whisper STT | *(gone — the microphone audio goes straight to the model)* |
| Cosmos-Reason2-2B VLM | Gemma 4 E2B (same `mmproj` handles both vision and audio) |
| Kokoro ONNX TTS | EryriLabs Pocket TTS via `llama-tts-server` |

Kept from upstream: Silero VAD, face detection and tracking, the 100 Hz
MovementManager, the official Pollen speaking movements, the camera ring
buffer, the web UI, and the Reachy Mini SDK glue.

## Licence obligations we are honouring (Apache-2.0 §4)

- `LICENSE` is kept as it is, and every NVIDIA `SPDX-FileCopyrightText` header
  stays on the file it was written for.
- Every file this fork changes carries a "Modified by the
  reachy-mini-gguf-assistant contributors, 2026" line under the NVIDIA header.
- `THIRD-PARTY-NOTICES.md` tracks what actually ships: Kokoro, faster-whisper,
  Cosmos and the GPL subprocess section are gone; Kyutai Pocket TTS
  (CC-BY-4.0, attribution required) and llama.cpp (MIT) are added.
- The fork is not called "NVIDIA" anything (Apache-2.0 §6, trademarks).

## Credit

- [NVIDIA-AI-IOT](https://github.com/NVIDIA-AI-IOT/reachy-mini-jetson-assistant)
  for the assistant this is built on.
- [Pollen Robotics](https://www.pollen-robotics.com/reachy-mini/) for Reachy
  Mini, its SDK and the movement library.
- [Kyutai](https://huggingface.co/kyutai) for the Pocket TTS voice work, and
  [EryriLabs](https://huggingface.co/EryriLabs/pocket-tts-GGUF) for the GGUF
  conversion.
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) for both servers.
