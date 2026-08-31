# Third-Party Software Notices

This project uses the following third-party open source software.

Modified by the reachy-mini-gguf-assistant contributors, 2026: Kokoro,
faster-whisper, Cosmos-Reason2 and ChromaDB are no longer part of the default
install, and with `app/tts_worker.py` deleted there is no GPL-licensed code
left in the tree to isolate. Kyutai Pocket TTS and llama.cpp are added.

## Direct Dependencies

| Package | License | URL |
|---------|---------|-----|
| PyYAML | MIT | https://github.com/yaml/pyyaml |
| Rich | MIT | https://github.com/Textualize/rich |
| Typer | MIT | https://github.com/fastapi/typer |
| psutil | BSD-3-Clause | https://github.com/giampaolo/psutil |
| sounddevice | MIT | https://github.com/spatialaudio/python-sounddevice |
| Silero VAD | MIT | https://github.com/snakers4/silero-vad |
| httpx | BSD-3-Clause | https://github.com/encode/httpx |
| opencv-python-headless | Apache-2.0 | https://github.com/opencv/opencv-python |
| FastAPI | MIT | https://github.com/fastapi/fastapi |
| Uvicorn | BSD-3-Clause | https://github.com/encode/uvicorn |

## Separately Installed Dependencies

| Package | License | URL |
|---------|---------|-----|
| onnxruntime-gpu | MIT | https://github.com/microsoft/onnxruntime |
| NumPy | BSD-3-Clause | https://github.com/numpy/numpy |
| reachy-mini | Apache-2.0 | https://github.com/pollen-robotics/reachy_mini |

## Optional Dependencies

Not installed by default. `faster-whisper` is only needed for
`pipeline.mode: "split"` or `transcribe_for_display`; ChromaDB only if you turn
RAG back on with an embedding server of your own.

| Package | License | URL |
|---------|---------|-----|
| faster-whisper | MIT | https://github.com/SYSTRAN/faster-whisper |
| CTranslate2 | MIT | https://github.com/OpenNMT/CTranslate2 |
| ChromaDB | Apache-2.0 | https://github.com/chroma-core/chroma |
| sentence-transformers | Apache-2.0 | https://github.com/UKPLab/sentence-transformers |

## Key Transitive Dependencies

| Package | License | URL |
|---------|---------|-----|
| PyTorch | BSD-3-Clause | https://github.com/pytorch/pytorch |
| torchaudio | BSD-3-Clause | https://github.com/pytorch/audio |
| Starlette | BSD-3-Clause | https://github.com/encode/starlette |
| Pydantic | MIT | https://github.com/pydantic/pydantic |

## External Services (Process-Isolated)

Both run as separate processes started by `start.sh` and are spoken to over
HTTP on localhost. `llama-tts-server` is a patch on top of llama.cpp that keeps
the speech model warm between requests instead of reloading it per sentence.

| Software | License | URL |
|----------|---------|-----|
| llama.cpp | MIT | https://github.com/ggml-org/llama.cpp |
| llama-tts-server (patch) | MIT | https://github.com/dwain-barnes/llama-tts-server |

`app/sentence_split.py` is ported from `scripts/voice-chat.py` in
llama-tts-server (MIT).

## Model Licenses

| Model | License | URL |
|-------|---------|-----|
| Gemma 4 E2B (GGUF) | Gemma Terms of Use | https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF |
| Pocket TTS (GGUF) | CC-BY-4.0 | https://huggingface.co/EryriLabs/pocket-tts-GGUF |
| Kyutai TTS voices | CC-BY-4.0 | https://huggingface.co/kyutai/tts-voices |
| YuNet face detection | MIT | https://huggingface.co/opencv/face_detection_yunet |
| FER+ int8 emotion | MIT | https://huggingface.co/onnxmodelzoo/emotion-ferplus-12-int8 |
| reachy-mini-emotions-library | Apache-2.0 | https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library |
| faster-whisper (small.en) | MIT | https://huggingface.co/Systran/faster-whisper-small.en |

### Attribution required

Pocket TTS and the reference voice are CC-BY-4.0, which means the credit is not
optional. The voice comes from Kyutai's TTS voice work, converted to GGUF by
EryriLabs. If you publish audio, a demo or a fork built on this, say so:

> Speech by [Pocket TTS](https://huggingface.co/EryriLabs/pocket-tts-GGUF)
> (Kyutai voices, CC-BY-4.0).

Gemma is under Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms),
not an OSI licence. Read them before using this commercially: they carry a use
policy that travels with the weights.
