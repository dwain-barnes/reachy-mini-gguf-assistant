# legacy/

The upstream launchers, kept for reference and for anyone running
`pipeline.mode: "split"` against their own servers. Nothing in this fork calls
them, and `./start.sh` replaces both.

| File | What it was for | Why it is here |
|---|---|---|
| `run_llama_cpp.sh` | Docker `llama.cpp` server for the Cosmos-Reason2 VLM | The fork runs the native binaries instead. The pinned upstream image predates server-side `input_audio` routing, so the microphone audio would never reach the model, and it has no `llama-tts-server` at all. |
| `run_llama_embedding.sh` | Docker embedding server for RAG | RAG is off by default: a third model does not fit next to Gemma and Pocket TTS on an 8 GB board. |

Both still carry their NVIDIA copyright headers and are unmodified.

If you want the old three-model pipeline back for an A/B, set
`pipeline.mode: "split"` in `config/settings.yaml`, point `vision.vlm_base_url`
at whatever `run_llama_cpp.sh` starts, and install `faster-whisper` — see the
optional block in `requirements.txt`.
