# Fork plan: reachy-mini-gguf-assistant

(Compiled from the scouting audit + Phase 0 validation. Phase 0 PASSED on the
target board: one /v1/chat/completions message with BOTH an input_audio part
and an image_url part → Gemma 4 E2B answered using both modalities.)

## Identity
- Fork of NVIDIA-AI-IOT/reachy-mini-jetson-assistant (Apache-2.0). Working name:
  reachy-mini-gguf-assistant (no "NVIDIA" in the name - Apache §6).
- Replacements: faster-whisper STT -> gone (audio-direct to Gemma);
  Cosmos-Reason2 VLM -> Gemma 4 E2B (same mmproj does vision);
  Kokoro ONNX TTS -> EryriLabs/pocket-tts-GGUF via our llama-tts-server.
- Keep: Silero VAD, face tracking, MovementManager, speaking movements,
  camera, web UI, reachy SDK glue.

## License obligations
- Keep their LICENSE + all NVIDIA SPDX headers; add "Modified by" notice to
  every changed file (Apache §4(b)).
- THIRD-PARTY-NOTICES.md: remove Kokoro/faster-whisper/Cosmos + the GPL
  subprocess section (app/tts_worker.py is deleted); add Kyutai Pocket TTS
  (CC-BY-4.0, attribution required) and llama.cpp (MIT).

## Config surface (settings.yaml)
- pipeline.mode: "unified" (default) | "split" (their original STT+VLM path,
  the A/B escape hatch - keep app/stt.py in tree for this)
- vision.vlm_base_url: "" (used by split mode)
- pipeline.transcribe_for_display: false (opt-in real transcripts, needs -np 2)
- tts: base_url http://127.0.0.1:8100, voice, max_seconds; retire
  first_chunk_words/max_chunk_words
- Ports: llama-server 8080 (their llm.base_url default), tts 8100, web UI 8090.
- rag.enabled: false; drop run_llama_embedding.sh.

## File-by-file
1. app/llm.py: add audio_b64 param to generate_stream(); emit
   {"type":"input_audio","input_audio":{"data":..., "format":"wav"}} alongside
   existing image_url parts. Strip Cosmos sentence from settings.yaml system
   prompt (keep the "ignore the image unless asked" lines). --reasoning off
   mandatory server-side.
2. app/pipeline.py: segment_to_wav_b64() (mirror save_wav, 16kHz mono s16 into
   BytesIO). Transcript placeholder: broadcast {"type":"transcript",
   "text":"(spoken)", "audio":true, "duration":X} on utterance end;
   static/index.html onTranscript renders a duration chip (~3 lines).
   Replace filler gate with empty-content skip.
3. NEW app/tts_client.py (replaces app/tts.py usage; delete app/tts_worker.py):
   class LlamaTTSClient with load()/health_check()/synthesize(text)->{"audio":
   np.int16, "sample_rate": int}/unload(). POST /v1/audio/speech
   {input, voice, response_format:"wav", max_seconds}; READ SAMPLE RATE FROM
   THE WAV HEADER (model-dependent, never hardcode). max_seconds heuristic:
   max(4.0, min(cfg, len(text)/13.0 + 3.0)). 4096-char input cap.
4. NEW app/sentence_split.py: port split_sentences/is_speakable + abbreviation
   guard from dwain-barnes/llama-tts-server scripts/voice-chat.py. Replace the
   3-word/8-word chunker in run_web_vision_chat.py:505-513 and
   pipeline.py:820-828 with sentence-based chunking.
5. pipeline.tts_player: pipeline synthesis ahead of playback (synth thread +
   play queue, as voice-chat.py). KEEP on_audio_start on first PLAYED chunk /
   on_audio_end in finally EXACTLY (gesture contract;
   tests/test_tts_player_callbacks.py pins it, must stay green).
6. Launch layer: NEW start.sh adapted from dwain-barnes/jetson-voice-assistant
   (native binaries, NOT their Docker - their pinned image predates server-side
   input_audio routing and lacks llama-tts-server entirely):
   drop_caches before EACH CUDA load; curl -fs health gating (503-while-loading);
   strict sequence llama-server -> tts -> app; CUDA_VISIBLE_DEVICES="" for CPU
   TTS; warm-up POSTs; pin llama.cpp 9f0d017 and assert it. setup.sh with the
   prebuilt-tarball fast path (jetson-voice-assistant v1.0.0 asset, sha256
   a0cf9b0650bdaf1d72ad93c87ac8a900c9d612f2855929a050da665c6b5a0826).
   Retire run_llama_cpp.sh / run_llama_embedding.sh.
7. Memory: Gemma UD-Q2_K_XL + mmproj-F16 + 2048 ctx + Pocket TTS GPU =
   ~4.0 GiB GPU, ~6.8 GiB total, ~0.6 GiB headroom. Phase-4 lever: call Silero
   ONNX via onnxruntime directly (drop torch import, ~0.6 GiB). Never raise ctx
   past 2048.

## Phases
- P1 [no robot]: tts_client + sentence_split + llm audio param + tts_player
  pipelining + start.sh/setup.sh + transcript placeholder. Verify: 26 existing
  tests green (they run on Windows); new unit tests for tts_client (mock
  server), sentence chunking, wav_b64; integration against the REAL Jetson
  servers over LAN (192.168.0.214:8090 Gemma / :8100 TTS) using WAV files -
  no mic needed.
- P2 [Jetson+camera]: vision A/B unified vs split. P3 [Jetson+camera]: web UI.
- P4 [ROBOT]: gestures, AEC retune, torch removal. P5 [ROBOT]: barge-in
  (own project; reuse llama-tts-server /api/interrupt design).

## Risks being managed
- Build-tag lock-in: pin + assert 9f0d017. Q2 vision quality: split-mode A/B.
- Python 3.10 / numpy 1.26 pin on Jetson (their onnxruntime-gpu wheel):
  keep runtime code 3.10-compatible, no numpy-2-only APIs.
- NanoOWL owns the user's GPU at boot: session systemctl stop, never disable.
