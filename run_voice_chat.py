#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by the reachy-mini-gguf-assistant contributors, 2026:
# unified mode sends the utterance itself to the model and never loads
# faster-whisper; speech comes from llama-tts-server.

"""
Voice Chat — speak anytime, dynamic recording.

  unified (default):  Mic -> Silero VAD -> Gemma (hears the audio) -> TTS -> Speaker
  split:              Mic -> Silero VAD -> STT -> (RAG) -> LLM -> TTS -> Speaker

RAG needs a transcript, so it only applies in split mode.

Usage:
  python3 run_voice_chat.py            # with RAG (split mode only)
  python3 run_voice_chat.py --no-rag   # without RAG
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config import Config
from app.audio import find_alsa_device
from app.llm import LLM
from app.tts_client import create_tts
from app.pipeline import (
    AUDIO_PROMPT, SAMPLE_RATE, MicRecorder, warmup_stt, vad_loop,
    segment_to_wav_b64, stream_and_speak, load_silero,
)
from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    config = Config.load()
    unified = (config.pipeline.mode or "unified").lower() != "split"
    # Retrieval needs words to search with, and in unified mode there are none.
    use_rag = "--no-rag" not in sys.argv and not unified
    active_system_prompt = config.llm.system_prompt if use_rag else config.llm.system_prompt_no_rag

    console.print(Panel.fit(
        "[bold cyan]Voice Chat[/bold cyan]\n"
        "Speak anytime — auto-detects speech\n"
        f"[dim]{'audio straight to the model' if unified else 'STT + LLM (split)'}"
        f"  |  {'RAG on' if use_rag else 'RAG off'}  |  Ctrl-C to quit[/dim]",
        border_style="cyan",
    ))

    # ── Audio setup ──────────────────────────────────────────────
    result = find_alsa_device(name_hint=config.audio.input_device or "Reachy Mini Audio")
    if not result:
        console.print("[red]No mic found![/red]")
        return
    card, dev, mic_name = result
    hw = f"hw:{card},{dev}"
    console.print(f"  Mic: {hw} ({mic_name})")

    # ── Load models ──────────────────────────────────────────────
    console.print("\n[bold]Loading...[/bold]")

    stt = None
    if not unified:
        from app.stt import STT
        stt = STT(
            model=config.stt.model, device=config.stt.device,
            compute_type=config.stt.compute_type, language=config.stt.language,
            beam_size=config.stt.beam_size,
        )
        stt.load()
        console.print(f"  ✓ STT (faster-whisper, {config.stt.model})")
        console.print("    CUDA warmup...", end=" ")
        console.print(f"done ({warmup_stt(stt):.1f}s)")
    else:
        console.print("  ✓ No STT — the model hears the microphone itself")

    silero_model = load_silero(console)

    llm = LLM(
        model=config.llm.model, base_url=config.llm.base_url,
        backend=config.llm.backend, max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature, timeout=config.llm.timeout,
        system_prompt=active_system_prompt,
    )
    llm.load()
    console.print(f"  ✓ LLM ({llm.model})")

    tts = create_tts(
        base_url=config.tts.base_url, voice=config.tts.voice,
        max_seconds=config.tts.max_seconds, timeout=config.tts.timeout,
    )
    tts = tts if tts.load() else None
    if tts:
        console.print(f"  ✓ TTS ({tts.backend_name} at {config.tts.base_url})")
    else:
        console.print("  ⚠ TTS unavailable")

    rag = None
    if use_rag and config.rag.enabled:
        try:
            from app.rag import KnowledgeBase, RAGRetriever
            kb = KnowledgeBase(
                persist_dir=config.rag.persist_dir,
                embedding_backend=config.rag.embedding_backend,
                embedding_model=config.rag.embedding_model,
                embedding_base_url=config.rag.embedding_base_url,
                chunk_size=config.rag.chunk_size,
                chunk_overlap=config.rag.chunk_overlap,
            )
            count, rebuilt = kb.sync_directory(config.rag.knowledge_dir)
            rag = RAGRetriever(kb, config.rag.n_results, config.rag.min_relevance)
            status = f"rebuilt, {count} chunks" if rebuilt else f"{count} chunks, cached"
            console.print(f"  ✓ RAG ({status})")
        except Exception as e:
            console.print(f"  ⚠ RAG: {e}")

    # ── Start mic ────────────────────────────────────────────────
    effective_chunk_ms = 32
    mic = MicRecorder(console, chunk_ms=effective_chunk_ms)
    if not mic.start(hw, config.audio.input_device or "Reachy Mini Audio"):
        console.print("[red]Cannot start recording! Check mic.[/red]")
        return

    console.print("\n[green bold]Ready — speak anytime![/green bold]\n")

    # ── Main loop ────────────────────────────────────────────────
    try:
        for segment in vad_loop(mic, console, vad_cfg=config.vad, silero=silero_model):
            audio_b64 = None
            dt_stt = 0.0

            if unified:
                audio_b64 = segment_to_wav_b64(segment.raw_chunks)
                text = AUDIO_PROMPT
                console.print(f"  [green]You:[/green] [dim](spoken, {segment.duration:.1f}s)[/dim]")
            else:
                t_stt = time.perf_counter()
                result = stt.transcribe(segment.audio, sample_rate=SAMPLE_RATE)
                text = result.get("text", "").strip()
                dt_stt = time.perf_counter() - t_stt

                if not text:
                    err = result.get("error", "")
                    console.print(
                        f"[dim]  (not recognized — {segment.duration:.1f}s, "
                        f"rms={segment.rms:.4f}{', err='+err if err else ''})[/dim]"
                    )
                    mic.resume()
                    continue

                console.print(f'  [green]You:[/green] "{text}"')

            prompt = text
            dt_rag = 0.0
            if rag:
                t_rag = time.perf_counter()
                docs = rag.kb.search(text, n_results=rag.n_results)
                relevant = [d for d in docs if d.get("distance", 2) < (2 - rag.min_relevance * 2)]
                dt_rag = time.perf_counter() - t_rag
                if relevant:
                    for j, d in enumerate(relevant):
                        score = 1 - d["distance"]
                        snippet = d["content"][:80].replace("\n", " ")
                        console.print(f"  [dim]  chunk{j+1} [{score:.2f}]: {snippet}...[/dim]")
                    ctx = "\n\n".join(d["content"] for d in relevant)
                    prompt = (
                        "Answer using ONLY the facts below. Do not invent names or details."
                        f"\n\n{ctx}\n\nQuestion: {text}"
                    )
                else:
                    console.print("  [dim]  (no relevant chunks)[/dim]")

            console.print("  [magenta]Assistant:[/magenta] ", end="")
            sys.stdout.flush()

            full_resp, dt_llm, ttft = stream_and_speak(
                llm, tts, prompt, active_system_prompt, mic.pa_sink,
                audio_b64=audio_b64,
            )
            console.print()

            timing = "  [dim]audio direct" if unified else f"  [dim]STT {dt_stt:.1f}s"
            if rag:
                timing += f" | RAG {dt_rag:.1f}s"
            if ttft is not None:
                toks = len(full_resp.split())
                timing += f" | TTFT {ttft:.1f}s | LLM {dt_llm:.1f}s ~{toks/(dt_llm or 1):.0f}w/s"
            else:
                timing += " | LLM no response"
            timing += "[/dim]"
            console.print(timing)

            mic.resume()

    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
    finally:
        mic.stop()
        if stt:
            stt.unload()
        llm.unload()
        if tts:
            tts.unload()


if __name__ == "__main__":
    main()
