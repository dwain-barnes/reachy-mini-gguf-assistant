# SPDX-FileCopyrightText: Copyright (c) 2026 the reachy-mini-gguf-assistant contributors
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
# New in the reachy-mini-gguf-assistant fork of
# NVIDIA-AI-IOT/reachy-mini-jetson-assistant. Replaces the Kokoro ONNX client
# (app/tts.py) and its GPL-isolating subprocess worker (app/tts_worker.py),
# both of which are deleted here.

"""TTS — HTTP client for llama-tts-server (Pocket TTS GGUF).

Speech is a separate process already: llama-tts-server holds the model warm and
answers ``POST /v1/audio/speech`` with a WAV. This client keeps the same
``load() / synthesize() / health_check() / unload()`` shape the pipeline
expects, so ``app/pipeline.py`` neither knows nor cares which backend is
speaking.

The sample rate is read from the WAV header of every response and never
assumed. It is a property of the model the server was started with, and a
hardcoded rate is how audio ends up played at the wrong pitch.
"""

import io
import wave
from typing import Any, Dict, Optional

import httpx
import numpy as np

# Refuse absurd inputs rather than making the server chew through them. A
# sentence is what gets sent here; anything near this is a bug upstream.
MAX_INPUT_CHARS = 4096

# Speech runs at roughly 13 characters a second. Used to size the server-side
# runaway cap per sentence instead of paying the full cap on every short one.
CHARS_PER_SECOND = 13.0
MIN_MAX_SECONDS = 4.0
MAX_SECONDS_SLACK = 3.0


def max_seconds_for(text: str, cap: Optional[float]) -> float:
    """Scale the runaway cap to the text, never above ``cap``.

    Returns 0.0 when ``cap`` is falsy, which means "send no cap and let the
    server use whatever it was started with".
    """
    if not cap:
        return 0.0
    scaled = len(text) / CHARS_PER_SECOND + MAX_SECONDS_SLACK
    return max(MIN_MAX_SECONDS, min(float(cap), scaled))


def decode_wav(blob: bytes) -> Dict[str, Any]:
    """Decode a mono/stereo 16-bit WAV into int16 samples plus its real rate."""
    with wave.open(io.BytesIO(blob), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if width != 2:
        raise ValueError("expected 16-bit PCM, got %d-bit" % (width * 8))

    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        # Playback downstream is mono; average the channels rather than
        # dropping one, which would halve the volume of a hard-panned voice.
        usable = (audio.size // channels) * channels
        audio = audio[:usable].reshape(-1, channels).mean(axis=1).astype(np.int16)
    return {"audio": audio, "sample_rate": rate}


class LlamaTTSClient:
    """Speech from a warm llama-tts-server over HTTP."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8100",
        voice: str = "",
        max_seconds: float = 30.0,
        timeout: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.voice = voice or ""
        self.max_seconds = max_seconds
        self.timeout = timeout
        self.backend_name = "Pocket TTS"
        self.sample_rate: Optional[int] = None
        self._loaded = False
        self._client: Optional[httpx.Client] = None

    # ---------------------------------------------------------------- lifecycle

    def load(self) -> bool:
        """Confirm the speech server is up and past its own model load.

        llama-tts-server answers /health with 503 while it is still loading, so
        anything other than 200 means "not ready", not "not there".
        """
        try:
            self._client = httpx.Client(timeout=self.timeout)
            r = self._client.get(f"{self.base_url}/health", timeout=10.0)
            if r.status_code != 200:
                print(f"TTS server not ready ({r.status_code}) at {self.base_url}")
                self._close()
                return False
        except Exception as e:
            print(f"TTS server unreachable at {self.base_url}: {e}")
            self._close()
            return False
        self._loaded = True
        return True

    def health_check(self) -> bool:
        if not self._loaded or self._client is None:
            return False
        try:
            return self._client.get(f"{self.base_url}/health", timeout=5.0).status_code == 200
        except Exception:
            return False

    def unload(self):
        self._loaded = False
        self._close()

    def _close(self):
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    # ---------------------------------------------------------------- synthesis

    def synthesize(self, text: str) -> Dict[str, Any]:
        """Speak ``text``. Returns {"audio": int16 ndarray, "sample_rate": int}.

        On any failure the audio is None and "error" explains why. The pipeline
        treats that as "say nothing" and carries on, so a single failed
        sentence never takes the conversation down with it.
        """
        text = (text or "").strip()
        if not text:
            return {"audio": None, "error": "Empty"}
        if len(text) > MAX_INPUT_CHARS:
            return {"audio": None, "error": f"Text too long ({len(text)} chars)"}
        if not self._loaded or self._client is None:
            return {"audio": None, "error": "TTS server not loaded"}

        payload: Dict[str, Any] = {"input": text, "response_format": "wav"}
        cap = max_seconds_for(text, self.max_seconds)
        if cap:
            payload["max_seconds"] = cap
        # OpenAI's field name. Left out entirely when unset, so the server falls
        # back to the reference voice it was started with.
        if self.voice:
            payload["voice"] = self.voice

        try:
            r = self._client.post(f"{self.base_url}/v1/audio/speech", json=payload)
            if r.status_code != 200:
                detail = r.text[:200] if r.text else ""
                return {"audio": None, "error": f"HTTP {r.status_code} {detail}"}
            blob = r.content
        except Exception as e:
            return {"audio": None, "error": str(e)}

        if not blob.startswith(b"RIFF"):
            return {"audio": None, "error": "server did not return a WAV"}

        try:
            result = decode_wav(blob)
        except Exception as e:
            return {"audio": None, "error": f"bad WAV: {e}"}

        if result["audio"].size == 0:
            return {"audio": None, "error": "server returned an empty WAV"}

        self.sample_rate = result["sample_rate"]
        return result

    def synthesize_to_file(self, text: str, path: str) -> bool:
        r = self.synthesize(text)
        if r.get("audio") is None:
            return False
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(r["sample_rate"])
            wf.writeframes(r["audio"].tobytes())
        return True

    def voices(self) -> list:
        """Names the server will accept for ``voice``. Empty if it has none."""
        if self._client is None:
            return []
        try:
            r = self._client.get(f"{self.base_url}/v1/audio/voices", timeout=10.0)
            if r.status_code != 200:
                return []
            data = r.json()
        except Exception:
            return []
        if isinstance(data, dict):
            items = data.get("voices") or data.get("data") or []
        else:
            items = data
        names = []
        for item in items:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                name = item.get("name") or item.get("id")
                if name:
                    names.append(name)
        return names


def create_tts(
    base_url: str = "http://127.0.0.1:8100",
    voice: str = "",
    max_seconds: float = 30.0,
    timeout: float = 300.0,
    **_kwargs,
) -> LlamaTTSClient:
    """Create the TTS backend (llama-tts-server over HTTP)."""
    return LlamaTTSClient(
        base_url=base_url, voice=voice, max_seconds=max_seconds, timeout=timeout,
    )
