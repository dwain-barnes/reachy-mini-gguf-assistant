# SPDX-FileCopyrightText: Copyright (c) 2026 the reachy-mini-gguf-assistant contributors
# SPDX-License-Identifier: Apache-2.0

"""The utterance-to-WAV path, and that synthesis really does run ahead of play."""

import base64
import io
import queue
import threading
import time
import wave

import numpy as np

from app import pipeline


# ------------------------------------------------------- segment_to_wav_b64

def test_segment_to_wav_b64_round_trips_the_samples():
    samples = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
    chunks = [samples[:2].tobytes(), samples[2:].tobytes()]

    blob = base64.b64decode(pipeline.segment_to_wav_b64(chunks))

    assert blob.startswith(b"RIFF")
    with wave.open(io.BytesIO(blob), "rb") as wf:
        assert wf.getnchannels() == pipeline.CHANNELS == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == pipeline.SAMPLE_RATE == 16000
        decoded = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    assert decoded.tolist() == samples.tolist()


def test_segment_to_wav_b64_matches_save_wav(tmp_path):
    """Same bytes as the on-disk path it mirrors, header included."""
    chunks = [np.arange(-50, 50, dtype=np.int16).tobytes()]
    path = tmp_path / "utterance.wav"
    pipeline.save_wav(chunks, str(path))

    assert base64.b64decode(pipeline.segment_to_wav_b64(chunks)) == path.read_bytes()


def test_segment_to_wav_b64_handles_an_empty_capture():
    blob = base64.b64decode(pipeline.segment_to_wav_b64([]))
    with wave.open(io.BytesIO(blob), "rb") as wf:
        assert wf.getnframes() == 0


def test_segment_to_wav_b64_is_ascii_base64():
    text = pipeline.segment_to_wav_b64([np.zeros(8, dtype=np.int16).tobytes()])
    assert isinstance(text, str)
    text.encode("ascii")  # would raise if it were not


# ------------------------------------------------------------- pipelining

class SlowTTS:
    """Records when each synthesis starts, so overlap is visible."""

    def __init__(self, events, lock, delay=0.05):
        self.events = events
        self.lock = lock
        self.delay = delay

    def synthesize(self, text):
        with self.lock:
            self.events.append(("synth-start", text))
        time.sleep(self.delay)
        with self.lock:
            self.events.append(("synth-end", text))
        return {"audio": np.ones(4, dtype=np.int16), "sample_rate": 16000}


def test_next_sentence_is_synthesized_while_the_previous_one_plays(monkeypatch):
    events = []
    lock = threading.Lock()

    def slow_play(audio, sample_rate, sink=None):
        with lock:
            events.append(("play-start", None))
        time.sleep(0.05)
        with lock:
            events.append(("play-end", None))

    monkeypatch.setattr(pipeline, "play_audio", slow_play)

    q = queue.Queue()
    q.put("one.")
    q.put("two.")
    q.put(None)
    pipeline.tts_player(SlowTTS(events, lock), q)

    names = [e[0] for e in events]
    # The second synthesis must start before the first playback finishes;
    # that overlap is the whole point of the change.
    assert names.index("synth-start") == 0
    second_synth = [i for i, e in enumerate(events) if e == ("synth-start", "two.")][0]
    first_play_end = names.index("play-end")
    assert second_synth < first_play_end


def test_playback_order_matches_the_queue(monkeypatch):
    played = []
    monkeypatch.setattr(
        pipeline, "play_audio",
        lambda audio, sample_rate, sink=None: played.append(int(audio[0])),
    )

    class Counter:
        def __init__(self):
            self.n = 0

        def synthesize(self, text):
            self.n += 1
            return {"audio": np.full(2, self.n, dtype=np.int16), "sample_rate": 16000}

    q = queue.Queue()
    for word in ("one.", "two.", "three."):
        q.put(word)
    q.put(None)
    pipeline.tts_player(Counter(), q)

    assert played == [1, 2, 3]


def test_a_failed_sentence_does_not_stop_the_rest(monkeypatch):
    played = []
    monkeypatch.setattr(
        pipeline, "play_audio",
        lambda audio, sample_rate, sink=None: played.append(int(audio[0])),
    )

    class Flaky:
        def synthesize(self, text):
            if text == "bad.":
                raise RuntimeError("server said no")
            if text == "empty.":
                return {"audio": None, "error": "server returned an empty WAV"}
            return {"audio": np.full(2, 9, dtype=np.int16), "sample_rate": 16000}

    events = []
    q = queue.Queue()
    for word in ("bad.", "empty.", "good."):
        q.put(word)
    q.put(None)
    pipeline.tts_player(
        Flaky(), q,
        on_audio_start=lambda: events.append("start"),
        on_audio_end=lambda: events.append("end"),
    )

    assert played == [9]
    assert events == ["start", "end"]
