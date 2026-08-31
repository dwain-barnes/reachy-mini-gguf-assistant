# SPDX-FileCopyrightText: Copyright (c) 2026 the reachy-mini-gguf-assistant contributors
# SPDX-License-Identifier: Apache-2.0

"""LlamaTTSClient against a real (tiny) HTTP server on localhost.

A stub server rather than a mocked httpx: the things worth pinning here are
wire-level — what we send, and that the sample rate comes out of the WAV header
the server chose rather than from a constant in our code.
"""

import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from app.tts_client import LlamaTTSClient, create_tts, max_seconds_for


def make_wav(samples, rate, channels=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return buf.getvalue()


class StubState:
    """What the stub should do next, and what it was asked for."""

    def __init__(self):
        self.health_status = 200
        self.speech_status = 200
        self.body = make_wav([100, -100, 200, -200], 24000)
        self.requests = []


class StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        state = self.server.state
        if self.path == "/health":
            self.send_response(state.health_status)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/v1/audio/voices":
            payload = json.dumps({"voices": ["default_voice", "narrator"]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        state = self.server.state
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        state.requests.append(json.loads(raw.decode("utf-8")))

        body = state.body if state.speech_status == 200 else b"model is busy"
        self.send_response(state.speech_status)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
    httpd.state = StubState()
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def client(server):
    host, port = server.server_address[:2]
    c = LlamaTTSClient(base_url=f"http://{host}:{port}", max_seconds=30.0)
    assert c.load()
    try:
        yield c
    finally:
        c.unload()


# ---------------------------------------------------------------- lifecycle

def test_load_fails_while_the_server_is_still_loading(server):
    server.state.health_status = 503
    host, port = server.server_address[:2]
    c = LlamaTTSClient(base_url=f"http://{host}:{port}")
    assert c.load() is False
    assert c.health_check() is False


def test_load_fails_when_nothing_is_listening():
    c = LlamaTTSClient(base_url="http://127.0.0.1:1")
    assert c.load() is False


def test_health_check_follows_the_server(client, server):
    assert client.health_check() is True
    server.state.health_status = 503
    assert client.health_check() is False


def test_synthesize_before_load_is_an_error():
    c = LlamaTTSClient(base_url="http://127.0.0.1:1")
    r = c.synthesize("hello")
    assert r["audio"] is None
    assert "not loaded" in r["error"]


# ---------------------------------------------------------------- synthesis

def test_sample_rate_comes_from_the_wav_header(client, server):
    # Nothing about 22050 is special; the point is that it is not the 24000 a
    # previous backend used, and the client reports what the server sent.
    server.state.body = make_wav([1, 2, 3, 4, 5], 22050)
    r = client.synthesize("Hello there.")
    assert r["sample_rate"] == 22050
    assert client.sample_rate == 22050
    assert r["audio"].dtype == np.int16
    assert r["audio"].tolist() == [1, 2, 3, 4, 5]


def test_request_shape(client, server):
    client.synthesize("Hello there.")
    sent = server.state.requests[-1]
    assert sent["input"] == "Hello there."
    assert sent["response_format"] == "wav"
    assert sent["max_seconds"] == pytest.approx(4.0)
    assert "voice" not in sent          # empty voice = server's own default


def test_voice_is_sent_when_configured(server):
    host, port = server.server_address[:2]
    c = create_tts(base_url=f"http://{host}:{port}", voice="narrator")
    assert c.load()
    c.synthesize("Hello.")
    assert server.state.requests[-1]["voice"] == "narrator"
    c.unload()


def test_no_cap_is_sent_when_max_seconds_is_zero(server):
    host, port = server.server_address[:2]
    c = LlamaTTSClient(base_url=f"http://{host}:{port}", max_seconds=0)
    assert c.load()
    c.synthesize("Hello.")
    assert "max_seconds" not in server.state.requests[-1]
    c.unload()


def test_stereo_is_mixed_down_to_mono(client, server):
    server.state.body = make_wav([100, 300, 200, 400], 16000, channels=2)
    r = client.synthesize("Hello.")
    assert r["audio"].tolist() == [200, 300]
    assert r["sample_rate"] == 16000


def test_empty_and_oversized_text_never_reach_the_server(client, server):
    before = len(server.state.requests)
    assert client.synthesize("")["audio"] is None
    assert client.synthesize("   ")["audio"] is None
    long_one = client.synthesize("x" * 5000)
    assert long_one["audio"] is None
    assert "too long" in long_one["error"]
    assert len(server.state.requests) == before


def test_http_error_is_reported_not_raised(client, server):
    server.state.speech_status = 500
    r = client.synthesize("Hello.")
    assert r["audio"] is None
    assert "500" in r["error"]


def test_non_wav_response_is_rejected(client, server):
    server.state.body = b"{\"error\": \"no\"}"
    r = client.synthesize("Hello.")
    assert r["audio"] is None
    assert "WAV" in r["error"]


def test_empty_wav_is_rejected(client, server):
    server.state.body = make_wav([], 24000)
    r = client.synthesize("Hello.")
    assert r["audio"] is None
    assert "empty" in r["error"]


def test_synthesize_to_file_round_trips(client, server, tmp_path):
    server.state.body = make_wav([7, -7, 21], 22050)
    out = tmp_path / "reply.wav"
    assert client.synthesize_to_file("Hello.", str(out)) is True
    with wave.open(str(out), "rb") as wf:
        assert wf.getframerate() == 22050
        assert wf.getnchannels() == 1
        assert np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).tolist() == [7, -7, 21]


def test_voices_lists_what_the_server_offers(client):
    assert client.voices() == ["default_voice", "narrator"]


# ---------------------------------------------------------- max_seconds cap

def test_max_seconds_scales_with_the_sentence():
    # ~13 characters a second, plus 3s of slack, floored at 4s and never above
    # the configured cap.
    assert max_seconds_for("Hi.", 30.0) == 4.0
    assert max_seconds_for("x" * 130, 30.0) == pytest.approx(13.0)
    assert max_seconds_for("x" * 10000, 30.0) == 30.0
    assert max_seconds_for("anything", 0) == 0.0
    assert max_seconds_for("x" * 130, 5.0) == 5.0
    assert max_seconds_for("", 30.0) == 4.0
