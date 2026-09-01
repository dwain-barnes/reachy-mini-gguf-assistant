# SPDX-FileCopyrightText: Copyright (c) 2026 the reachy-mini-gguf-assistant contributors
# SPDX-License-Identifier: Apache-2.0

"""Writing down a spoken turn after the reply has already been given.

A stub llama-server rather than a mocked client: what is worth pinning here is
wire-level and cancellation behaviour - what the request looks like, and what
happens to it when somebody starts talking again while it is in flight.
"""

import asyncio
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.after_transcript import (          # noqa: E402
    AfterTranscriber, Cancellable, Cancelled, TRANSCRIBE_MAX_CHARS,
    TRANSCRIBE_MAX_TOKENS, TRANSCRIBE_PROMPT, clean_transcript, transcribe_audio,
)
from app.web import Broadcaster              # noqa: E402


# ── stub model ───────────────────────────────────────────────────

class StubState:
    def __init__(self):
        self.requests = []
        self.reply = "what time is it"
        self.slow = False           # keep generating, like a model with a lot to say
        self.status = 200
        self.opened = threading.Event()
        self.hung_up_on = threading.Event()
        self.url = ""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "gemma"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        st = self.server.state
        st.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        if st.status != 200:
            self.send_response(st.status)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        st.opened.set()
        pieces = st.reply.split(" ") if not st.slow else ["tick"] * 300
        try:
            for piece in pieces:
                self.wfile.write(
                    ("data: " + json.dumps({"choices": [{"delta": {"content": piece + " "}}]})
                     + "\n\n").encode())
                self.wfile.flush()
                if st.slow:
                    time.sleep(0.1)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception:
            # What llama-server does when the client goes away: notice, and
            # stop generating for somebody who is no longer listening.
            st.hung_up_on.set()


@pytest.fixture
def stub():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.state = StubState()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address[:2]
    httpd.state.url = f"http://{host}:{port}"
    try:
        yield httpd.state
    finally:
        httpd.shutdown()
        httpd.server_close()


class FakeBroadcaster:
    """Just the one method the transcriber uses."""

    def __init__(self):
        self.sent = []
        self.arrived = threading.Event()

    def send(self, msg):
        self.sent.append(msg)
        self.arrived.set()


# ── the answer is tidied, not rewritten ──────────────────────────

@pytest.mark.parametrize("raw,want", [
    ("  what   time is it  ", "what time is it"),
    ('"what time is it"', "what time is it"),
    ("'what time is it'", "what time is it"),
    ("“what time is it”", "what time is it"),
    ("two\nlines", "two lines"),
    ("", ""),
    (None, ""),
])
def test_clean_transcript(raw, want):
    assert clean_transcript(raw) == want


def test_clean_transcript_cuts_a_runaway():
    assert len(clean_transcript("word " * 400)) <= TRANSCRIBE_MAX_CHARS


# ── the request ──────────────────────────────────────────────────

def test_the_request_asks_for_the_words_and_carries_the_clip(stub):
    said = transcribe_audio(stub.url, "QUJD", model="gemma")
    assert said == "what time is it"
    body = stub.requests[0]
    assert body["stream"] is True
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == TRANSCRIBE_MAX_TOKENS
    parts = body["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": TRANSCRIBE_PROMPT}
    assert parts[1]["input_audio"] == {"data": "QUJD", "format": "wav"}


def test_a_refusing_server_raises(stub):
    stub.status = 503
    with pytest.raises(RuntimeError):
        transcribe_audio(stub.url, "QUJD")


# ── the conversation always wins ─────────────────────────────────

def test_a_new_utterance_hangs_up_on_a_transcription_in_flight(stub):
    """The whole point of holding the connection: a new utterance frees the
    model now, rather than when the transcript happens to finish. What is
    asserted is what the model can see - the client it was generating for is
    gone - and that the reader thread then ends of its own accord."""
    stub.slow = True
    handle = Cancellable()
    box = {}

    def work():
        started = time.time()
        try:
            box["said"] = transcribe_audio(stub.url, "QUJD", handle)
        except Exception as exc:
            box["exc"] = exc
        box["elapsed"] = time.time() - started

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    assert stub.opened.wait(5), "the stub never started answering"
    time.sleep(0.3)
    handle.cancel()
    assert stub.hung_up_on.wait(5), "the model was never told to stop"
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert box["elapsed"] < 25, "cancelling waited for the whole generation"
    assert not box.get("said")


def test_attaching_to_a_cancelled_handle_raises():
    handle = Cancellable()
    handle.cancel()
    with pytest.raises(Cancelled):
        handle.attach(object())


def test_begin_turn_counts_and_cancels(stub):
    bus = FakeBroadcaster()
    t = AfterTranscriber(bus, stub.url)
    first = t.begin_turn()
    second = t.begin_turn()
    assert (first, second) == (1, 2)
    assert t.is_current(second) and not t.is_current(first)


def test_a_superseded_turn_is_never_sent(stub):
    """Rapid fire: the reply for turn one finishes, but by then the VAD has
    already handed over turn two. Nothing goes to the model at all."""
    bus = FakeBroadcaster()
    t = AfterTranscriber(bus, stub.url)
    turn = t.begin_turn()
    t.begin_turn()                       # somebody is talking again
    assert t.after_reply(turn, "QUJD") is None
    assert stub.requests == []
    assert bus.sent == []


def test_a_transcription_overtaken_in_flight_is_thrown_away(stub):
    stub.slow = True
    bus = FakeBroadcaster()
    t = AfterTranscriber(bus, stub.url)
    turn = t.begin_turn()
    thread = t.after_reply(turn, "QUJD")
    assert stub.opened.wait(5)
    t.begin_turn()                       # the next utterance arrives
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert bus.sent == [], "words from a turn nobody is looking at any more"


def test_the_words_reach_the_browser_keyed_to_their_turn(stub):
    bus = FakeBroadcaster()
    t = AfterTranscriber(bus, stub.url, model="gemma")
    turn = t.begin_turn()
    t.after_reply(turn, "QUJD").join(timeout=10)
    assert bus.sent == [{"type": "transcript_update", "turnId": turn,
                         "text": "what time is it"}]


def test_switched_off_asks_nothing(stub):
    bus = FakeBroadcaster()
    t = AfterTranscriber(bus, stub.url, enabled=False)
    assert t.after_reply(t.begin_turn(), "QUJD") is None
    assert stub.requests == []


def test_a_failure_is_silent(stub):
    """A transcript that did not arrive leaves the chip exactly as it was."""
    stub.status = 500
    logged = []
    bus = FakeBroadcaster()
    t = AfterTranscriber(bus, stub.url, log=logged.append)
    t.after_reply(t.begin_turn(), "QUJD").join(timeout=10)
    assert bus.sent == []
    assert logged and "transcription failed" in logged[0]


def test_cancel_on_shutdown_stops_a_transcription(stub):
    stub.slow = True
    bus = FakeBroadcaster()
    t = AfterTranscriber(bus, stub.url)
    thread = t.after_reply(t.begin_turn(), "QUJD")
    assert stub.opened.wait(5)
    t.cancel()
    assert stub.hung_up_on.wait(5)
    thread.join(timeout=10)
    assert bus.sent == []


# ── the message actually reaches a browser ───────────────────────

def test_transcript_update_fans_out_through_the_real_broadcaster(stub):
    """The pipeline thread is not the event loop thread, which is the whole
    reason Broadcaster exists; this pins that a transcript written down on a
    worker thread still lands in a client's queue."""
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    bus = Broadcaster()
    bus.set_loop(loop)
    queue = asyncio.Queue(maxsize=8)
    bus.register(queue)
    try:
        t = AfterTranscriber(bus, stub.url, model="gemma")
        turn = t.begin_turn()
        t.after_reply(turn, "QUJD").join(timeout=10)
        msg = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(queue.get(), 5), loop).result(10)
    finally:
        bus.unregister(queue)
        loop.call_soon_threadsafe(loop.stop)
    assert msg == {"type": "transcript_update", "turnId": turn,
                   "text": "what time is it"}
