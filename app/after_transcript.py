# SPDX-FileCopyrightText: Copyright (c) 2026 the reachy-mini-gguf-assistant contributors
# SPDX-License-Identifier: Apache-2.0

"""Writing down what was said, after the answer has already been given.

In unified mode there is no speech-to-text anywhere in the chain and there is
not about to be: Gemma hears the microphone itself, which is the whole point.
The browser is left showing a chip - "spoken", and how long it lasted - where
the words should be.

So once the reply is finished and spoken, the same model is asked one extra
question: what did that clip say? The answer goes to the browser as a
`transcript_update` and the chip becomes the words. It is display only; the
reply the person heard was produced from the audio and is already on screen and
out of the speaker by the time any of this runs.

The conversation always wins. The turn counter is bumped the moment the VAD
hands over a new utterance, so a transcription that has not started yet is
never sent, and one already in flight has its connection closed under it. That
last part is what actually matters on this hardware: llama-server runs with a
single slot (`-np 1`), so a transcript still generating is a transcript
standing in the next question's way. Hanging up tells the server to stop
generating for a client that has gone.
"""

import http.client
import json
import socket
import threading
import urllib.parse

TRANSCRIBE_PROMPT = (
    "Transcribe the speech in this audio exactly, word for word. Output only "
    "the spoken words, nothing else."
)
TRANSCRIBE_MAX_TOKENS = 96
TRANSCRIBE_TIMEOUT = 60.0     # network timeout on the transcription request
TRANSCRIBE_MAX_CHARS = 600    # a runaway answer is not a transcript


class Cancelled(Exception):
    """The job was superseded before it could finish. Not an error."""


class Cancellable:
    """Somewhere for a background job to park the connection it is using.

    Whoever supersedes the job reaches in and hangs up. Closing the socket is
    what frees the model: it stops generating for a client that has gone, so
    the next real question is not queued behind an answer nobody wants any
    more.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._conn = None
        self._sock = None
        self.cancelled = False

    def attach(self, conn):
        """Hand over the connection, already dialled. Raises if the job was
        cancelled in the moment between the check and the dial.

        The socket is taken now rather than at cancel time because http.client
        lets go of it as soon as it decides the response will close the
        connection - and by then the only thing still holding it is the reader
        we are trying to interrupt.
        """
        with self._lock:
            if self.cancelled:
                raise Cancelled()
            self._conn = conn
            self._sock = getattr(conn, "sock", None)

    def cancel(self):
        with self._lock:
            self.cancelled = True
            conn, self._conn = self._conn, None
            sock, self._sock = self._sock, None
        # shutdown() before close(): shutting the connection down is what sends
        # the far end a FIN, which is how llama-server learns that the client
        # has gone and abandons the generation.
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def clean_transcript(text: str) -> str:
    """Tidy the model's answer without rewriting it.

    Whitespace, a pair of quotation marks it decided to add, a runaway that is
    plainly no longer a transcript. Nothing else: the words are the point, and
    a transcript that quietly edits what someone said is worse than none.
    """
    text = " ".join((text or "").split())
    while len(text) > 1 and text[0] == text[-1] and text[0] in "\"'“”":
        text = text[1:-1].strip()
    if text.startswith("“") and text.endswith("”"):
        text = text[1:-1].strip()
    return text[:TRANSCRIBE_MAX_CHARS].strip()


def transcribe_audio(base_url, audio_b64, handle=None, model="",
                     timeout=TRANSCRIBE_TIMEOUT):
    """Ask the model to write down what one clip of audio says.

    Streamed, even though nobody watches these tokens arrive, and raw
    http.client rather than the httpx the rest of the app uses. Both for the
    same reason: a streamed request over a connection we hold ourselves is the
    only shape that can be hung up on mid-generation. With a plain request the
    socket does not come back until the model has finished, by which time the
    GPU time we wanted to give back has already been spent.

    `handle` is where the caller parks that connection so a newer utterance can
    close it.
    """
    payload = {
        "model": model or "",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": TRANSCRIBE_PROMPT},
                {"type": "input_audio",
                 "input_audio": {"data": audio_b64, "format": "wav"}},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": TRANSCRIBE_MAX_TOKENS,
        "stream": True,
    }
    parts = urllib.parse.urlsplit(base_url)
    opener = (http.client.HTTPSConnection if parts.scheme == "https"
              else http.client.HTTPConnection)
    conn = opener(parts.hostname, parts.port, timeout=timeout)
    try:
        # Dial first, then hand the live connection over: a handle holding a
        # socket that does not exist yet cannot hang up on anything.
        conn.connect()
        if handle is not None:
            handle.attach(conn)
        conn.request("POST", parts.path.rstrip("/") + "/v1/chat/completions",
                     body=json.dumps(payload).encode("utf-8"),
                     headers={"Content-Type": "application/json",
                              "Accept": "text/event-stream"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"llama-server answered {resp.status}")
        pieces = []
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except ValueError:
                continue
            for choice in obj.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    pieces.append(piece)
        return clean_transcript("".join(pieces))
    finally:
        try:
            conn.close()
        except Exception:
            pass


class AfterTranscriber:
    """Owns the turn counter, the worker thread and the broadcast.

    The pipeline is synchronous - one turn at a time in the main loop - so the
    only safe place for a second request to the model is a background thread
    that the loop never waits on. `begin_turn()` is called as each utterance
    arrives; `after_reply()` is called once that utterance's reply is finished
    and spoken, and returns immediately.
    """

    def __init__(self, broadcaster, base_url, model="", enabled=True, log=None):
        self.broadcaster = broadcaster
        self.base_url = base_url
        self.model = model
        self.enabled = enabled
        self._log = log or (lambda msg: None)
        self._lock = threading.Lock()
        self._turn = 0
        self._pending = None

    def begin_turn(self) -> int:
        """A new utterance has been segmented. Returns its turn number, and
        hangs up on whatever the previous turn left running."""
        with self._lock:
            self._turn += 1
            turn = self._turn
            pending, self._pending = self._pending, None
        if pending is not None:
            pending.cancel()
        return turn

    def is_current(self, turn: int) -> bool:
        with self._lock:
            return turn == self._turn

    def cancel(self):
        """Give up on anything in flight - shutdown, or a mode that no longer
        wants transcripts."""
        with self._lock:
            pending, self._pending = self._pending, None
        if pending is not None:
            pending.cancel()

    def after_reply(self, turn: int, audio_b64: str):
        """Write down turn `turn`, unless the conversation has moved on.

        Returns the worker thread (the tests wait on it); the main loop throws
        it away and goes back to listening.
        """
        if not self.enabled or not audio_b64:
            return None
        with self._lock:
            if turn != self._turn:
                return None             # somebody is already talking again
            handle = Cancellable()
            self._pending = handle
        thread = threading.Thread(target=self._work, args=(turn, audio_b64, handle),
                                  daemon=True, name=f"transcribe-{turn}")
        thread.start()
        return thread

    # -- worker ------------------------------------------------------------

    def _release(self, handle):
        with self._lock:
            if self._pending is handle:
                self._pending = None

    def _work(self, turn, audio_b64, handle):
        said = None
        try:
            if self.is_current(turn):
                said = transcribe_audio(self.base_url, audio_b64, handle,
                                        model=self.model)
            if said and (handle.cancelled or not self.is_current(turn)):
                said = None             # the conversation moved on while we asked
        except Cancelled:
            said = None
        except Exception as exc:
            said = None
            # Silent as far as the browser is concerned: a transcript that did
            # not arrive leaves the chip exactly as it was, which is the honest
            # thing for it to look like.
            if not handle.cancelled:
                self._log(f"transcription failed: {exc}")
        finally:
            self._release(handle)
        if said:
            self.broadcaster.send({"type": "transcript_update",
                                   "turnId": turn, "text": said})
        return said
