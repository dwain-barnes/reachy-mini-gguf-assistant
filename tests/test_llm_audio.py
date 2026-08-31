# SPDX-FileCopyrightText: Copyright (c) 2026 the reachy-mini-gguf-assistant contributors
# SPDX-License-Identifier: Apache-2.0

"""The chat request the model actually receives when the user speaks."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.llm import LLM

AUDIO = "UklGRiQAAABXQVZF"      # not real audio; we only check it is passed through
IMAGE = "/9j/4AAQSkZJRg"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        payload = json.dumps({"data": [{"id": "gemma"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.server.requests.append(json.loads(self.rfile.read(length).decode("utf-8")))
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":" there."}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        body = b"".join(chunks)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def llm():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.requests = []
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address[:2]
    client = LLM(base_url=f"http://{host}:{port}", system_prompt="be brief")
    assert client.load()
    client.requests = httpd.requests
    try:
        yield client
    finally:
        httpd.shutdown()
        httpd.server_close()


def user_content(request):
    return request["messages"][-1]["content"]


def test_audio_is_sent_as_an_input_audio_part(llm):
    text = "".join(c for c, _ in llm.generate_stream("say hi", audio_b64=AUDIO))
    assert text == "Hello there."

    content = user_content(llm.requests[-1])
    assert content[0] == {"type": "text", "text": "say hi"}
    assert content[1] == {
        "type": "input_audio",
        "input_audio": {"data": AUDIO, "format": "wav"},
    }


def test_audio_and_images_travel_together(llm):
    list(llm.generate_stream("what do you see", images_b64=[IMAGE], audio_b64=AUDIO))

    content = user_content(llm.requests[-1])
    kinds = [part["type"] for part in content]
    # Audio before image: the utterance is the question, the frame is context.
    assert kinds == ["text", "input_audio", "image_url"]
    assert content[2]["image_url"]["url"] == f"data:image/jpeg;base64,{IMAGE}"


def test_a_text_only_turn_is_still_a_plain_string(llm):
    list(llm.generate_stream("just text"))
    assert user_content(llm.requests[-1]) == "just text"


def test_images_without_audio_are_unchanged(llm):
    list(llm.generate_stream("look", images_b64=[IMAGE]))
    kinds = [part["type"] for part in user_content(llm.requests[-1])]
    assert kinds == ["text", "image_url"]


def test_system_prompt_and_few_shot_survive_an_audio_turn(llm):
    few_shot = [
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello!"},
    ]
    list(llm.generate_stream("say hi", system_prompt="you are a robot",
                             few_shot=few_shot, audio_b64=AUDIO))

    messages = llm.requests[-1]["messages"]
    assert messages[0] == {"role": "system", "content": "you are a robot"}
    assert messages[1:3] == few_shot
    assert messages[3]["role"] == "user"
