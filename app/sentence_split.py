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
# NVIDIA-AI-IOT/reachy-mini-jetson-assistant. Ported from
# dwain-barnes/llama-tts-server (MIT), scripts/voice-chat.py.

"""Sentence splitting for a growing LLM token stream.

Replaces the fork parent's word-count chunker (3 words, then 8) which cut
speech mid-phrase and made the voice sound stitched together. Pocket TTS
carries prosody across a whole sentence, so a sentence is the right unit: the
first one starts speaking while the model is still writing the second.

Deliberately simple. Split after . ! ? (and any closing quote or bracket that
follows) when whitespace comes next, and after newlines. The only guards are
the abbreviation list and the rule that a lone capital followed by a dot is an
initial. Worst case a fragment gets spoken and the next one follows it, which
sounds like a short pause.
"""

import re

ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "st", "sr", "jr", "vs", "etc", "e.g",
    "i.e", "approx", "no", "fig", "inc", "ltd", "co", "op", "al",
})

_BOUNDARY = re.compile(r'([.!?]+["\'\)\]]*)(\s)|(\n+)')
_TRAILING_WORD = re.compile(r'([A-Za-z.]+)\.$')
_HAS_ALNUM = re.compile(r"[0-9A-Za-z]")


def ends_with_abbreviation(text: str) -> bool:
    """True when the trailing full stop belongs to an abbreviation or initial."""
    m = _TRAILING_WORD.search(text.rstrip())
    if not m:
        return False
    word = m.group(1).lower().rstrip(".")
    if len(word) == 1:  # initials: "J. Smith"
        return True
    return word in ABBREVIATIONS


def split_sentences(buffer: str):
    """Return (complete_sentences, remainder) for a growing text buffer.

    The remainder is whatever has not yet been terminated; feed it back in with
    the next tokens.
    """
    out = []
    start = 0
    for m in _BOUNDARY.finditer(buffer):
        end = m.end(1) if m.group(1) else m.end(3)
        candidate = buffer[start:end]
        if m.group(1) and ends_with_abbreviation(candidate):
            continue
        sentence = candidate.strip()
        if sentence:
            out.append(sentence)
        start = m.end()
    return out, buffer[start:]


def is_speakable(text: str) -> bool:
    """Skip fragments with no letters or digits (stray punctuation, bullets)."""
    return bool(_HAS_ALNUM.search(text))
