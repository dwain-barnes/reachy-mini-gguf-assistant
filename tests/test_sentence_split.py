# SPDX-FileCopyrightText: Copyright (c) 2026 the reachy-mini-gguf-assistant contributors
# SPDX-License-Identifier: Apache-2.0

"""Sentence splitting for the TTS stream, including the abbreviation guards."""

from app.sentence_split import is_speakable, split_sentences


def test_splits_on_terminators_and_keeps_the_remainder():
    sentences, rest = split_sentences("Hello there. How are you? I am fine")
    assert sentences == ["Hello there.", "How are you?"]
    assert rest == "I am fine"


def test_nothing_complete_yet():
    sentences, rest = split_sentences("Still writing the fir")
    assert sentences == []
    assert rest == "Still writing the fir"


def test_newline_ends_a_sentence_without_punctuation():
    sentences, rest = split_sentences("First line\nSecond line\n")
    assert sentences == ["First line", "Second line"]
    assert rest == ""


def test_closing_quote_or_bracket_stays_with_the_sentence():
    sentences, rest = split_sentences('He said "go now." Then he left. ')
    assert sentences == ['He said "go now."', "Then he left."]
    assert rest == ""


def test_abbreviations_do_not_end_a_sentence():
    for text in ("Dr. Jones is here. ", "See e.g. this one. ", "Acme Ltd. did it. "):
        sentences, _ = split_sentences(text)
        assert len(sentences) == 1, text
        assert sentences[0] == text.strip()


def test_initials_do_not_end_a_sentence():
    sentences, rest = split_sentences("J. R. Tolkien wrote it. ")
    assert sentences == ["J. R. Tolkien wrote it."]
    assert rest == ""


def test_ellipsis_and_multiple_marks_split_once():
    sentences, rest = split_sentences("Really?! Yes... Fine. ")
    assert sentences == ["Really?!", "Yes...", "Fine."]
    assert rest == ""


def test_streaming_a_reply_token_by_token_yields_each_sentence_once():
    reply = "Hi there. I am a robot from Dr. Pollen. Nice to meet you!"
    buffer = ""
    spoken = []
    for char in reply:
        buffer += char
        sentences, buffer = split_sentences(buffer)
        spoken.extend(sentences)
    tail = buffer.strip()
    if tail:
        spoken.append(tail)
    assert spoken == [
        "Hi there.",
        "I am a robot from Dr. Pollen.",
        "Nice to meet you!",
    ]
    assert "".join(spoken).replace(" ", "") == reply.replace(" ", "")


def test_is_speakable_rejects_punctuation_only_fragments():
    assert is_speakable("Hello.")
    assert is_speakable("42")
    assert not is_speakable("-")
    assert not is_speakable("  ...  ")
    assert not is_speakable("")
