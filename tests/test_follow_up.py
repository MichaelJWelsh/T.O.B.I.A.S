import sys

import pytest

import tobias.stt
from tobias.config import settings

PACKAGE = sys.modules["tobias.stt"]


@pytest.fixture
def mic(monkeypatch):
    """Drive listen() from a scripted list of transcripts, with a clock the test controls."""
    clock = {"now": 0.0}
    monkeypatch.setattr(settings, "stt_wake_word", "Tobias")
    monkeypatch.setattr(PACKAGE, "load", lambda: None)
    monkeypatch.setattr(PACKAGE.time, "monotonic", lambda: clock["now"])

    def script(*utterances, gap=0.0):
        def segments():
            for i, utterance in enumerate(utterances):
                if i:
                    clock["now"] += gap  # silence between one utterance and the next
                yield utterance

        monkeypatch.setattr(PACKAGE, "segments", segments)
        monkeypatch.setattr(PACKAGE, "transcribe", lambda audio: audio)
        return clock

    return script


def test_an_utterance_without_the_wake_word_is_ignored(mic):
    mic("What is the weather?", "Nothing to do with him either.")
    assert list(tobias.stt.listen()) == []


def test_a_reply_within_the_window_needs_no_wake_word(mic):
    mic("Tobias, what is the weather?", "And the time?")
    assert list(tobias.stt.listen()) == ["Tobias, what is the weather?", "And the time?"]


def test_the_window_closes_once_the_silence_outlasts_it(mic):
    mic("Tobias, what is the weather?", "And the time?", gap=settings.stt_follow_up_s + 1)
    assert list(tobias.stt.listen()) == ["Tobias, what is the weather?"]


def test_the_window_opens_when_the_reply_ends_not_when_the_user_stops(mic):
    clock = mic("Tobias, what is the weather?", "And the time?")
    stream = tobias.stt.listen()
    next(stream)
    # A long spoken reply: the caller only comes back for more once speak() has returned.
    clock["now"] = 30.0
    assert next(stream) == "And the time?"


def test_zero_always_requires_the_wake_word(mic, monkeypatch):
    monkeypatch.setattr(settings, "stt_follow_up_s", 0.0)
    mic("Tobias, what is the weather?", "And the time?")
    assert list(tobias.stt.listen()) == ["Tobias, what is the weather?"]
