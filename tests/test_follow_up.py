import sys

import pytest

import tobias.stt
from tobias.config import settings

PACKAGE = sys.modules["tobias.stt"]


@pytest.fixture
def mic(monkeypatch):
    """Drive listen() from scripted transcripts, with both clocks under the test's control.

    `now` is the wall clock; `spoke` is when TOBIAS last stopped talking, which is what the
    follow-up window is actually measured from.
    """
    clock = {"now": 100.0, "spoke": float("-inf")}
    monkeypatch.setattr(settings, "stt_wake_word", "Tobias")
    monkeypatch.setattr(settings, "stt_follow_up_s", 20.0)
    monkeypatch.setattr(PACKAGE, "load", lambda: None)
    monkeypatch.setattr(PACKAGE.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(PACKAGE, "spoken_at", lambda: clock["spoke"])

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


def test_naming_him_is_always_enough(mic):
    mic("Tobias, what is the weather?")
    assert list(tobias.stt.listen()) == ["Tobias, what is the weather?"]


def test_a_reply_just_after_he_stops_needs_no_wake_word(mic):
    clock = mic("And the time?")
    clock["spoke"] = clock["now"]  # he has this moment finished speaking
    assert list(tobias.stt.listen()) == ["And the time?"]


def test_the_window_closes(mic):
    clock = mic("And the time?")
    clock["spoke"] = clock["now"] - settings.stt_follow_up_s - 1
    assert list(tobias.stt.listen()) == []


def test_a_long_reply_does_not_eat_the_window(mic):
    # The window runs from when he stopped, not when the user's question ended, so a rambling
    # answer must not shorten the chance to answer back.
    clock = mic("And the time?")
    clock["now"] += 30.0  # he talked for half a minute
    clock["spoke"] = clock["now"]
    assert list(tobias.stt.listen()) == ["And the time?"]


def test_the_window_is_judged_from_when_speech_began(mic):
    # segments() only yields once an utterance has closed. Judging by "now" would charge the
    # user for however long they spoke plus VAD_SILENCE_MS plus the transcription.
    clock = mic("x" * 16000)  # a full second of audio
    clock["spoke"] = clock["now"] - settings.stt_follow_up_s + 0.5
    assert list(tobias.stt.listen()) == ["x" * 16000], "speech that began inside the window counts"


def test_zero_always_requires_the_wake_word(mic):
    clock = mic("And the time?")
    clock["spoke"] = clock["now"]
    with pytest.MonkeyPatch.context() as m:
        m.setattr(settings, "stt_follow_up_s", 0.0)
        assert list(tobias.stt.listen()) == []
