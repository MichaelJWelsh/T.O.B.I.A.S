import sys

import numpy as np
import pytest

from tobias.config import settings
from tobias.tts import speak_stream
from tobias.tts.synthesize import SAMPLE_RATE

PLAYBACK = sys.modules["tobias.tts.playback"]
TTS = sys.modules["tobias.tts"]


class FakeStream:
    """Records what actually reached the device, so a cut-off sentence is visible."""

    def __init__(self):
        self.written = 0
        self.aborted = False

    def write(self, audio):
        self.written += len(audio)

    def abort(self):
        self.aborted = True

    def start(self):
        pass


@pytest.fixture
def device(monkeypatch):
    stream = FakeStream()
    monkeypatch.setattr(PLAYBACK, "_stream", lambda: stream)
    monkeypatch.setattr(TTS, "warm", lambda: None)
    monkeypatch.setattr(TTS, "synthesize", lambda text: np.zeros(SAMPLE_RATE, dtype=np.float32))
    return stream


def test_without_a_poll_the_whole_reply_plays(device):
    assert speak_stream(iter(["one.", "two.", "three."])) is False
    assert device.written == 3 * SAMPLE_RATE
    assert not device.aborted


def test_a_poll_that_never_fires_changes_nothing(device):
    assert speak_stream(iter(["one.", "two."]), lambda: False) is False
    assert device.written == 2 * SAMPLE_RATE


def test_an_immediate_interrupt_plays_nothing(device):
    assert speak_stream(iter(["one.", "two."]), lambda: True) is True
    assert device.written == 0
    assert device.aborted


def test_the_remaining_sentences_are_abandoned_not_queued(device):
    played = {"chunks": 0}

    def poll():
        played["chunks"] += 1
        return played["chunks"] > 12  # part way through the second sentence

    assert speak_stream(iter(["one.", "two.", "three."]), poll) is True
    assert device.written < 2 * SAMPLE_RATE, "the third sentence must never play"


def test_a_sentence_is_cut_off_part_way_through(device):
    assert speak_stream(iter(["a long sentence."]), lambda: device.written >= SAMPLE_RATE // 2) is True
    assert device.written < SAMPLE_RATE, "playback should have stopped mid-sentence"
    assert device.aborted, "the device buffer must be discarded, or he trails off"


def test_the_poll_runs_on_the_calling_thread(device):
    import threading

    caller = threading.current_thread()
    seen = []
    speak_stream(iter(["one."]), lambda: seen.append(threading.current_thread()) or False)
    assert seen and all(t is caller for t in seen), "GPU work must not move to another thread"


NEVER = float("-inf")


@pytest.fixture
def unspoken(monkeypatch):
    """Reset the "when did he last reply" clock. A sentinel, not a timestamp comparison —
    time.monotonic() has ~15ms resolution on Windows and a fake device replies inside one tick."""
    monkeypatch.setattr(TTS, "_spoken_at", NEVER)


def test_an_announcement_does_not_open_the_follow_up_window(device, unspoken):
    # The startup greeting is not a reply. If it moved the clock, the first thing the user said
    # would skip the wake word — and since every reply reopens the window, a whole session could
    # then run without ever naming him.
    TTS.speak("Good evening, sir.")
    assert TTS.spoken_at() == NEVER, "speak() announces; it does not invite an answer"


def test_a_reply_does_open_the_follow_up_window(device, unspoken):
    speak_stream(iter(["Fifteen degrees, sir."]))
    assert TTS.spoken_at() > NEVER


def test_the_window_opens_even_when_he_is_cut_off(device, unspoken):
    speak_stream(iter(["one.", "two."]), lambda: True)
    assert TTS.spoken_at() > NEVER, "being interrupted still ends his turn"


def test_barge_in_can_be_switched_off(monkeypatch):
    import threading

    from tobias.stt.interrupt import watching

    monkeypatch.setattr(settings, "stt_barge_in", False)
    before = threading.active_count()
    with watching() as interrupted:
        assert interrupted() is False
        assert threading.active_count() == before, "no watcher thread when barge-in is off"


def test_the_watcher_is_cleaned_up_afterwards(monkeypatch):
    import threading

    from tobias.stt.interrupt import watching

    monkeypatch.setattr(settings, "stt_barge_in", True)
    monkeypatch.setattr(sys.modules["tobias.stt.interrupt"], "mic", lambda: __import__("queue").Queue())
    before = threading.active_count()
    with watching() as interrupted:
        assert interrupted() is False
        assert threading.active_count() == before + 1, "the watcher should be running"
    assert threading.active_count() == before, "the watcher must not outlive the reply"
