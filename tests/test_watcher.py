"""The watcher's decision logic, with silero and whisper stubbed.

Both cases here are regressions. Real audio is needed to catch them for real, which is why the
comments say what the live symptom was.
"""

import queue
import sys
import threading

import numpy as np
import pytest

import tobias.stt.interrupt as interrupt
from tobias.config import settings
from tobias.stt.vad import FRAME_SIZE

LOUD = np.full(FRAME_SIZE, 0.5, dtype=np.float32)
QUIET = np.zeros(FRAME_SIZE, dtype=np.float32)


class FakeVad:
    """Speech is whatever is not silent, so a test can spell out the pattern frame by frame."""

    def __call__(self, tensor, rate):
        loud = float(np.abs(tensor.numpy()).max()) > 0.1
        return type("P", (), {"item": lambda self: 1.0 if loud else 0.0})()

    def reset_states(self):
        pass


@pytest.fixture
def watcher(monkeypatch):
    monkeypatch.setattr(settings, "stt_barge_in", True)
    monkeypatch.setattr(settings, "stt_wake_word", "Tobias")
    monkeypatch.setattr(interrupt, "_model", FakeVad)

    def run(frames, transcripts):
        said = iter(transcripts)
        seen = []

        def transcribe(buffer):
            seen.append(len(buffer) / FRAME_SIZE)
            return next(said, "")

        monkeypatch.setattr(interrupt, "transcribe", transcribe)
        fake: queue.Queue = queue.Queue()
        for frame in frames:
            fake.put(frame)
        monkeypatch.setattr(interrupt, "mic", lambda: fake)

        stop, result = threading.Event(), interrupt.Interruption()
        thread = threading.Thread(target=interrupt._watch, args=(stop, result), daemon=True)
        thread.start()
        thread.join(timeout=5)
        stop.set()
        return result, seen

    return run


def test_the_wake_word_stops_him(watcher):
    result, _ = watcher([LOUD] * 40, ["Tobias, stop"])
    assert result()


def test_speech_that_does_not_name_him_is_ignored(watcher):
    result, _ = watcher([LOUD] * 40 + [QUIET] * 15, ["the weather is fine", "still nothing"])
    assert not result()


def test_the_window_grows_instead_of_resetting(watcher):
    # Live symptom: fixed slices cut words in half and whisper guessed — "Tobias" came back as
    # "Dubai" — and clearing the buffer misaligned every slice after it, so it never recovered.
    result, sizes = watcher([LOUD] * 60, ["Dubai", "Dubai what", "Tobias, what is"])
    assert result(), "a later, longer look must still be able to hear the name"
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0], f"window should grow, got {sizes}"


def test_a_short_burst_is_checked_when_it_ends(watcher):
    # Live symptom: "Tobias" alone is ~16 frames, under the 19 needed, so a pause after his name
    # discarded the buffer unheard and the interrupt never fired.
    result, _ = watcher([LOUD] * 16 + [QUIET] * 15, ["Tobias"])
    assert result(), "his name on its own, then a pause, must still count"


def test_the_interruption_becomes_the_next_request(watcher):
    # Otherwise a barge-in carrying a question has to be said twice: once to stop him, once to
    # ask. The last transcript is of the whole utterance, heard out after he was cut off.
    result, _ = watcher([LOUD] * 40 + [QUIET] * 15, ["Tobias, what", "Tobias, what is the time?"])
    assert result()
    assert result.said == "Tobias, what is the time?"


def test_a_pause_for_breath_does_not_truncate_the_request(monkeypatch):
    # Live symptom: hearing out stopped after GAP_FRAMES, a third of a second, so "Tobias, what
    # is the weather" was cut to "Tobias, what is" and "the weather" arrived as a second request.
    monkeypatch.setattr(settings, "stt_barge_in", True)
    monkeypatch.setattr(settings, "stt_wake_word", "Tobias")
    monkeypatch.setattr(settings, "vad_silence_ms", 700)
    monkeypatch.setattr(interrupt, "_model", FakeVad)
    monkeypatch.setattr(interrupt, "transcribe", lambda buffer: "Tobias, what is the weather")

    # A breath in the middle, shorter than VAD_SILENCE_MS but longer than GAP_FRAMES.
    frames = [LOUD] * 25 + [QUIET] * 15 + [LOUD] * 20 + [QUIET] * 30
    fake: queue.Queue = queue.Queue()
    for frame in frames:
        fake.put(frame)
    monkeypatch.setattr(interrupt, "mic", lambda: fake)

    stop, result = threading.Event(), interrupt.Interruption()
    thread = threading.Thread(target=interrupt._watch, args=(stop, result), daemon=True)
    thread.start()
    thread.join(timeout=5)
    stop.set()

    assert result()
    left = [fake.get() for _ in range(fake.qsize())]
    assert not any(np.abs(f).max() > 0.1 for f in left), (
        "the speech after the breath must be part of the same request, not left for listen()"
    )


def test_hearing_out_survives_the_stop_signal(monkeypatch):
    # Live symptom, and the reason redirects kept arriving in pieces. `stop` is set the instant
    # playback ends — a fraction of a second after the interrupt fired, while the user is still
    # mid-sentence. Obeying it here truncated "Tobias, what is the weather" to "Tobias, what"
    # and left "the weather" for listen() to answer as a separate request.
    monkeypatch.setattr(settings, "stt_wake_word", "Tobias")
    monkeypatch.setattr(settings, "vad_silence_ms", 700)
    monkeypatch.setattr(interrupt, "transcribe", lambda buffer: "Tobias, what is the weather")

    fake: queue.Queue = queue.Queue()
    for frame in [LOUD] * 30 + [QUIET] * 25:
        fake.put(frame)

    already_stopped = threading.Event()
    already_stopped.set()
    said = interrupt._hear_out(fake, FakeVad(), already_stopped, [LOUD] * 5)

    assert said == "Tobias, what is the weather"
    left = [fake.get() for _ in range(fake.qsize())]
    assert not any(np.abs(f).max() > 0.1 for f in left), "the whole sentence must be consumed"


def test_being_told_only_to_stop_carries_nothing_forward(watcher):
    result, _ = watcher([LOUD] * 40 + [QUIET] * 15, ["Tobias, stop", ""])
    assert result()
    assert result.said == "", "an empty request must not be sent to the LLM"


def test_the_rest_of_the_interruption_is_swallowed(watcher):
    # Live symptom: "Tobias stop" fired on "Tobias" and left "stop" queued, which listen() then
    # accepted as a new question because the follow-up window had just opened.
    frames = [LOUD] * 40 + [QUIET] * 15
    fake: queue.Queue = queue.Queue()
    for frame in frames:
        fake.put(frame)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(settings, "stt_barge_in", True)
        m.setattr(settings, "stt_wake_word", "Tobias")
        m.setattr(interrupt, "_model", FakeVad)
        m.setattr(interrupt, "transcribe", lambda buffer: "Tobias, stop")
        m.setattr(interrupt, "mic", lambda: fake)

        stop, heard = threading.Event(), interrupt.Interruption()
        thread = threading.Thread(target=interrupt._watch, args=(stop, heard), daemon=True)
        thread.start()
        thread.join(timeout=5)
        stop.set()

    assert heard()
    left = [fake.get() for _ in range(fake.qsize())]
    assert not any(np.abs(f).max() > 0.1 for f in left), (
        "every speech frame of the interruption must be consumed, or listen() answers it; "
        "trailing silence is fine, _swallow stops once the utterance has ended"
    )
