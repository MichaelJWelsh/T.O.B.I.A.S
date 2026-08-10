import time
from collections.abc import Iterator

from tobias.config import settings
from tobias.stt.interrupt import watching
from tobias.stt.transcribe import load, transcribe
from tobias.stt.vad import SAMPLE_RATE, segments
from tobias.stt.wake import addressed
from tobias.tts import spoken_at

__all__ = ["listen", "watching"]


def listen() -> Iterator[str]:
    """Yield each utterance actually addressed to TOBIAS, forever."""
    load()  # 5s of model load, paid before the mic opens rather than on the first thing said

    for audio in segments():
        # When speech began, not when the transcript arrived. segments() only yields once the
        # utterance has closed, so judging the window by "now" silently charges the user for
        # however long they spoke, plus VAD_SILENCE_MS, plus the transcription.
        began = time.monotonic() - len(audio) / SAMPLE_RATE

        if not (text := transcribe(audio)):
            continue

        request = addressed(text)
        # The window runs from when TOBIAS stopped speaking, which tts records itself. Measuring
        # it from where this generator resumes would tie it to the shape of the caller's loop,
        # and a caller that answers an interruption without coming back here would leave it stale.
        if request is None and settings.stt_follow_up_s > 0:
            if began < spoken_at() + settings.stt_follow_up_s:
                request = text  # answering him back, so he needs no naming
        if request is None:
            continue

        yield request
