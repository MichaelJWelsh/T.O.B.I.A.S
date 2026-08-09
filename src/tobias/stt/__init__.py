import time
from collections.abc import Iterator

from tobias.config import settings
from tobias.stt.transcribe import load, transcribe
from tobias.stt.vad import segments
from tobias.stt.wake import addressed

__all__ = ["listen"]


def listen() -> Iterator[str]:
    """Yield each utterance actually addressed to TOBIAS, forever."""
    load()  # 5s of model load, paid before the mic opens rather than on the first thing said
    follow_up_until = 0.0

    for audio in segments():
        if not (text := transcribe(audio)):
            continue

        request = addressed(text)
        if request is None and time.monotonic() < follow_up_until:
            request = text  # inside the follow-up window, he does not need naming again
        if request is None:
            continue

        yield request
        # Execution resumes here only once the caller has finished replying, so the window
        # opens when TOBIAS stops talking rather than when the user did.
        follow_up_until = time.monotonic() + settings.stt_follow_up_s
