import time
from collections.abc import Callable, Iterable

from tobias.tts.playback import play, warm
from tobias.tts.synthesize import synthesize

__all__ = ["speak", "speak_stream", "spoken_at"]

_spoken_at = float("-inf")


def spoken_at() -> float:
    """When he last stopped speaking, as a monotonic timestamp.

    The follow-up window is measured from here rather than from whenever the caller happens to
    ask for the next utterance, so it does not depend on how the conversation loop is shaped.
    """
    return _spoken_at


def _finished() -> None:
    global _spoken_at
    _spoken_at = time.monotonic()


def speak(text: str) -> None:
    """Say text aloud through the speakers, blocking until it has finished."""
    warm()
    play(synthesize(text))
    _finished()


def speak_stream(sentences: Iterable[str], until: Callable[[], bool] | None = None) -> bool:
    """Say each sentence as it arrives. True if `until` cut him off.

    The rest of the reply is abandoned, not queued — being interrupted means he has been asked
    to stop, not to pause.
    """
    warm()
    try:
        for sentence in sentences:
            if play(synthesize(sentence), until):
                return True
        return False
    finally:
        _finished()
