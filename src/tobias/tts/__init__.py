import time
from collections.abc import Callable, Iterable

from tobias.tts.playback import play, warm
from tobias.tts.synthesize import synthesize

__all__ = ["speak", "speak_stream", "spoken_at"]

_spoken_at = float("-inf")


def spoken_at() -> float:
    """When he last finished *replying*, as a monotonic timestamp.

    The follow-up window is measured from here rather than from whenever the caller happens to
    ask for the next utterance, so it does not depend on how the conversation loop is shaped.

    Only speak_stream() moves it. An announcement is not a reply and must not open the window:
    the startup greeting otherwise let the first thing the user said skip the wake word, and
    since every reply reopens the window, a whole session could run without ever naming him.
    """
    return _spoken_at


def speak(text: str) -> None:
    """Announce something aloud, blocking until it has finished. Not part of a conversation."""
    warm()
    play(synthesize(text))


def speak_stream(sentences: Iterable[str], until: Callable[[], bool] | None = None) -> bool:
    """Say each sentence as it arrives. True if `until` cut him off.

    The rest of the reply is abandoned, not queued — being interrupted means he has been asked
    to stop, not to pause.
    """
    global _spoken_at
    warm()
    try:
        for sentence in sentences:
            if play(synthesize(sentence), until):
                return True
        return False
    finally:
        _spoken_at = time.monotonic()  # he has replied, so an answer back is now expected
