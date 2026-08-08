from collections.abc import Iterable

from tobias.tts.playback import play, warm
from tobias.tts.synthesize import synthesize

__all__ = ["speak", "speak_stream"]


def speak(text: str) -> None:
    """Say text aloud through the speakers, blocking until it has finished."""
    warm()
    play(synthesize(text))


def speak_stream(sentences: Iterable[str]) -> None:
    """Say each sentence as it arrives."""
    warm()
    for sentence in sentences:
        play(synthesize(sentence))
