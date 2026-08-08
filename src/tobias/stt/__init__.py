from collections.abc import Iterator

from tobias.stt.transcribe import load, transcribe
from tobias.stt.vad import segments

__all__ = ["listen"]


def listen() -> Iterator[str]:
    """Yield a transcript for each utterance spoken into the microphone, forever."""
    load()  # 5s of model load, paid before the mic opens rather than on the first thing said
    for audio in segments():
        if text := transcribe(audio):
            yield text
