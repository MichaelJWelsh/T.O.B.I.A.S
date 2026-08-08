import atexit
from functools import cache

import numpy as np
import sounddevice as sd

from tobias.config import settings
from tobias.tts.synthesize import SAMPLE_RATE

# A cold output device swallows the opening of the first stream it is given. This stream then
# stays open for the life of the process, so that is paid once instead of on every reply.
WAKE_MS = 400


@cache
def _stream() -> sd.OutputStream:
    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=settings.output_device,
    )
    stream.start()
    stream.write(np.zeros(int(WAKE_MS / 1000 * SAMPLE_RATE), dtype=np.float32))
    atexit.register(_drain, stream)
    return stream


def _drain(stream: sd.OutputStream) -> None:
    # write() returns while the device is still draining, so a process that exits straight after
    # speaking would cut off its own last words. Push silence through until they are out.
    stream.write(np.zeros(int((stream.latency + 0.1) * SAMPLE_RATE), dtype=np.float32))
    stream.stop()


def warm() -> None:
    """Open the output device before there is anything to say.

    A cold device is slow to start and loses whatever plays while it does — 400ms of silence was
    not enough to cover it. Opening here means the device spends the model load warming up
    instead of eating the first syllable.
    """
    _stream()


def play(audio: np.ndarray) -> None:
    """Play, blocking until the audio has been handed to the device."""
    _stream().write(audio)
