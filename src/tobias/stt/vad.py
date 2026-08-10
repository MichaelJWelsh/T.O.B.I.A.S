import queue
from collections import deque
from collections.abc import Iterator
from functools import cache

import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad

from tobias.config import settings

SAMPLE_RATE = 16_000
FRAME_SIZE = 512  # silero v5 accepts nothing else at 16 kHz
FRAME_MS = FRAME_SIZE / SAMPLE_RATE * 1000


# The live input stream. It MUST stay referenced for as long as it is running: PortAudio keeps
# calling the callback regardless, so letting Python collect the object is a use-after-free that
# crashes the process at random and can take the display driver down with it.
_open: sd.InputStream | None = None


@cache
def mic() -> queue.Queue[np.ndarray]:
    """The microphone, opened once and left open, feeding one shared queue.

    Two things read it: segments(), and the barge-in poll during playback. They never overlap —
    listen() is suspended at its yield the whole time TOBIAS is speaking — so whatever the poll
    consumes is gone before segments() resumes, which is exactly what keeps his own voice from
    reaching the transcriber and stops the queue growing while nobody is draining it.
    """
    global _open
    frames: queue.Queue[np.ndarray] = queue.Queue()
    _open = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        channels=1,
        dtype="float32",
        device=settings.input_device,
        callback=lambda data, *_: frames.put(data[:, 0].copy()),
    )
    _open.start()
    return frames


def _frames() -> Iterator[np.ndarray]:
    frames = mic()
    while True:
        yield frames.get()


def segments() -> Iterator[np.ndarray]:
    """Yield the mono audio of each utterance spoken into the microphone."""
    model = load_silero_vad()
    preroll_frames = max(1, round(settings.vad_speech_pad_ms / FRAME_MS))
    hangover_frames = round(settings.vad_silence_ms / FRAME_MS)
    min_frames = round(settings.vad_min_speech_ms / FRAME_MS)
    max_frames = round(settings.max_utterance_s * 1000 / FRAME_MS)

    preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)
    speech: list[np.ndarray] = []
    silent = 0

    for frame in _frames():
        is_speech = model(torch.from_numpy(frame), SAMPLE_RATE).item() >= settings.vad_threshold

        if not speech:
            preroll.append(frame)
            if is_speech:
                speech = list(preroll)
                preroll.clear()
            continue

        speech.append(frame)
        silent = 0 if is_speech else silent + 1

        if silent >= hangover_frames or len(speech) >= max_frames:
            if len(speech) - silent >= min_frames:
                yield np.concatenate(speech)
            model.reset_states()
            speech, silent = [], 0
