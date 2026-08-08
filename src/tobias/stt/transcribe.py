from functools import cache

import numpy as np
from faster_whisper import WhisperModel

from tobias.config import settings


@cache
def load() -> WhisperModel:
    model = WhisperModel(
        settings.stt_model,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
    )
    # First inference compiles CUDA kernels and runs ~2x slow; spend that on silence, not on the user.
    list(model.transcribe(np.zeros(16_000, dtype=np.float32), language=settings.stt_language)[0])
    return model


def transcribe(audio: np.ndarray) -> str:
    segments, _ = load().transcribe(
        audio,
        language=settings.stt_language,
        beam_size=settings.stt_beam_size,
        # Each utterance stands alone; carrying context invites hallucination loops.
        condition_on_previous_text=False,
        without_timestamps=True,
    )
    # The filter is the hallucination gate: whisper invents fluent text from near-silence, and
    # no_speech_prob is what tells them apart. Dropping every segment yields "", which listen()
    # skips. Do not quietly delete this — see the Gotchas in CLAUDE.md for the measured numbers.
    return " ".join(
        segment.text.strip()
        for segment in segments
        if segment.no_speech_prob <= settings.stt_max_no_speech
    ).strip()
