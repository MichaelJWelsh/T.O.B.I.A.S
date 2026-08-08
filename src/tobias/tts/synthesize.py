from functools import cache

import numpy as np

from tobias.config import settings

SAMPLE_RATE = 24_000  # what kokoro emits; not a choice
# Kokoro's padding is true digital zero, so this only has to clear the noise floor. Set it any
# higher and it eats the quiet decay of final consonants — measured down to 0.003 well after the
# last audible vowel.
SILENCE = 0.001
KEEP_MS = 60  # air left either side, so nothing is clipped at the edges


@cache
def load():
    # Deferred because `import kokoro` drags in spacy and transformers and costs ~20s. Importing
    # this module for SAMPLE_RATE or _trim — as the tests do — should be free.
    from kokoro import KPipeline

    # 'b' is British English — it also selects the British G2P, so it must match a bm_/bf_ voice.
    pipeline = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M", device=settings.tts_device)
    # A full sentence, not a word: the first synthesis compiles CUDA kernels and costs ~2.5s,
    # and too short a warm-up fails to trigger it, leaving the bill for the first real reply.
    list(pipeline("Good evening, sir. All systems are online and standing by.", voice=settings.tts_voice))
    return pipeline


def _trim(audio: np.ndarray) -> np.ndarray:
    # Kokoro pads every clip with ~380ms of lead-in and ~530ms of tail. Between streamed
    # sentences that lands as a full second of dead air, and before the first word it is
    # pure latency.
    loud = np.flatnonzero(np.abs(audio) > SILENCE)
    if not len(loud):
        return audio
    keep = int(KEEP_MS / 1000 * SAMPLE_RATE)
    return audio[max(0, loud[0] - keep) : loud[-1] + 1 + keep]


def synthesize(text: str) -> np.ndarray:
    results = load()(text, voice=settings.tts_voice, speed=settings.tts_speed)
    return _trim(np.concatenate([np.asarray(result.audio, dtype=np.float32) for result in results]))
