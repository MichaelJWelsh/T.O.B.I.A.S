import numpy as np

from tobias.tts.synthesize import KEEP_MS, SAMPLE_RATE, SILENCE, _trim


def ms(count: int) -> int:
    return int(count / 1000 * SAMPLE_RATE)


def clip(lead_ms: int, tail_ms: int, body: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros(ms(lead_ms)), body, np.zeros(ms(tail_ms))]).astype(np.float32)


def test_padding_is_reduced_to_the_keep_margin():
    body = np.full(ms(500), 0.3, dtype=np.float32)
    trimmed = _trim(clip(380, 530, body))
    loud = np.flatnonzero(np.abs(trimmed) > SILENCE)
    assert loud[0] / SAMPLE_RATE * 1000 == KEEP_MS
    assert (len(trimmed) - loud[-1] - 1) / SAMPLE_RATE * 1000 == KEEP_MS


def test_quiet_consonant_decay_survives():
    # The bug this exists for: a threshold of 0.01 treated the fading tail of a final consonant
    # as silence and audibly cut the last syllable off every sentence. Real decay measures ~0.003.
    body = np.concatenate([np.full(ms(400), 0.3), np.full(ms(120), 0.003)]).astype(np.float32)
    trimmed = _trim(clip(380, 530, body))
    assert np.count_nonzero(np.abs(trimmed) >= 0.003) >= ms(120)


def test_no_speech_is_lost_at_either_edge():
    body = np.full(ms(500), 0.3, dtype=np.float32)
    original = clip(380, 530, body)
    trimmed = _trim(original)
    assert np.count_nonzero(np.abs(trimmed) > SILENCE) == np.count_nonzero(np.abs(original) > SILENCE)


def test_silent_input_is_returned_untouched():
    silent = np.zeros(ms(500), dtype=np.float32)
    assert np.array_equal(_trim(silent), silent)
