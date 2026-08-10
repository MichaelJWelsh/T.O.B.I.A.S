import queue
import re
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache

import numpy as np
import torch
from silero_vad import load_silero_vad

from tobias.config import settings
from tobias.stt.transcribe import transcribe
from tobias.stt.vad import FRAME_MS, SAMPLE_RATE, mic
from tobias.stt.wake import addressed

# Consecutive non-speech frames that end a run. One quiet frame mid-word is normal; a third of a
# second of nothing means whatever was said is over.
GAP_FRAMES = 10
# How much new speech to gather before looking again. Every check costs a transcription.
RECHECK_FRAMES = 8
# Consecutive empty reads that mean the microphone has genuinely stopped delivering, rather than
# the ordinary gap between PortAudio's bursts. Twenty reads at 50ms is a second of nothing.
STARVED_FRAMES = 20
# An interruption that is only an order to be quiet carries no question, so answering it would
# be absurd — "Tobias stop" used to get a reply about stopping. Anything else is a redirect and
# becomes the next request. A vocabulary rather than a list of phrases, because speech inflects
# and whisper guesses: "Tobias stop" came back as "Tobias stopped", which no phrase list caught.
# Deliberately not a model call — classifying this would add a round trip to the one moment in
# the system that has to feel instant.
HUSH_WORDS = frozenset(
    "be just please now ok okay no nope not"
    " stop stopped stopping stops halt cease"
    " quiet quieten silence silent shush shh hush"
    " enough thats that s it up shut down"
    " never mind nevermind forget cancel wait hold on"
    " talking speaking sorry".split()
)


@dataclass
class Interruption:
    """Whether he was cut off, and what was said to cut him off.

    Callable so it can be handed straight to speak_stream as the stop check. `said` is only
    final once the watcher has been joined, which watching() does on the way out.
    """

    heard: threading.Event = field(default_factory=threading.Event)
    said: str = ""

    def __call__(self) -> bool:
        return self.heard.is_set()


@cache
def _model():
    # The watcher's own detector, separate from the one segments() uses: silero carries
    # recurrent state, and interleaving two callers through one instance would corrupt it.
    return load_silero_vad()


def _hear_out(frames: queue.Queue, model, stop: threading.Event, so_far: list) -> str:
    """Consume the rest of the interrupting utterance and return the whole request.

    Two jobs at once. Nothing is left queued for listen() to answer — "Tobias stop" used to fire
    on the name and leave "stop" behind as the next question — and the interruption itself
    becomes the next request, so a barge-in that carries one does not have to be repeated.

    End-of-utterance uses VAD_SILENCE_MS, the same rule as ordinary listening. GAP_FRAMES is a
    third of a second, which is shorter than an ordinary pause for breath: "Tobias, what is the
    weather" would be cut to "Tobias, what is" and the rest would arrive as a separate request.
    """
    speech = list(so_far)
    silent = 0
    starved = 0
    hangover = round(settings.vad_silence_ms / FRAME_MS)
    deadline = time.monotonic() + settings.max_utterance_s

    # Deliberately not checking `stop`. It is set the instant playback ends, which is a fraction
    # of a second after the interrupt fired and long before the user has finished their sentence
    # — obeying it here truncated every redirect to the fragment the watcher recognised.
    while silent < hangover and starved < STARVED_FRAMES and time.monotonic() < deadline:
        try:
            frame = frames.get(timeout=0.05)
        except queue.Empty:
            # PortAudio delivers in bursts, so an empty queue is normal between them. Only a
            # long drought means the audio really has stopped coming.
            starved += 1
            continue
        starved = 0
        speech.append(frame)
        speaking = model(torch.from_numpy(frame), SAMPLE_RATE).item() >= settings.vad_threshold
        silent = 0 if speaking else silent + 1

    return _request(addressed(transcribe(np.concatenate(speech))) or "")


def _request(heard: str) -> str:
    """The interruption as a question to answer, or "" if he was only told to be quiet."""
    if not heard:
        return ""
    without_name = re.sub(
        rf"\b{re.escape(settings.stt_wake_word)}\b", " ", heard, flags=re.IGNORECASE
    )
    words = re.findall(r"[a-z]+", without_name.lower())
    return "" if not words or all(word in HUSH_WORDS for word in words) else heard


def _watch(stop: threading.Event, result: Interruption) -> None:
    model = _model()
    model.reset_states()
    frames = mic()
    needed = round(settings.stt_barge_in_ms / FRAME_MS)
    # His name alone is ~16 frames, under `needed`. Without checking short bursts too, "Tobias"
    # followed by any pause would reset the buffer and never be looked at.
    least = max(1, needed // 2)
    cap = round(3000 / FRAME_MS)  # stop re-reading a monologue; he was not being named
    preroll: deque[np.ndarray] = deque(maxlen=max(1, round(settings.vad_speech_pad_ms / FRAME_MS)))
    speech: list[np.ndarray] = []
    silent = 0
    checked = 0

    while not stop.is_set():
        try:
            frame = frames.get(timeout=0.05)
        except queue.Empty:
            continue

        is_speech = model(torch.from_numpy(frame), SAMPLE_RATE).item() >= settings.vad_threshold

        if not speech:
            preroll.append(frame)
            if is_speech:
                speech = list(preroll)
                preroll.clear()
            continue

        # Every frame from here, silence included. Keeping only the VAD-positive ones and gluing
        # them together destroys the envelope: 500ms of "Tobias" transcribes fine, the same audio
        # with its gaps removed transcribes to nothing at all.
        speech.append(frame)
        silent = 0 if is_speech else silent + 1

        ended = silent >= GAP_FRAMES
        due = len(speech) >= needed and len(speech) - checked >= RECHECK_FRAMES
        if not (due or (ended and len(speech) >= least)):
            continue

        # Re-transcribe a window that keeps growing from the moment speech began, rather than a
        # fresh slice each time. Fixed slices cut words in half and whisper guesses — "Tobias"
        # came back as "Dubai" — and clearing the buffer just misaligns the next slice too.
        checked = len(speech)
        if addressed(transcribe(np.concatenate(speech))):
            result.heard.set()  # stop him talking first; hearing the rest out takes a moment
            result.said = _hear_out(frames, model, stop, speech)
            return
        if ended or len(speech) >= cap:
            speech, silent, checked = [], 0, 0


@contextmanager
def watching() -> Iterator[Interruption]:
    """Listen for the wake word while TOBIAS speaks.

    This has to be a thread. Doing the same work between writes on the playing thread starves
    the output device — silero alone costs ~6% of realtime and a transcription is ~250ms — and
    the result is audibly glitchy. Whatever feeds the speaker must do nothing else.
    """
    result = Interruption()
    if not settings.stt_barge_in:
        yield result
        return

    stop = threading.Event()
    watcher = threading.Thread(target=_watch, args=(stop, result), daemon=True)
    watcher.start()
    try:
        yield result
    finally:
        stop.set()
        # Generous, because a watcher that fired is still listening to the end of the sentence
        # that cut him off. It returns as soon as the user stops talking; this only bounds the
        # pathological case. The watch loop itself exits within one read of `stop`.
        watcher.join(timeout=settings.max_utterance_s + 5)
