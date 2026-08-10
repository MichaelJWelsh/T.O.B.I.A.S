import json
from pathlib import Path
from typing import Any

from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

ROOT = Path(__file__).resolve().parents[2]
# What TOBIAS has changed about himself. Written by update(), gitignored, and absent until the
# first change — the JSON source treats a missing file as no settings at all.
STATE_FILE = ROOT / "state.json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        json_file=STATE_FILE,
        extra="ignore",
        # So update() rejects a bad value loudly instead of writing it to disk.
        validate_assignment=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Highest priority first. state.json outranks .env because a dial TOBIAS changed must
        # survive a restart, but a real environment variable still wins so a one-off override
        # works. The consequence to remember: once he changes a dial, editing .env stops moving
        # it — .env is the seed, state.json is the truth.
        return (
            init_settings,
            env_settings,
            JsonConfigSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )

    stt_model: str = "distil-large-v3"
    stt_device: str = "cuda"
    stt_compute_type: str = "float16"
    stt_language: str = "en"
    stt_beam_size: int = 1
    # Whisper invents confident text from noise. Measured across clean, noisy, quiet, clipped and
    # one-word speech, every real utterance scored under 0.04 while noise and silence scored over
    # 0.11 — far wider separation than avg_logprob, which a short "yes" can fail on its own.
    stt_max_no_speech: float = 0.1

    # He only answers an utterance that names him. Empty disables the gate entirely, which is
    # what you want for test_stt.py or a headset where you are the only speaker.
    stt_wake_word: str = "Tobias"
    # Whether naming him while he is talking cuts him off. Costs a transcription every ~600ms of
    # speech heard during playback, and on speakers he will interrupt himself — see CLAUDE.md.
    stt_barge_in: bool = True
    # Speech gathered before the watcher checks it for the wake word. 500ms is enough for whisper
    # to return "Tobias" (measured); below ~400ms it returns a fragment and the no-speech gate
    # discards it. Raising this delays every interrupt by the same amount.
    stt_barge_in_ms: int = 600
    # How long after he finishes replying you can answer back without naming him again. The
    # window opens when he stops talking, not when you did. 0 always requires the wake word.
    stt_follow_up_s: float = 20.0

    vad_threshold: float = 0.5
    # How long I can pause mid-sentence before I get cut off
    vad_silence_ms: int = 700 
    vad_min_speech_ms: int = 250
    # Silero needs a moment of speech before it fires, and whatever it took is already gone. This
    # is free — the audio is buffered either way, and whisper pads to 30s regardless of length.
    vad_speech_pad_ms: int = 400
    max_utterance_s: float = 30.0

    tts_voice: str = "bm_george"
    tts_speed: float = 1.0
    # Kokoro is a torch model, so this needs the CUDA torch build pinned in requirements.txt.
    tts_device: str = "cuda"

    # Empty by default so the STT and TTS scripts still run without a key.
    openai_api_key: str = ""
    # The gpt-5 family locks temperature/top_p/penalties and adds reasoning latency, so a
    # non-reasoning model is the right choice here — the sliders below need those knobs.
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = 1.0
    llm_top_p: float = 1.0
    llm_presence_penalty: float = 0.6
    llm_frequency_penalty: float = 0.3
    # None lets the model use its own maximum. Set a number only to cap a runaway reply.
    llm_max_tokens: int | None = None

    # Personality, 0-10, one per register: comic, social, emotional. Rendered into the system
    # prompt by llm/prompts/. Everything else about how he talks is fixed character, not a dial.
    llm_sarcasm: int = 8
    llm_warmth: int = 10
    llm_anxiety: int = 6

    input_device: int | None = None
    output_device: int | None = None


settings = Settings()


def update(**changes: Any) -> None:
    """Change settings now and keep them, so a restart does not undo them.

    Both halves matter: `settings` is a singleton read fresh on every prompt render, so mutating
    it takes effect immediately, and writing state.json is what survives the process. Only the
    named fields are persisted — never the whole object, which holds the API key.
    """
    for field, value in changes.items():
        setattr(settings, field, value)

    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    STATE_FILE.write_text(json.dumps(state | changes, indent=2) + "\n")
