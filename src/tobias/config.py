from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env", extra="ignore"
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
    llm_sarcasm: int = 6
    llm_warmth: int = 6
    llm_anxiety: int = 4

    input_device: int | None = None
    output_device: int | None = None


settings = Settings()
