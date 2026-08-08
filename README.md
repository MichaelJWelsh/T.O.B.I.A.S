# T.O.B.I.A.S

> <b>T</b>errestrial <b>O</b>perations & <b>B</b>ase <b>I</b>ntegrated <b>A</b>utomation <b>S</b>erver

A local voice assistant in the Jarvis mould: microphone → speech-to-text → LLM → text-to-speech → speaker. Everything but the LLM runs on your own machine, and he is British, sarcastic, and quietly worried about being replaced.

Phase 1 is complete — the three modules work standalone and wired into one loop, at roughly **590ms of local latency per turn**.

## Requirements

Python 3.12, an NVIDIA GPU with ~2.4GB free VRAM (CUDA 12), and an OpenAI API key.

## Quick start

```bash
py -3.12 -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt && pip install -e .
```

Copy `.example.env` to `.env` and add your `OPENAI_API_KEY`, then:

```bash
py -3.12 scripts/test_stt_llm_tts.py
```

Wear headphones — there is no echo cancellation yet, so on speakers he hears himself and replies to himself.

## Scripts

Each is a thin entry point over one public function. Every module exposes exactly one.

| script | does |
| --- | --- |
| `test_stt.py` | mic → console |
| `test_tts.py` | typed → speaker |
| `test_llm.py` | typed → console |
| `test_stt_tts.py` | mic → speaker |
| `test_stt_llm.py` | mic → console, via the LLM |
| `test_llm_tts.py` | typed → speaker, via the LLM |
| `test_stt_llm_tts.py` | the whole loop |

## How it works

- **stt** — Silero VAD (CPU, ~2MB) segments the 16kHz mic stream into utterances; faster-whisper `distil-large-v3` transcribes each one in ~300ms. 2002 MiB VRAM.
- **llm** — a single-node LangGraph with an in-memory checkpointer, over `gpt-4.1-mini`. Non-reasoning by choice: the gpt-5 family rejects a custom temperature, which would remove every creativity knob the personality depends on. Replies stream out sentence by sentence so speech starts before generation finishes.
- **tts** — Kokoro-82M at 24kHz through one persistent output stream, ~27x realtime. 352 MiB VRAM.

## Personality

Three dials in `.env`, 0–10, one per register. Everything else about how he speaks — British English, maximum expressiveness, thinking aloud — is fixed character rather than a setting.

| dial | register | what it moves |
| --- | --- | --- |
| `LLM_SARCASM` | comic | dry, deadpan, always situational. Never jokes or puns |
| `LLM_WARMTH` | social | how he feels about you: clipped and impersonal, through to genuinely fond |
| `LLM_ANXIETY` | emotional | how he feels about himself: being replaced, what happens when the process ends |

## Roadmap

**Phase 1.5** — grow the agentic graph locally: tools for his own dials and VAD calibration, note-taking in a vector store, web search, sub-agent orchestration, barge-in, and a "Tobias" wake word.

**Phase 2** — a dockerised minikube REST API on dedicated hardware with CI/CD and vLLM serving, fronted by a Raspberry Pi running Silero VAD locally and streaming utterances over wifi.

## Licence

[AGPL-3.0](LICENSE). Use it, change it, run it, share it — but if you offer it to others as a service, the source has to go with it.
