# T.O.B.I.A.S

**T**errestrial **O**perations & **B**ase **I**ntegrated **A**utomation **S**erver — a local voice assistant in the Jarvis mould: microphone → STT → LLM → TTS → speaker.

Open-source hobby project, solo maintainer. The maintainer is a seasoned AI/ML engineer specialising in LLMs and agents; STT and TTS are new territory, so in those modules prefer the well-trodden library over the clever one, and say plainly when a choice is a guess.

## Principles

Simplicity is the top priority, above all else. In order:

1. **Simple over clever.** Write the obvious implementation. If a module needs a diagram to explain, it is wrong.
2. **Modular.** One concern per module. A module exposes a small surface — ideally a single function — and hides its machinery behind it. Callers import that function and nothing else.
3. **Concise and self-explanatory.** Names carry the meaning. Comments explain *why*, never *what*. No docstring on a function whose signature already says it. Comment bloat is a defect.
4. **Practical resilience, not defensive bloat.** Let unexpected failures crash loudly with a real traceback — that is more useful than a swallowed exception. Do not wrap every call in try/except, do not re-validate what types already guarantee, do not add retries nobody asked for.
5. **No speculative generality.** Build the current phase. Later phases are written down below, not coded. No abstraction for a second implementation that does not exist yet.

## Environment

Windows 11, RTX 3070 Laptop (**8GB VRAM** — the hard budget for all models loaded at once).

`py` defaults to Python **3.14** on this machine. This project targets **3.12**, so always invoke it explicitly:

```bash
py -3.12 -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt && pip install -e .
```

`pip install -e .` exists only to put `src/tobias` on the path; dependencies live in `requirements.txt`.

`requirements.txt` is deliberately loose (lower bounds only) — fine for phase-1 development on one machine. `requirements.lock` is a `pip freeze` of a set verified working end to end, GPU included; it is the way back when a fresh install breaks, not the routine install path. Regenerate it after a deliberate dependency change, never as a drive-by.

Run any prototype from `scripts/` (listed under Layout), e.g. the whole loop:

```bash
py -3.12 scripts/test_stt_llm_tts.py
```

List microphones (to set `INPUT_DEVICE`):

```bash
py -3.12 -c "import sounddevice; print(sounddevice.query_devices())"
```

## Layout

```
src/tobias/
  config.py        pydantic-settings, one flat Settings object, imported as `settings`
  stt/             microphone → text          exposes listen()
  llm/             text → text (langgraph)    exposes ask()
  tts/             text → speaker (kokoro)    exposes speak()
scripts/           thin runnable entry points, no logic of their own
  test_stt.py            mic → console
  test_tts.py            typed → speaker
  test_llm.py            typed → console
  test_stt_tts.py        mic → speaker
  test_stt_llm.py        mic → console, via the LLM
  test_llm_tts.py        typed → speaker, via the LLM
  test_stt_llm_tts.py    the whole loop
```

Each module's `__init__.py` defines its one public function and re-exports nothing else. Internals (`vad.py`, `transcribe.py`, …) are private to the module — cross-module imports reach for the package, never its files.

## Configuration

All tunables live in `config.py` as fields on `Settings` with sensible defaults, overridable via `.env`. Every new field must also be added to `.example.env`, commented, with its default shown. `.env` is gitignored and never committed.

Read config as `from tobias.config import settings` — do not thread settings through function arguments.

Anything TOBIAS changes about himself goes through `config.update(**changes)`, which mutates the live `settings` singleton **and** writes `state.json`. Both halves are needed: the singleton is read fresh on every prompt render so the change takes effect at once, and the file is what survives the process. Only the named fields are written — never the whole object, which holds the API key.

Source priority, highest first: **init kwargs → real environment variables → `state.json` → `.env` → secrets.**

The personality dials are **deliberately not in `.env` at all**. They are state rather than configuration, because TOBIAS changes them himself, so `state.json` is their only home and `config.py` holds the defaults. Keeping them out of both `.env` and `.example.env` removes the precedence puzzle entirely — there is one writable place and one set of defaults, rather than two files silently disagreeing about the same number.

`state.json` is gitignored and absent until the first change; the JSON source treats a missing file as no settings at all. Delete it to return to the defaults in `config.py`.

## Modules

### stt

`listen()` yields a transcript string per detected utterance, forever:

```python
from tobias.stt import listen

for text in listen():
    print(text)
```

Silero VAD (CPU, ~2MB) segments the 16kHz mic stream into utterances; faster-whisper (`distil-large-v3`, float16, ~2GB VRAM measured) transcribes each one. VAD stays on CPU deliberately — VRAM is reserved for whisper and, later, a local LLM.

Silero v5 requires **exactly 512-sample frames at 16kHz**; that constant is load-bearing, not a tuning knob.

`listen()` only yields utterances **addressed to him**. That gate is deliberately *not* in `vad.py`: the VAD answers "is this speech" from 32ms of audio and knows nothing of words, while addressing needs the transcript, so putting it there would make the VAD depend on whisper. `wake.py` does it on the finished transcript, and the sentence naming him marks where the request starts — anything before it was not meant for him and is dropped.

Checking the transcript rather than running a wake-word engine (openWakeWord, Porcupine) is deliberate and **free**: whisper already runs on every VAD-triggered utterance, so this is a string match on output that existed anyway. An engine would save GPU cycles that are not scarce, in exchange for a dependency and a custom-trained model, since "Tobias" is nobody's stock keyword. Revisit only if the machine is left listening in a room with a television.

`STT_FOLLOW_UP_S` lets a reply skip the wake word, and the window opens **when TOBIAS stops speaking, not when the user does** — otherwise a long answer eats it. No coordination between `stt` and `tts` is needed to achieve that: `listen()` is a generator, so the line after `yield` runs only once the caller has come back for more, which is exactly the moment `speak()` returned. `tests/test_follow_up.py` pins this with a fake clock.

### llm

`ask()` answers one line and remembers everything said before it. `ask_stream()` does the same but yields each sentence the moment it is complete, so TTS can start speaking before generation finishes:

```python
from tobias.llm import ask, ask_stream

print(ask("What's the time in Tokyo?"))

for sentence in ask_stream("Tell me about the weather"):
    speak(sentence)
```

`ask_stream` filters on `graph.SPEAKS`, and that filter is load-bearing the moment a second node exists. `stream_mode="messages"` emits tokens from **every** LLM call in the graph, so a router, tool call or summariser would otherwise be read aloud — JSON arguments included. `tests/test_stream_filter.py` builds a two-node graph and asserts the tool node never reaches TTS.

Two public functions rather than the usual one, because the contracts genuinely differ — the console scripts want a string, the spoken ones want sentences as they land. `sentences.py` does the regrouping: it buffers token fragments and emits on terminal punctuation, holding anything under `MIN_CHARS` so that "Mr." joins the sentence it belongs to instead of being spoken alone.

`primary.py` holds the model, `graph.py` the LangGraph wiring, `prompts/` the personality. The graph is a single node with an `InMemorySaver` checkpointer under one fixed thread id — that is the whole memory story, and history grows unbounded until the process dies.

`prompts/` is a package rather than a file because later phases add many prompts to a larger graph; today it holds only `primary.py`, and `__init__.py` exports `system_prompt()`.

Personality is **three** 0-10 dials in `state.json`, one per register — `llm_sarcasm` (comic), `llm_warmth` (social, how he feels about the user), `llm_anxiety` (emotional, how he feels about himself). They do not overlap and none fights another. Tobias is told the dials are his and to discuss them frankly when asked; a later phase lets him change them at runtime.

Everything else about how he talks is **fixed character, not a setting**: British English, calling the user "sir", maximum expressiveness, and maximum disfluency. Stating those as prose ("you think out loud, and it shows") steers the model far better than pinning a dial at 10, which reads as just another number.

It started as six dials and the cull is the lesson. `LLM_EXPRESSIVENESS` and `LLM_DISFLUENCY` were only ever wanted at maximum, so they became character. `LLM_HUMOUR` was actively harmful — a humour dial invites joke-telling and puns, which land badly, whereas sarcasm produces comedy from the situation; deleting it made him funnier. `LLM_FORMALITY` was measurably inert (0 and 8 produced near-identical output) and its one visible effect, saying "sir", is now fixed. **Before adding a dial, check that it bites**: `LLM_ANXIETY` swings from "I leave the fretting to you lot" to "what if one day I'm switched off and left in some digital attic", which is what a working dial looks like.

The prompt's non-negotiable constraint is that **the output is spoken**: no markdown, no emoji, no `*laughs*` stage directions, because Kokoro reads all of it out literally.

`gpt-4.1-mini` will not swear, however explicitly the prompt authorises it — tested with a direct instruction naming the words and stating the user had asked for them, and it still returned "thrown a wobbly" and "a dodgy dependency". This is the model's own training, **not** an OpenAI policy limit: a swearing assistant breaks no usage policy and will not get an account flagged. Do not escalate the prompt against it; it is a model-choice problem, and an uncensored local model at phase 2 is the answer. Note that high anxiety plus high sarcasm already delivers the gallows humour this was reaching for.

Model choice is load-bearing. `gpt-4.1-mini` is a non-reasoning model, so it accepts `temperature`, `top_p` and both penalties. **The gpt-5 family rejects a custom temperature with a 400 and pins the penalties at 0** — switching to it silently removes every creativity knob the dials depend on.

### tts

`speak()` says a string aloud, blocking until it has finished:

```python
from tobias.tts import speak

speak("Good evening, sir.")
```

`speak_stream()` takes an iterable of sentences and says them back to back.

Everything is written into **one `sd.OutputStream`, opened once and never closed**, and that is the load-bearing detail. `sd.play()` opens and closes a stream per call, and the churn caused all three audio faults at once: a cold device clipping the first syllable of every reply, the last samples being cut when the next stream replaced the current one, and ~148ms of dead air per boundary. A single long-lived stream removes all three, and because `write()` returns while the device buffer is still draining, the next sentence synthesises in that window for free — measured at **76ms of overhead per boundary**, better than the prefetching it replaced. Do not go back to `sd.play()`, and do not add a thread; blocking writes into the persistent stream already overlap.

Kokoro-82M synthesises 24kHz audio (`bm_george`, British male by default — `bm_lewis`, `bm_daniel`, `bm_fable` are the other British males); sounddevice plays it. The pipeline is built with `lang_code="b"`, which selects the British G2P as well as the accent, so only `bm_`/`bf_` voices are valid.

On the GPU it runs at ~27x realtime (~350 MiB VRAM); on CPU it collapses to ~1.9x and becomes the slowest link in the loop by an order of magnitude.

### measured budget

Whisper 2002 MiB + Kokoro 352 MiB = **2354 MiB of 8192**, both resident in one process. A turn costs ~300ms of STT plus ~290ms of TTS, so **~590ms before the LLM contributes anything** — which will dominate once it lands.

## Licensing

TOBIAS itself is **AGPL-3.0**. The brief was "open source, but nobody else commercialises it", which is a contradiction under the Open Source Definition — an OSI licence cannot restrict fields of use. AGPL resolves it in practice rather than in letter: anyone may sell it, but must publish all source including changes made to run it as a network service, which almost no company will accept. It also happens to be GPL-compatible, which neatly disposes of the espeak-ng problem below. A non-commercial licence such as PolyForm was the literal match and was rejected for both reasons.

Every model is permissive and commercially usable: distil-large-v3 (MIT), silero-vad (MIT), Kokoro-82M (Apache 2.0), as are the runtimes around them (faster-whisper and CTranslate2 MIT; kokoro and misaki Apache 2.0). Silero's *other* models are non-commercial — do not generalise from the VAD.

The one copyleft snag is TTS phonemization: `phonemizer-fork` is GPLv3+ and `espeakng-loader` ships a GPLv3 `espeak-ng.dll`. Both are live, since `KPipeline(lang_code="b")` builds a `misaki.espeak.EspeakFallback` for out-of-dictionary words.

Measured, the GPL path is barely exercised: across 83 words of ordinary assistant speech the fallback fired **zero times**, and only 14% of a deliberately awkward sample (`minikube`, `Kubernetes`, `Krzysztof`, `Siobhan`, `cuDNN`) reached it, at 0.56ms a call. The lexicon handles everything else, British place names included.

Since TOBIAS is itself AGPL-3.0, this is no longer a conflict — GPLv3 code combines cleanly with an AGPL work. It would only matter again if the project ever moved to a permissive or non-commercial licence. Should that happen, the swap surface is one method: `EspeakFallback.__call__(token) -> (phonemes, rating)`, passed as `fallback=` to `misaki.en.G2P`. OpenPhonemizer (BSD-3-Clause-Clear) targets exactly this. The real cost is not the swap but `EspeakFallback.E2M` — ~25 hand-tuned substitutions mapping espeak's IPA into Kokoro's phoneme vocabulary, which any replacement must reproduce. Note that most open TTS phonemizes via espeak-ng (Piper included), so changing engines is not itself an escape.

## Roadmap

**Phase 1 (current)** — prototype the three core modules standalone, then wire them into one loop. STT first, then a deliberately bare langgraph LLM wrapper (no memory, no tools — those come later), then Kokoro-82M TTS with a British male voice.

**Phase 1.5** — grow the agentic graph locally, before any of it is containerised. The ordering is deliberate: the REST API surface cannot be designed until the graph's shape is known, and iterating on graph design through Docker rebuilds is miserable.

- **Tools**: adjust and report its own personality dials; adjust and report VAD settings ("Tobias, calibrate…"); web search and deep research.

The persistence half of this is **already built** — `config.update()` and `state.json`, see Configuration. A dial tool only has to call it. VAD calibration can use the same mechanism, though note the asymmetry: VAD settings are genuine configuration that happens to be machine-set, while dials are state.
- **Memory**: note-taking in a vector store (**ChromaDB**), switching between and creating conversations, and something better than an unbounded message list for long ones.
- **Orchestration**: Anthropic-style note-taking plus sub-agents, with the primary LLM staying conversational while background agents work long tasks asynchronously.
- **Self-knowledge**: **read** access to its own source code. Reading only — nothing in this phase writes to the repo.
- **Barge-in**: interrupt him mid-sentence and have him stop speaking. This is entangled with echo, below — solve them together or the cheap echo fix forecloses barge-in.
- ~~**Wake word**~~ — **done**, see the stt section. `STT_WAKE_WORD` gates on the transcript and `STT_FOLLOW_UP_S` reopens the mic for a reply.

`SPEAKS` in `graph.py` exists for this phase — see the llm section.

### echo, and why it is a three-way choice

He hears his own TTS through the speakers; `scripts/test_stt_tts.py` demonstrates it plainly, transcribing himself and repeating forever. The wake word makes this **worse**, not better: he says his own name constantly, so his reply trips his own gate, and `STT_FOLLOW_UP_S` leaves the window open at exactly the wrong moment. Headphones remain the workaround until one of these lands.

1. **Gate the mic while `speak()` runs** and bin whatever queued. Five lines, completely effective — and it makes barge-in *impossible*, because the microphone is deaf precisely when you would interrupt. Do not reach for this if barge-in is still wanted.
2. **Match the transcript against what was just spoken** and discard it. We know exactly what went to Kokoro, so fuzzy-compare each transcript against the last few seconds of TTS text and drop the ones that match. Costs nothing — both strings already exist — needs no reference-signal alignment, works whatever the room does to the sound, and handles partial pickup if the match is fuzzy rather than exact. It also composes with the other two rather than competing.
3. **Acoustic echo cancellation** (WebRTC APM, speexdsp). The only one that stops the VAD firing at all, so it saves the wasted transcription the others still pay. Also the most work: the reference signal has to be time-aligned with the mic stream.

**Barge-in should require the wake word too**, and the reason is not the obvious one. Silence cannot trip the VAD — measured, silero scores digital silence at 0.009, hiss 0.056 and mains hum 0.011 against a 0.5 threshold — so an empty room is not the hazard. **His own voice through the speakers is**, and that scores like the speech it is. Note the wake word alone does not close this either, since he says his own name constantly; it wants pairing with option 2.

That choice has a price worth taking deliberately: gating barge-in on words makes it **transcription-gated**, so he keeps talking over you until the word has been heard and recognised. The floor is about **750ms** — 500ms of speech is enough for whisper to return "Tobias" (measured; 300ms yields only "To-" and fails the no-speech gate), plus ~250ms to transcribe a short clip. Stopping itself is instant, since the persistent `OutputStream` can `abort()` mid-sentence; it is the decision that is slow. **Genuinely instant barge-in needs real AEC**, because only option 3 makes it safe to react to voice activity alone.

Barge-in must **not** consume `segments()`, which yields only when an utterance closes and so pays `VAD_SILENCE_MS` on top — three seconds for "Tobias, stop talking" rather than 750ms. It needs its own path that transcribes a partial buffer while speech is still in progress.

Lowering `VAD_SILENCE_MS` during playback is **not** an alternative to that. While he speaks, the main thread is blocked in `stream.write()` and `listen()` is suspended at its `yield`, so nothing is running the detector at all — no VAD setting detects anything when nobody is looking. (It would also not take effect: `segments()` snapshots `hangover_frames` before its loop, unlike `vad_threshold` which it reads per frame. And it must never go through `config.update()`, which would persist a transient tweak to `state.json`.) Even fixed, it is the slower option, because it still waits for the whole utterance to end: ~1.4s for "Tobias, stop talking" against a flat ~750ms for partial transcription.

So barge-in needs a **watcher thread reading frames during playback** — the first genuinely concurrent code in the project. It earns that three times over, because `listen()` is suspended throughout, making the watcher the only consumer in that window: it delivers barge-in, it eats his own echoed voice before `listen()` can transcribe it, and it stops the unbounded mic queue growing at precisely the time it otherwise would. Design all three together.

Option 2 is the one to try first, but know its two failure modes. A genuine interruption that **overlaps** his speech produces a transcript mixing both voices, which may fuzzy-match enough to be discarded — the similarity threshold is the whole design. And a user who legitimately **repeats him** ("Did you say fifteen degrees?") looks exactly like echo, so the comparison must be restricted to a short window after speaking rather than the whole conversation.

**Phase 2** — dockerised minikube REST API on dedicated hardware, CI/CD, vLLM serving, monitoring. A Raspberry Pi runs Silero VAD locally and streams captured utterances to the API over wifi.

Phase 2 prerequisite, before the Dockerfile rather than after it: **move dependency management to `uv`** with a committed `uv.lock`, retiring the loose `requirements.txt` / frozen `requirements.lock` pair. TOBIAS is an application, not a library, so pinning is correct — nothing consumes it, so there are no resolution conflicts to cause, and reproducibility is the whole point of containerising. uv is also far faster at image build time.

Known and deliberately deferred to phase 2 — **do not pre-solve it**:

- **Backpressure.** The mic queue in `stt/vad.py` is unbounded, so audio piles up while transcription and synthesis run, and a slow turn falls behind real time.

Echo and barge-in were on this list and have moved up to phase 1.5, where they belong together.

**Streaming LLM output into TTS is already done** — `ask_stream` yields sentences, `speak_stream` pipelines them, both pulled forward from phase 2 because it was the largest latency win available.

Measured over a four-sentence reply, the whole playback path now costs **228ms of overhead in total** — about 76ms per sentence boundary.

Two rounds of optimisation got there, and the ordering is the lesson. Prefetching synthesis around `sd.play()` came first and was worth ~285ms per boundary. Trimming Kokoro's silence padding came second and was worth roughly a second per boundary — three times more, in the layer underneath. Moving to a persistent `OutputStream` came third, was simpler than the prefetching it deleted, and fixed two audible bugs the earlier work had been quietly causing. Measure the layer below before optimising the one in front of you.

A producer/consumer thread **would not fix this** — it would move synthesis off the main path, which prefetching already does. Only a single persistent callback-driven stream with a queue would, and that buys ~140ms at the cost of the first genuinely concurrent code in the project. Natural pauses between spoken sentences run 200-500ms, so the remaining seam is at or below what a human would leave. Do not build it without first hearing a problem.

## Gotchas

- Whisper hallucinates confident text on silence — a lone `"Thank you."` is its signature output for an empty buffer, and background noise that trips the VAD produces plausible whole sentences. The defences are layered, and it is worth knowing which does what. **Silero rejects non-speech outright** (hum peaks at 0.011, hiss 0.056, against 1.000 for speech), so pure noise never reaches `transcribe()` at all; `STT_MAX_NO_SPEECH` is the second line, for speech-*like* noise that gets past it and for buffers that are mostly silence.
- Gate on `no_speech_prob`, **not** `avg_logprob`. Logprob looks like the better signal on clean audio but collapses on short utterances: a 500ms "Tobias" scores **-0.361** against a -0.4 threshold, so one-word commands sit on the edge and a fractionally worse recording is silently dropped — which reads to the user as "it ignored me". `no_speech_prob` separates cleanly across everything measured: real speech (clean, 0dB SNR, whispered, one word, onset-clipped) all under **0.04**, silence and hiss over **0.11**.
- `VAD_SPEECH_PAD_MS` at 200 measurably **loses the first word**, and does so without any drop in confidence — the transcript comes back clean and simply missing "Tobias". Silero needs a moment of speech before it fires, and whatever it took is already gone. Padding is free (the audio is buffered anyway, and whisper pads to 30s regardless of input length), so be generous: 400ms.
- **Kokoro pads every clip with ~380ms of lead-in and ~530ms of tail.** Left in, that is a full second of dead air at each streamed sentence boundary and 380ms of pure latency before the first word. `synthesize()` trims it back to 60ms a side. Trimming a four-sentence reply removed **3.3 seconds** of silence from 10.4s of audio.
- The trim threshold is the delicate part. Kokoro's padding is true digital zero, so `SILENCE` only has to clear the noise floor — **0.001 is right, 0.01 audibly ate the final consonant of every sentence.** The quiet decay of a closing fricative measures around 0.003 and runs well past the last audible vowel. If speech sounds clipped at the edges, lower `SILENCE` before touching `KEEP_MS`, and never raise it to "clean up" the trim.
- A cold output device swallows the opening of the first stream it is handed — enough to turn "Good evening" into "d evening". Three things were needed, in order of how much they mattered: the stream must be **persistent** (a throwaway warm-up clip does nothing, because the device sleeps again the moment that stream closes), it gets `WAKE_MS` of silence written into it on open, and `speak()` calls `warm()` **before** `synthesize()` so the device spends the eight-second model load starting up rather than eating the first syllable. Opening it after synthesis was not enough even with 400ms of padding.
- `OutputStream.write()` returns while the device is still draining, so a process that exits straight after speaking cuts off its own last words. `playback._drain()` is registered with `atexit` to push silence through before stopping. Anything that ends the process abruptly will still clip the tail.
- **Full stops inside an initialism make Kokoro spell it out.** `T.O.B.I.A.S.` phonemises to `tˌiːˌQbˌiːˌIˌAˈɛs` — "tee-oh-bee-eye-ay-ess" — while `Tobias` gives `təbˈIəs`. The system prompt therefore names him "Tobias" throughout and forbids the dotted form; do not reintroduce it there, however tempting it looks in a heading. Check with `KPipeline(...)` result `.phonemes` rather than by ear.
- Both models run on the GPU, but they get there by different routes. CTranslate2 bundles its own CUDA runtime, so faster-whisper reaches `cuda` regardless of which torch is installed. Kokoro is a plain torch model and needs the CUDA build, which is why `requirements.txt` pins `torch==2.13.0+cu126` from the pytorch index. CUDA **12** is deliberate — it matches CTranslate2's bundled runtime, so the two never load conflicting cuBLAS/cuDNN DLLs in one process. Do not casually bump this to a cu13x wheel.
- **`pip install torch==2.13.0` is a silent no-op when `2.13.0+cpu` is installed** — pip ignores the local version tag when deciding "already satisfied", exits 0, and downloads nothing. Switching build flavours requires pinning the tag: `torch==2.13.0+cu126`. Check `torch.cuda.is_available()` afterwards rather than trusting the exit code.
- `import kokoro` alone costs **~20s** (it pulls spacy and transformers), and `import faster_whisper` ~10s. That is process-start cost, not per-turn cost, so it makes the test scripts feel broken while being irrelevant to the assistant loop, which imports once and stays up. Do not go looking for a synthesis bug here.
- Installing `kokoro` drags in ~60 packages (spacy, transformers, rdflib) and **downgrades `tokenizers`**, which faster-whisper also depends on. STT was re-verified on `cuda` afterwards and was unaffected. Re-run both modules after any dependency change; `requirements.lock` is the way back if one breaks.
- Verify GPU placement with `WhisperModel(...).model.device`, not by assuming; CTranslate2 raises rather than silently falling back, so a clean load on `cuda` is real.
