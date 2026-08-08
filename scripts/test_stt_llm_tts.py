from tobias.llm import ask_stream
from tobias.stt import listen
from tobias.tts import speak, speak_stream


def shown(sentences):
    for sentence in sentences:
        print(f"TOBIAS > {sentence}")
        yield sentence


if __name__ == "__main__":
    speak("Good evening, sir. All systems are online.")
    print("Wear headphones — with speakers TOBIAS hears himself and talks to himself.")
    print("Speak to TOBIAS (ctrl-c to stop)...")
    try:
        for text in listen():
            print(f"you > {text}")
            speak_stream(shown(ask_stream(text)))
            print()
    except KeyboardInterrupt:
        pass
