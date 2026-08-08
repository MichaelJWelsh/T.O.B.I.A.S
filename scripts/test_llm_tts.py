from tobias.llm import ask_stream
from tobias.tts import speak, speak_stream


def shown(sentences):
    for sentence in sentences:
        print(f"TOBIAS > {sentence}")
        yield sentence


if __name__ == "__main__":
    speak("Good evening, sir. All systems are online.")
    print("Type to TOBIAS (ctrl-c to stop).")
    try:
        while True:
            if text := input("you > ").strip():
                speak_stream(shown(ask_stream(text)))
                print()
    except (KeyboardInterrupt, EOFError):
        pass
