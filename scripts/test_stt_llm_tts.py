from tobias.llm import ask_stream
from tobias.stt import listen, watching
from tobias.tts import speak, speak_stream


def shown(sentences):
    for sentence in sentences:
        print(f"TOBIAS > {sentence}")
        yield sentence


if __name__ == "__main__":
    speak("Good evening, sir. All systems are online.")
    print("Wear headphones — with speakers TOBIAS hears himself and talks to himself.")
    print("Say his name while he is talking to cut him off.")
    print("Speak to TOBIAS (ctrl-c to stop)...")
    try:
        for text in listen():
            while text:
                print(f"you > {text}")
                with watching() as interruption:
                    if speak_stream(shown(ask_stream(text)), interruption):
                        print("  (interrupted)")
                print()
                # A barge-in that carried a request becomes the next one, so it need not be
                # repeated. An empty string means he was only told to stop.
                text = interruption.said
    except KeyboardInterrupt:
        pass
