from tobias.stt import listen
from tobias.tts import speak

if __name__ == "__main__":
    speak("Good evening, sir. All systems are online.")
    print("Wear headphones — with speakers TOBIAS hears itself and repeats forever.")
    print("Listening (ctrl-c to stop)...")
    try:
        for text in listen():
            print(f"> {text}")
            speak(text)
    except KeyboardInterrupt:
        pass
