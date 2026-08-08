from tobias.tts import speak

if __name__ == "__main__":
    speak("Good evening, sir. All systems are online.")  # also warms the model
    print("Type a line for TOBIAS to say (ctrl-c to stop).")
    try:
        while True:
            if text := input("> ").strip():
                speak(text)
    except (KeyboardInterrupt, EOFError):
        pass
