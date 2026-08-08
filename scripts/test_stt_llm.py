from tobias.llm import ask
from tobias.stt import listen

if __name__ == "__main__":
    print("Speak to TOBIAS (ctrl-c to stop)...")
    try:
        for text in listen():
            print(f"you > {text}")
            print(f"TOBIAS > {ask(text)}\n")
    except KeyboardInterrupt:
        pass
