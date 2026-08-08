from tobias.llm import ask

if __name__ == "__main__":
    print("Talk to TOBIAS (ctrl-c to stop).")
    try:
        while True:
            if text := input("you > ").strip():
                print(f"TOBIAS > {ask(text)}\n")
    except (KeyboardInterrupt, EOFError):
        pass
