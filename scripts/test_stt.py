from tobias.stt import listen

if __name__ == "__main__":
    print("Listening (ctrl-c to stop)...")
    try:
        for text in listen():
            print(f"> {text}")
    except KeyboardInterrupt:
        pass
