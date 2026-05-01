from predict import predict_sms


def main():
    print("SMS Spam Classifier")
    print("-------------------")
    print("Gib eine SMS ein. Mit 'exit' beenden.")
    print()

    while True:
        sms = input("SMS: ")

        if sms.lower() in ["exit", "quit", "q"]:
            print("App beendet.")
            break

        if not sms.strip():
            print("Bitte eine SMS eingeben.")
            continue

        result = predict_sms(sms)

        print()
        print("Vorhersage:", result["prediction"])
        print("Spam-Wahrscheinlichkeit:", result["spam_probability"])
        print()


if __name__ == "__main__":
    main()