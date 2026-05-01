import os
import joblib


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sms_spam_model.joblib")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modell nicht gefunden: {MODEL_PATH}\n"
            "Bitte zuerst src/train_model.py ausführen."
        )

    return joblib.load(MODEL_PATH)


def predict_sms(text: str):
    model = load_model()

    prediction = model.predict([text])[0]
    probability = model.predict_proba([text])[0]

    return {
        "text": text,
        "prediction": "SPAM" if prediction == 1 else "HAM",
        "spam_probability": round(float(probability[1]), 4)
    }


if __name__ == "__main__":
    sms = input("SMS eingeben: ")
    result = predict_sms(sms)

    print()
    print("Ergebnis:")
    print("Text:", result["text"])
    print("Vorhersage:", result["prediction"])
    print("Spam-Wahrscheinlichkeit:", result["spam_probability"])