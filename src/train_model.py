import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "sms_spam.tsv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sms_spam_model.joblib")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Datensatz nicht gefunden: {DATA_PATH}\n"
            "Bitte zuerst src/download_data.py ausführen."
        )

    df = pd.read_csv(DATA_PATH, sep="\t")

    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("Die Datei muss die Spalten 'label' und 'text' enthalten.")

    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    df = df.dropna(subset=["label", "text"])

    return df


def train():
    df = load_data()

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print()
    print(classification_report(y_test, predictions))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print()
    print("Modell gespeichert unter:")
    print(MODEL_PATH)


if __name__ == "__main__":
    train()