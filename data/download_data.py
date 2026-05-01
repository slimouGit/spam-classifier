import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "sms_spam.tsv")

URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    df = pd.read_csv(
        URL,
        sep="\t",
        header=None,
        names=["label", "text"]
    )

    df.to_csv(DATA_PATH, sep="\t", index=False)

    print(f"Datensatz gespeichert unter: {DATA_PATH}")
    print(df.head())


if __name__ == "__main__":
    download_dataset()