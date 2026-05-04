import os
from typing import List, Dict

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


def load_documents() -> List[Dict]:
    """
    Lädt alle .txt-Dateien aus dem data-Ordner.
    """

    documents = []

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(
            f"Data-Ordner nicht gefunden: {DATA_DIR}"
        )

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(DATA_DIR, filename)

            with open(path, "r", encoding="utf-8") as file:
                text = file.read()

            documents.append({
                "source": filename,
                "text": text
            })

    if not documents:
        raise ValueError("Keine .txt-Dateien im data-Ordner gefunden.")

    return documents


def chunk_text(text: str, chunk_size: int = 120) -> List[str]:
    """
    Zerlegt Text in kleinere Abschnitte.
    """

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def build_chunks(documents: List[Dict]) -> List[Dict]:
    """
    Baut Chunks aus allen Dokumenten.
    """

    chunks = []

    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            chunks.append({
                "source": doc["source"],
                "text": chunk
            })

    return chunks


def ask_ollama(prompt: str) -> str:
    """
    Sendet den Prompt an Ollama und gibt die Antwort zurück.
    """

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )
        response.raise_for_status()

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Keine Verbindung zu Ollama. Prüfe, ob Ollama läuft."
        )

    except requests.exceptions.Timeout:
        raise TimeoutError(
            "Ollama hat zu lange für die Antwort gebraucht."
        )

    data = response.json()
    return data.get("response", "").strip()


class OllamaRAG:
    """
    Einfache RAG-App:

    1. Retrieval:
       Sucht passende Textstellen per TF-IDF + Cosine Similarity.

    2. Augmentation:
       Übergibt die gefundenen Textstellen als Kontext an das LLM.

    3. Generation:
       Ollama erzeugt daraus eine Antwort.
    """

    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.texts = [chunk["text"] for chunk in chunks]

        self.vectorizer = TfidfVectorizer()
        self.document_vectors = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        """
        Sucht die relevantesten Dokumentstellen zur Frage.
        """

        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(
            question_vector,
            self.document_vectors
        )[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for index in top_indices:
            score = float(similarities[index])

            if score > 0:
                results.append({
                    "source": self.chunks[index]["source"],
                    "text": self.chunks[index]["text"],
                    "score": round(score, 4)
                })

        return results

    def build_prompt(self, question: str, retrieved_chunks: List[Dict]) -> str:
        """
        Baut den Prompt für Ollama.
        """

        context = "\n\n".join(
            f"Quelle: {chunk['source']}\nText: {chunk['text']}"
            for chunk in retrieved_chunks
        )

        prompt = f"""
Du bist ein sachlicher Assistent.

Beantworte die Frage ausschließlich anhand des folgenden Kontexts.
Wenn die Antwort nicht im Kontext steht, sage:
"Diese Information steht nicht in den bereitgestellten Dokumenten."

Kontext:
{context}

Frage:
{question}

Antwort:
""".strip()

        return prompt

    def ask(self, question: str) -> Dict:
        """
        Führt den gesamten RAG-Ablauf aus.
        """

        retrieved_chunks = self.retrieve(question)

        if not retrieved_chunks:
            return {
                "answer": "Ich habe keine passende Stelle in den Dokumenten gefunden.",
                "sources": []
            }

        prompt = self.build_prompt(question, retrieved_chunks)
        answer = ask_ollama(prompt)

        return {
            "answer": answer,
            "sources": retrieved_chunks
        }


def main():
    documents = load_documents()
    chunks = build_chunks(documents)

    rag = OllamaRAG(chunks)

    print("Ollama RAG Demo")
    print("----------------")
    print("Stelle eine Frage zu den Dokumenten.")
    print("Mit 'exit' beenden.\n")

    while True:
        question = input("Frage: ")

        if question.lower() in ["exit", "quit", "q"]:
            print("App beendet.")
            break

        if not question.strip():
            print("Bitte eine Frage eingeben.\n")
            continue

        result = rag.ask(question)

        print("\nAntwort:")
        print(result["answer"])

        print("\nQuellen:")
        for source in result["sources"]:
            print(
                f"- {source['source']} "
                f"(Score: {source['score']})"
            )

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()