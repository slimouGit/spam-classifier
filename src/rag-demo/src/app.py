import os
from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_documents() -> List[Dict]:
    """
    Lädt alle .txt-Dateien aus dem data-Ordner.
    Jedes Dokument wird als Dictionary gespeichert.
    """

    documents = []

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(DATA_DIR, filename)

            with open(path, "r", encoding="utf-8") as file:
                text = file.read()

            documents.append({
                "source": filename,
                "text": text
            })

    return documents


def chunk_text(text: str, chunk_size: int = 250) -> List[str]:
    """
    Zerlegt lange Texte in kleinere Abschnitte.
    Bei echtem RAG ist das wichtig, damit gezielt relevante Passagen gefunden werden.
    """

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def build_chunks(documents: List[Dict]) -> List[Dict]:
    """
    Erstellt Chunks aus allen Dokumenten.
    Jeder Chunk behält seine Quelle.
    """

    chunks = []

    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            chunks.append({
                "source": doc["source"],
                "text": chunk
            })

    return chunks


class SimpleRAG:
    """
    Sehr einfache RAG-Demo.

    Retrieval:
    - TF-IDF wandelt Texte in Zahlenvektoren um.
    - Cosine Similarity findet ähnliche Textstellen.

    Generation:
    - Die Antwort wird aus den gefundenen Textstellen zusammengesetzt.
    - In einer echten App würde hier ein LLM stehen.
    """

    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.texts = [chunk["text"] for chunk in chunks]

        self.vectorizer = TfidfVectorizer()
        self.document_vectors = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, question: str, top_k: int = 2) -> List[Dict]:
        """
        Sucht die relevantesten Textstellen zur Nutzerfrage.
        """

        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.document_vectors)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for index in top_indices:
            results.append({
                "source": self.chunks[index]["source"],
                "text": self.chunks[index]["text"],
                "score": round(float(similarities[index]), 4)
            })

        return results

    def generate_answer(self, question: str, retrieved_chunks: List[Dict]) -> str:
        """
        Erzeugt eine einfache Antwort auf Basis der gefundenen Quellen.
        """

        if not retrieved_chunks or retrieved_chunks[0]["score"] == 0:
            return "Ich habe in den Dokumenten keine passende Information gefunden."

        context = "\n\n".join(
            f"- Quelle: {chunk['source']}\n  Inhalt: {chunk['text']}"
            for chunk in retrieved_chunks
        )

        answer = (
            f"Frage: {question}\n\n"
            f"Antwort auf Basis der gefundenen Dokumentstellen:\n\n"
            f"{context}"
        )

        return answer

    def ask(self, question: str) -> str:
        """
        Kompletter RAG-Ablauf:
        1. Frage entgegennehmen
        2. Relevante Stellen suchen
        3. Antwort erzeugen
        """

        retrieved_chunks = self.retrieve(question)
        return self.generate_answer(question, retrieved_chunks)


def main():
    documents = load_documents()
    chunks = build_chunks(documents)

    rag = SimpleRAG(chunks)

    print("RAG Demo")
    print("--------")
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

        answer = rag.ask(question)

        print("\n" + answer)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()