
from pathlib import Path
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "outputs" / "policy_chunks.json"


class PolicyRetriever:
    def __init__(self):
        with CHUNKS_PATH.open("r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        self.texts = [chunk["text"] for chunk in self.chunks]

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )

        self.matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, k=3):
        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.matrix
        )[0]

        ranked_indices = similarities.argsort()[::-1][:k]

        results = []

        for index in ranked_indices:
            results.append({
                "source": self.chunks[index]["source"],
                "chunk_id": self.chunks[index]["chunk_id"],
                "text": self.chunks[index]["text"],
                "score": round(float(similarities[index]), 4)
            })

        return results


if __name__ == "__main__":
    retriever = PolicyRetriever()

    query = "refund policy late delivery"
    results = retriever.retrieve(query, k=3)

    print("POLICY RETRIEVAL RESULTS")
    print("Query:", query)

    for number, result in enumerate(results, start=1):
        print(f"\nResult {number}")
        print("Source:", result["source"])
        print("Chunk:", result["chunk_id"])
        print("Similarity:", result["score"])
        print("Text:", result["text"])
