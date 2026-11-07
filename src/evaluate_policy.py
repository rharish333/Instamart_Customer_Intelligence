
from pathlib import Path
import sys
import json
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = PROJECT_ROOT / "data" / "policies"
EVAL_PATH = PROJECT_ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mock_llm
import policy_assistant


def normalize(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()


class LocalRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.texts = [item["text"] for item in chunks]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, k=3):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        indices = scores.argsort()[::-1][:k]

        return [
            {
                "source": self.chunks[i]["source"],
                "chunk_id": self.chunks[i]["chunk_id"],
                "text": self.chunks[i]["text"],
                "score": float(scores[i])
            }
            for i in indices
        ]


def build_no_overlap_chunks():
    chunks = []

    for path in sorted(POLICY_DIR.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()

        for start in range(0, len(words), 40):
            text = " ".join(words[start:start + 40])

            if text:
                chunks.append({
                    "source": path.name,
                    "chunk_id": start // 40,
                    "text": text
                })

    return chunks


def evaluate(label, retriever, questions):
    policy_assistant.retriever = retriever
    rows = []

    for _, item in questions.iterrows():
        question = str(item["question"])
        expected_doc = str(item["expected_doc"]).strip()
        expected_fact = str(item["expected_fact"]).strip()

        retrieved = retriever.retrieve(question, k=3)
        retrieved_docs = [r["source"] for r in retrieved]

        model_answer, sources = policy_assistant.answer(question)

        is_answerable = expected_doc.lower() != "none"

        doc_found = (
            expected_doc in retrieved_docs if is_answerable else True
        )

        fact_found = (
            normalize(expected_fact) in normalize(model_answer)
            if is_answerable else True
        )

        fallback_text = normalize(model_answer)

        fallback_found = (
            any(phrase in fallback_text for phrase in [
                "not covered",
                "could not find",
                "not available",
                "not mentioned",
                "cannot find"
            ])
            if not is_answerable else True
        )

        rows.append({
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "retrieved_docs": ";".join(retrieved_docs),
            "answer": model_answer,
            "doc_found": doc_found,
            "fact_found": fact_found,
            "fallback_found": fallback_found
        })

    results = pd.DataFrame(rows)

    answerable = results[
        results["expected_doc"].str.lower() != "none"
    ]

    unanswered = results[
        results["expected_doc"].str.lower() == "none"
    ]

    print(f"\n========== {label} ==========")
    print("Total questions:", len(results))
    print("Answerable questions:", len(answerable))
    print("Not-covered questions:", len(unanswered))
    print("Expected document found:",
          int(answerable["doc_found"].sum()), "/", len(answerable))
    print("Expected fact found:",
          int(answerable["fact_found"].sum()), "/", len(answerable))
    print("Correct fallback:",
          int(unanswered["fallback_found"].sum()), "/", len(unanswered))

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = pd.read_csv(EVAL_PATH).fillna("")

    original_retriever = policy_assistant.retriever

    mock_llm.reset()
    normal_results = evaluate(
        "OVERLAP = 10",
        original_retriever,
        questions
    )
    normal_path = OUTPUT_DIR / "policy_eval_overlap10.csv"
    normal_results.to_csv(normal_path, index=False)

    no_overlap_chunks = build_no_overlap_chunks()
    no_overlap_path = OUTPUT_DIR / "policy_chunks_no_overlap.json"

    with no_overlap_path.open("w", encoding="utf-8") as file:
        json.dump(no_overlap_chunks, file, indent=2)

    no_overlap_retriever = LocalRetriever(no_overlap_chunks)

    mock_llm.reset()
    zero_results = evaluate(
        "OVERLAP = 0",
        no_overlap_retriever,
        questions
    )
    zero_path = OUTPUT_DIR / "policy_eval_overlap0.csv"
    zero_results.to_csv(zero_path, index=False)

    policy_assistant.retriever = original_retriever

    failed = zero_results[
        (zero_results["expected_doc"].str.lower() != "none")
        & (
            ~zero_results["doc_found"]
            | ~zero_results["fact_found"]
        )
    ]

    print("\nZERO-OVERLAP ANSWERABLE FAILURES:", len(failed))

    if len(failed):
        print(failed[
            ["question", "expected_doc", "expected_fact"]
        ].to_string(index=False))

    print("\nPOLICY EVALUATION COMPLETED")
    print("Saved:", normal_path)
    print("Saved:", zero_path)
    print("Saved:", no_overlap_path)


if __name__ == "__main__":
    main()
