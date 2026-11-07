from pathlib import Path
import re
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = PROJECT_ROOT / "data" / "policies"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHUNK_SIZE = 40
OVERLAP = 10


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = []

    for policy_file in sorted(POLICY_DIR.glob("*.txt")):
        text = policy_file.read_text(encoding="utf-8")
        chunks = split_into_chunks(text)

        for index, chunk in enumerate(chunks):
            all_chunks.append({
                "source": policy_file.name,
                "chunk_id": index,
                "text": chunk
            })

    output_path = OUTPUT_DIR / "policy_chunks.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(all_chunks, file, indent=2, ensure_ascii=False)

    print("POLICY CHUNKING COMPLETED")
    print("Policy documents:", len(list(POLICY_DIR.glob("*.txt"))))
    print("Total chunks:", len(all_chunks))
    print("Saved:", output_path)

    print("\nCHUNKS PER POLICY:")
    counts = {}
    for chunk in all_chunks:
        counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1

    for source, count in counts.items():
        print(source, ":", count)


if __name__ == "__main__":
    main()