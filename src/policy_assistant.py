
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mock_llm
from policy_retriever import PolicyRetriever


retriever = PolicyRetriever()

SYSTEM_PROMPT = """
You are an Instamart POLICY ASSISTANT.

Answer customer questions using ONLY the policy context provided
in the user's message.

Rules:
1. Do not invent or assume policy information.
2. If the answer is not available in the context, clearly say:
   "This information is not covered in the available policies."
3. Give clear and simple answers.
4. Follow the policy exactly.
"""


def answer(question):
    results = retriever.retrieve(question, k=3)

    context_parts = []

    for result in results:
        context_parts.append(
            f'[{result["source"]}]\n{result["text"]}'
        )

    context = "\n".join(context_parts)

    user_message = (
        f"CONTEXT:\n{context}\n"
        f"QUESTION:\n{question}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    response = mock_llm.chat(messages)
    model_answer = response.choices[0].message.content

    sources = list(dict.fromkeys(
        result["source"] for result in results
    ))

    return model_answer, sources


if __name__ == "__main__":
    question = "What happens if an order is delivered after 45 minutes?"

    response, sources = answer(question)

    print("POLICY ASSISTANT")
    print("\nQuestion:", question)
    print("\nAnswer:", response)
    print("\nSources:")

    for source in sources:
        print("-", source)
