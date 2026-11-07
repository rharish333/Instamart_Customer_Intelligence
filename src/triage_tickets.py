
from pathlib import Path
import sys
import re
import json
import time
import random
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import mock_llm

ALLOWED_CATEGORIES = {
    "billing", "delivery", "technical", "fraud"
}

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05

# --------------------------------------------------
# 1. LOAD TICKETS
# --------------------------------------------------

tickets = pd.read_csv(
    BASE_DIR / "data" / "raw" / "tickets.csv",
    keep_default_na=False
)

# --------------------------------------------------
# 2. REDACT PERSONAL INFORMATION
# --------------------------------------------------

EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\b[6-9]\d{9}\b"
ORDER_PATTERN = r"ORD-\d{6}"

def redact_text(text):
    text = str(text)

    text, emails = re.subn(
        EMAIL_PATTERN, "[EMAIL]", text
    )

    text, phones = re.subn(
        PHONE_PATTERN, "[PHONE]", text
    )

    text, orders = re.subn(
        ORDER_PATTERN, "[ORDER_ID]", text
    )

    return text, emails, phones, orders


redaction_results = tickets["text"].apply(redact_text)

tickets["redacted_text"] = redaction_results.apply(
    lambda x: x[0]
)

email_count = sum(x[1] for x in redaction_results)
phone_count = sum(x[2] for x in redaction_results)
order_count = sum(x[3] for x in redaction_results)

print("REDACTION RESULTS")
print("Emails:", email_count)
print("Phones:", phone_count)
print("Order IDs:", order_count)
print("Total:", email_count + phone_count + order_count)

# Verify that no personal information remains
for text in tickets["redacted_text"]:
    assert not re.search(EMAIL_PATTERN, text)
    assert not re.search(PHONE_PATTERN, text)
    assert not re.search(ORDER_PATTERN, text)

tickets.to_csv(
    BASE_DIR / "outputs" / "redacted_tickets.csv",
    index=False
)

# --------------------------------------------------
# 3. ZERO-SHOT AND FEW-SHOT PROMPTS
# --------------------------------------------------

ZERO_SHOT_PROMPT = """
CLASSIFY customer support tickets.

Allowed categories:
billing, delivery, technical, fraud.

Return JSON only in this exact form:
{"category": "...", "reason": "..."}
The category must be one of the four allowed categories.
"""

FEW_SHOT_PROMPT = """
CLASSIFY customer support tickets.

Allowed categories:
billing, delivery, technical, fraud.

EXAMPLES:
Ticket: My card was used without permission.
Category: fraud

Ticket: My invoice shows the wrong amount.
Category: billing

Ticket: The delivery rider arrived late.
Category: delivery

Ticket: The app crashes when I open it.
Category: technical

Return JSON only in this exact form:
{"category": "...", "reason": "..."}
The category must be one of the four allowed categories.
"""

# --------------------------------------------------
# 4. VALIDATE MODEL RESPONSE
# --------------------------------------------------

def validate_response(content):

    try:
        data = json.loads(content)

        if (
            isinstance(data, dict)
            and data.get("category") in ALLOWED_CATEGORIES
        ):
            return data["category"], "classified_json"

        return "", "flagged_invalid_output"

    except (json.JSONDecodeError, TypeError):

        words = set(
            re.findall(r"[a-z]+", str(content).lower())
        )

        found = words.intersection(ALLOWED_CATEGORIES)

        if len(found) == 1:
            return found.pop(), "classified_keyword"

        return "", "flagged_invalid_output"


# --------------------------------------------------
# 5. RETRY FUNCTION
# --------------------------------------------------

def call_with_retry(messages, ticket_id):

    for attempt in range(4):

        try:
            response = mock_llm.chat(messages)

            return response

        except mock_llm.TransientLLMError:

            if attempt == 3:
                raise

            wait = (
                BASE_DELAY
                * (2 ** attempt)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"RETRY ticket={ticket_id} "
                f"retry={attempt} wait={wait:.4f}s"
            )

            time.sleep(wait)


# --------------------------------------------------
# 6. RUN CLASSIFICATION PIPELINE
# --------------------------------------------------

def run_pipeline(system_prompt, token_budget=None):

    results = []
    total_tokens = 0

    for _, ticket in tickets.iterrows():

        ticket_id = ticket["ticket_id"]
        original_text = ticket["text"]
        safe_text = ticket["redacted_text"]

        category = ""
        status = ""
        ticket_tokens = 0

        # Empty ticket guardrail
        if not original_text.strip():

            status = "skipped_empty"

        # Long ticket guardrail
        elif mock_llm.est_tokens(original_text) > MAX_INPUT_TOKENS:

            status = "flagged_long"

        # Token budget guardrail
        elif (
            token_budget is not None
            and total_tokens >= token_budget
        ):

            status = "budget_exhausted"

        else:

            # Privacy assertion before model call
            assert not re.search(EMAIL_PATTERN, safe_text)
            assert not re.search(PHONE_PATTERN, safe_text)
            assert not re.search(ORDER_PATTERN, safe_text)

            # User message must contain ONLY redacted text
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": safe_text
                }
            ]

            try:
                response = call_with_retry(
                    messages, ticket_id
                )

                ticket_tokens = (
                    response.usage.total_tokens
                )

                total_tokens += ticket_tokens

                content = response.choices[0].message.content

                category, status = validate_response(
                    content
                )

            except mock_llm.TransientLLMError:

                status = "permanent_error"

        results.append({
            "ticket_id": ticket_id,
            "customer_id": ticket["customer_id"],
            "status": status,
            "category": category,
            "true_category": ticket["true_category"],
            "tokens_used": ticket_tokens
        })

    return pd.DataFrame(results), total_tokens


# --------------------------------------------------
# 7. ZERO-SHOT COMPARISON RUN
# --------------------------------------------------

print("\n========== ZERO-SHOT RUN ==========")

mock_llm.reset()
random.seed(42)

zero_results, zero_tokens = run_pipeline(
    ZERO_SHOT_PROMPT
)

zero_classified = zero_results[
    zero_results["status"].isin([
        "classified_json",
        "classified_keyword"
    ])
]

zero_accuracy = (
    zero_classified["category"]
    == zero_classified["true_category"]
).mean() * 100

print("\nZERO-SHOT STATUS COUNTS:")
print(zero_results["status"].value_counts())

print("\nZERO-SHOT CLASSIFIED:", len(zero_classified))
print("ZERO-SHOT ACCURACY:", round(zero_accuracy, 2), "%")
print("ZERO-SHOT TOTAL TOKENS:", zero_tokens)

print(
    "TOKENS PER CLASSIFIED TICKET:",
    round(zero_tokens / len(zero_classified), 2)
)

print("\nZERO-SHOT CONFUSION TABLE:")

print(pd.crosstab(
    zero_classified["true_category"],
    zero_classified["category"],
    rownames=["Actual"],
    colnames=["Predicted"]
))

# --------------------------------------------------
# 8. FEW-SHOT COMPARISON RUN
# --------------------------------------------------

print("\n========== FEW-SHOT RUN ==========")

mock_llm.reset()
random.seed(42)

few_results, few_tokens = run_pipeline(
    FEW_SHOT_PROMPT
)

few_classified = few_results[
    few_results["status"].isin([
        "classified_json",
        "classified_keyword"
    ])
]

few_accuracy = (
    few_classified["category"]
    == few_classified["true_category"]
).mean() * 100

print("\nFEW-SHOT STATUS COUNTS:")
print(few_results["status"].value_counts())

print("\nFEW-SHOT CLASSIFIED:", len(few_classified))
print("FEW-SHOT ACCURACY:", round(few_accuracy, 2), "%")
print("FEW-SHOT TOTAL TOKENS:", few_tokens)

print(
    "TOKENS PER CLASSIFIED TICKET:",
    round(few_tokens / len(few_classified), 2)
)

# --------------------------------------------------
# 9. TOKEN BUDGET DEMONSTRATION
# --------------------------------------------------

print("\n========== TOKEN BUDGET DEMO ==========")

mock_llm.reset()
random.seed(42)

budget_results, budget_tokens = run_pipeline(
    ZERO_SHOT_PROMPT,
    token_budget=6000
)

budget_results.to_csv(
    BASE_DIR / "outputs" / "triage_budget_demo.csv",
    index=False
)

print("\nBUDGET USED: 6000")
print(
    "Budget exhausted tickets:",
    (budget_results["status"] == "budget_exhausted").sum()
)
print("Tokens used:", budget_tokens)

# --------------------------------------------------
# 10. FINAL FEW-SHOT RUN WITH ROUTING
# --------------------------------------------------

print("\n========== FINAL FEW-SHOT RUN ==========")

mock_llm.reset()
random.seed(42)

final_results, final_tokens = run_pipeline(
    FEW_SHOT_PROMPT,
    token_budget=100000
)

# Load Part B customer segments
segments = pd.read_csv(
    BASE_DIR / "outputs" / "customer_segments.csv"
)

final_results = final_results.merge(
    segments[
        ["customer_id", "segment_name", "is_premium"]
    ],
    on="customer_id",
    how="left"
)

# Apply routing rules
def assign_route(row):

    status = row["status"]
    category = row["category"]

    if status in [
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
        "permanent_error"
    ]:

        return "human_review", "NORMAL"

    if category == "fraud":

        return "security_team", "HIGH"

    if row["is_premium"] == True:

        return "premium_desk", "HIGH"

    return "support_queue", "NORMAL"


routing = final_results.apply(
    assign_route,
    axis=1
)

final_results["route"] = routing.apply(
    lambda x: x[0]
)

final_results["priority"] = routing.apply(
    lambda x: x[1]
)

# Final required columns
final_output = final_results[
    [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used"
    ]
].copy()

final_output["category"] = (
    final_output["category"].fillna("")
)

final_output["segment_name"] = (
    final_output["segment_name"].fillna("")
)

final_output.to_csv(
    BASE_DIR / "outputs" / "triage_results.csv",
    index=False
)

print("\nFINAL STATUS COUNTS:")
print(final_output["status"].value_counts())

print("\nFINAL ROUTING COUNTS:")
print(final_output["route"].value_counts())

print("\nFINAL TOKENS USED:", final_tokens)

print("\nPART C PIPELINE COMPLETED")
print("Saved: outputs/triage_budget_demo.csv")
print("Saved: outputs/triage_results.csv")
