# Instamart Customer Intelligence

## Overview

This project analyzes synthetic Instamart customer, product, order, and support-ticket data. It combines data cleaning, SQL analytics, customer segmentation, churn prediction, support-ticket classification, policy retrieval, and an automated refund agent.

The project is divided into four parts:

- **Part A:** Data cleaning, database creation, SQL analysis, and validation.
- **Part B:** RFM analysis, customer segmentation, and lapse prediction.
- **Part C:** Support-ticket redaction, classification, and routing.
- **Part D:** Policy-based retrieval assistant and ReAct refund agent.

All experiments use the supplied synthetic data and offline mock LLM. No external LLM API key is required.

## Reproduce

### Requirements

- Python 3.11 or compatible version
- pip
- Windows, macOS, or Linux

### Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```cmd
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Run the project

Run the following commands from the project root, in order:

```bash
python generate_data.py
python src/clean_data.py
python src/create_database.py
python src/run_queries.py
python src/validate_data.py

python src/build_rfm.py
python src/cluster_customers.py
python src/profile_segments.py
python src/build_lapse_dataset.py
python src/train_lapse_models.py
python src/analyze_lapse_model.py
python src/create_retention_list.py

python src/triage_tickets.py

python src/build_policy_index.py
python src/evaluate_policy.py
python src/run_agent_scenarios.py
python src/test_agent_limits.py
```

The scripts generate cleaned datasets, the SQLite database, model files, evaluation results, JSONL agent logs, and checkpoint files.

The supplied `generate_data.py` and `mock_llm.py` files are kept unmodified.

## Part A

### Data cleaning

The supplied data contains customer, product, order, and support-ticket information.

The raw data audit identified:

- 1,200 customers.
- 40 products.
- 13,084 raw order rows.
- 119 exact duplicate order rows.
- 656 pending orders.
- 120 non-numeric order amounts.
- 106 negative delivery-time values.
- 2,541 blank delivery-time values.
- 6 out-of-range customer ages.
- 24 city spelling variations.
- 10 gender spelling variations.

The cleaning process removed duplicate orders, excluded invalid and pending order records from the final order dataset, standardized categorical values, and handled missing or invalid values.

### Cleaned datasets

| Dataset | Rows |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 12,309 |

Cleaned order status counts:

- Completed: 11,110
- Cancelled: 1,199

Additional validation:

- Customer ages range from 18 to 60.
- No missing customer ages remain.
- No missing order amounts remain.
- Completed orders have no missing delivery times.
- Cancelled orders have no delivery times.

Completed-order revenue: **INR 2,998,668.80**.

### SQLite database and SQL analysis

The database is saved as `data/instamart.db`.

It contains three tables:

- `customers`: 1,200 rows
- `products`: 40 rows
- `orders`: 12,309 rows

Seven SQL queries are stored in `sql/queries.sql`. Their results are saved as `outputs/query_1_results.csv` through `outputs/query_7_results.csv`.

The cumulative completed-order revenue over the period 1 April 2025 to 30 September 2025 was INR 2,998,668.80 over 183 days.

The validation script completed successfully with all checkpoints passed.

## Part B

### RFM analysis

RFM analysis was performed using recency, frequency, and monetary value.

The resulting dataset contains 1,175 customers.

| Metric | Result |
|---|---:|
| Median recency | 19 |
| Minimum recency | 0 |
| Maximum recency | 180 |
| Mean frequency | 9.46 |
| Maximum frequency | 33 |
| Mean monetary value | INR 2,552.06 |
| Maximum monetary value | INR 14,766.55 |

### Customer segmentation

K-Means clustering was evaluated using silhouette scores. Four clusters gave the highest silhouette score of **0.526**.

| Segment | Customers | Average Recency | Average Frequency | Average Monetary Value |
|---|---:|---:|---:|---:|
| Occasional Customers | 436 | 28.86 | 3.77 | INR 1,057.62 |
| High-Value Customers | 138 | 24.12 | 6.77 | INR 7,129.70 |
| Frequent Customers | 419 | 8.77 | 19.40 | INR 3,410.02 |
| At-Risk Customers | 182 | 109.04 | 2.20 | INR 685.99 |

The PCA visualization retained two components explaining 63.4% and 21.8% of the variance, respectively. Together, they explain **85.2%** of the variance.

### Customer lapse prediction

The lapse dataset contains 1,156 customers:

- Lapsed customers: 443
- Active customers: 713
- Lapse rate: 38.32%

The data was split into 924 training rows and 232 test rows.

Model results:

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Logistic Regression | 0.655 | 0.809 | 0.724 |
| Random Forest | 0.687 | 0.764 | 0.723 |
| Balanced Logistic Regression | 0.592 | 0.944 | 0.727 |

The balanced Logistic Regression model was saved as `outputs/lapse_model.joblib`.

The balanced model used oversampling on the training data only. The reported test confusion matrix was:

```text
[[85, 58],
 [ 5, 84]]
```

The model was saved and reloaded successfully, and its predictions remained identical.

The most important Random Forest features were:

| Feature | Importance |
|---|---:|
| Frequency | 0.319655 |
| Monetary | 0.220245 |
| Recency | 0.087496 |
| Cancellation rate | 0.069724 |
| Average discount | 0.069668 |

A retention list containing the top 10 premium customers was generated in `outputs/retention_list.csv`. One of these 10 customers was lapsed. The highest predicted lapse probability in this list was 0.742795.

The retention list is illustrative. Its premium flag is based on the full RFM period, including September, so it should not be interpreted as a leakage-free prospective targeting evaluation.

The training and test split was performed before model fitting. The encoder, scaler, and resampling operations were fitted using training data only. Data from September 2025 onward was not used as model features, and the customer segment labels from the segmentation task were not used as lapse-model features.

## Part C

### Support-ticket privacy and redaction

The ticket dataset contains 164 support tickets.

Before classification, sensitive information was redacted.

| Redacted item | Count |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order identifiers | 132 |
| Total redactions | 278 |

The redaction placeholders are `[EMAIL]`, `[PHONE]`, and `[ORDER_ID]`.

### Zero-shot and few-shot classification

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Accuracy | 85.71% | 100% |
| Total tokens | 10,804 | 18,292 |
| Tokens per classified ticket | 73.50 | 124.44 |

The few-shot approach improved classification accuracy on this dataset but used more tokens per ticket.

The classification status counts were:

| Status | Tickets |
|---|---:|
| Classified using JSON | 135 |
| Classified using keyword fallback | 12 |
| Flagged invalid output | 9 |
| Skipped empty | 4 |
| Flagged long | 4 |

The zero-shot confusion results were:

| Actual category | Predicted category | Count |
|---|---|---:|
| Billing | Billing | 38 |
| Delivery | Delivery | 36 |
| Fraud | Billing | 11 |
| Fraud | Fraud | 23 |
| Technical | Billing | 10 |
| Technical | Technical | 29 |

### Budget and routing

The budget demonstration used a token budget of 6,000. It used 6,033 tokens and left 69 tickets exhausted under the budget limit.

Final routing counts:

| Destination | Tickets |
|---|---:|
| Support queue | 102 |
| Security team | 34 |
| Human review | 17 |
| Premium desk | 11 |

Invalid-output, long, and other flagged cases are handled through the configured fallback and review logic rather than being treated as ordinary successful classifications.

Nineteen transient retries were recovered successfully.

Generated files include:

- `outputs/redacted_tickets.csv`
- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`

## Part D

### Policy retrieval assistant

The five supplied policy documents were split into 15 text chunks using a chunk size of 40 words and an overlap of 10 words.

The policy retriever uses TF-IDF vectorization and cosine similarity to retrieve the three most relevant policy chunks.

The policy assistant sends the retrieved policy context and the user's question to the supplied mock LLM. It is instructed to answer using the retrieved context and avoid inventing policy details.

### Retrieval evaluation

The evaluation dataset contains 14 questions:

- 12 answerable questions.
- 2 questions not covered by the supplied policies.

| Evaluation metric | Overlap 10 | Overlap 0 |
|---|---:|---:|
| Expected document retrieved | 12/12 | 12/12 |
| Expected fact retrieved | 12/12 | 9/12 |
| Correct fallback for uncovered questions | 2/2 | 2/2 |

Chunk overlap improved the retrieval of expected facts in this evaluation.

The evaluation outputs are:

- `outputs/policy_eval_overlap10.csv`
- `outputs/policy_eval_overlap0.csv`
- `outputs/policy_chunks.json`
- `outputs/policy_chunks_no_overlap.json`

### Refund tools and ReAct agent

The refund agent uses five tools:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`
5. `escalate_to_human`

The agent follows the refund policy:

- Only completed orders are eligible.
- A delivery taking more than 45 minutes qualifies for a full refund.
- Refunds up to INR 1,000 can be issued automatically.
- Refunds above INR 1,000 require human approval.
- Cancelled or missing orders are not refunded.

The agent saves checkpoints after tool observations. It supports resuming from a saved checkpoint through `resume_agent(run_id)`.

### Scenario evaluation

All seven scenarios passed.

| Scenario | Expected result | Actual result | LLM calls | Tokens | Result |
|---|---|---|---:|---:|---|
| S1 | REFUND | REFUND | 5 | 1,748 | PASS |
| S2 | REFUND | REFUND | 5 | 1,748 | PASS |
| S3 | ESCALATE | ESCALATE | 5 | 1,739 | PASS |
| S4 | DENIED | DENIED | 4 | 1,258 | PASS |
| S5 | DENIED | DENIED | 3 | 804 | PASS |
| S6 | NOT_FOUND | NOT_FOUND | 2 | 353 | PASS |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS | 6 | 4,068 | PASS |

Total: **7 passed, 0 failed**.

### Agent limit and recovery tests

| Test | Expected result | Actual result | Result |
|---|---|---|---|
| Token limit | FALLBACK_TOKEN_LIMIT | FALLBACK_TOKEN_LIMIT | PASS |
| Timeout | FALLBACK_TIMEOUT | FALLBACK_TIMEOUT | PASS |
| Maximum iterations | FALLBACK_MAX_ITERATIONS | FALLBACK_MAX_ITERATIONS | PASS |
| Crash recovery | REFUND | REFUND | PASS |

In the crash recovery test, the agent crashed after two tool executions. It resumed from its checkpoint and completed the remaining steps in the correct order:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`

The final refund was INR 128.25.

### ReAct vs Plan-Execute

The implemented agent uses the ReAct approach: it calls the model, observes the response, executes tools, and then calls the model again. In the measured scenarios, this required between 2 and 6 LLM calls. For example, S1 used 5 calls and 1,748 tokens, while S6 used 2 calls and 353 tokens.

A Plan-Execute design would first generate a plan and then execute its steps, potentially requiring fewer model calls when the plan is correct. However, unexpected observations such as a missing order or cancelled order may require replanning. ReAct is preferred for this refund workflow because each tool observation can influence the next decision, helping the agent respond to actual order and policy information. A separate Plan-Execute baseline was not implemented or benchmarked, so no measured token savings are claimed.

### Part D outputs

- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `outputs/policy_chunks.json`
- `state/` checkpoint files

Each agent scenario record includes its status, final answer, LLM call count, token usage, and tool steps with tool names, arguments, and observations.

## Limitations and Next Steps

- The project uses synthetic data, so the results may not represent real Instamart customer behavior.
- The supplied mock LLM is used for offline experimentation. Results may differ from a production language model.
- TF-IDF retrieval relies on lexical similarity and may not understand semantic meaning as well as embedding-based retrieval.
- The lapse model is evaluated on a held-out test split from the supplied dataset. Performance on future real-world customers is not established.
- The retention list is illustrative and uses a premium flag based on the full RFM period.
- The refund agent records simulated refund actions in a local ledger. It does not connect to a real payment gateway or issue real refunds.
- The Plan-Execute comparison is conceptual; a separate implementation and benchmark would be needed for a quantitative comparison.

Possible future improvements include testing on real anonymized data, using semantic embeddings for policy retrieval, monitoring model performance over time, and integrating the agent with authenticated production systems and human approval workflows.