
import json
import re
import pandas as pd

from refund_agent import run_agent


def extract_amount(result):
    for item in result.get("tool_history", []):
        if item.get("tool") in ("issue_refund", "escalate_to_human"):
            observation = item.get("observation", {})

            if observation.get("success"):
                amount = observation.get("refund_inr")
                if amount is not None:
                    return float(amount)

            amount = item.get("arguments", {}).get("refund_inr")
            if amount is not None:
                return float(amount)

    answer = result.get("answer", "")
    match = re.search(
        r"INR\s*([\d,]+(?:\.\d+)?)",
        answer,
        re.IGNORECASE
    )

    if match:
        return float(match.group(1).replace(",", ""))

    return 0.0


def main():
    scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
    output_file = "outputs/agent_runs.jsonl"

    results = []
    passed = 0

    print("REFUND AGENT SCENARIO EVALUATION")
    print("=" * 60)

    with open(output_file, "w", encoding="utf-8") as output:

        for _, scenario in scenarios.iterrows():

            scenario_id = str(scenario["scenario_id"])
            message = str(scenario["message"])
            expected = str(scenario["expected_outcome"]).upper()
            expected_refund = float(scenario["expected_refund_inr"])

            result = run_agent(
                user_message=message,
                scenario_id=scenario_id,
                max_iterations=6,
                token_limit=6000,
                time_limit_seconds=10
            )

            actual = str(result.get("outcome", "")).upper()
            actual_refund = extract_amount(result)

            if expected == "FALLBACK":
                outcome_pass = actual.startswith("FALLBACK")
            else:
                outcome_pass = actual == expected

            refund_pass = abs(expected_refund - actual_refund) < 0.01

            if expected not in ("REFUND", "ESCALATE"):
                refund_pass = True

            scenario_pass = outcome_pass and refund_pass

            if scenario_pass:
                passed += 1

            # Required step-by-step tool execution record.
            steps = [
                {
                    "tool_name": item.get("tool"),
                    "arguments": item.get("arguments", {}),
                    "observation": item.get("observation", {})
                }
                for item in result.get("tool_history", [])
            ]

            # Required repository output fields.
            record = {
                "scenario_id": scenario_id,
                "status": actual,
                "final_answer": result.get("answer", ""),
                "llm_calls": result.get("iterations", 0),
                "tokens_used": result.get("total_tokens", 0),
                "steps": steps,

                # Additional evaluation fields.
                "expected_outcome": expected,
                "actual_outcome": actual,
                "expected_refund_inr": expected_refund,
                "actual_refund_inr": actual_refund,
                "passed": scenario_pass,
                "iterations": result.get("iterations", 0),
                "tool_executions": result.get("tool_executions", 0),
                "total_tokens": result.get("total_tokens", 0),
                "tool_history": result.get("tool_history", [])
            }

            results.append(record)
            output.write(json.dumps(record, default=str) + "\n")

            print(f"\nScenario: {scenario_id}")
            print(f"Expected: {expected} | Actual: {actual}")
            print(
                f"Expected refund: INR {expected_refund:.2f} | "
                f"Actual refund: INR {actual_refund:.2f}"
            )
            print(f"LLM calls: {record['llm_calls']}")
            print(f"Tokens used: {record['tokens_used']}")
            print(f"Tool executions: {record['tool_executions']}")
            print(f"Result: {'PASS' if scenario_pass else 'FAIL'}")

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print(f"Total scenarios: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(results) - passed}")
    print(f"Results saved to: {output_file}")
    print("\nREFUND AGENT EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
