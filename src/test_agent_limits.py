
import pandas as pd

from refund_agent import run_agent, SimulatedCrash


def main():
    scenarios = pd.read_csv("data/raw/agent_scenarios.csv")

    s1 = scenarios[scenarios["scenario_id"] == "S1"].iloc[0]
    s7 = scenarios[scenarios["scenario_id"] == "S7"].iloc[0]

    message_s1 = str(s1["message"])
    message_s7 = str(s7["message"])

    print("=" * 60)
    print("AGENT LIMIT AND RECOVERY TESTS")
    print("=" * 60)

    # TEST 1: TOKEN LIMIT
    print("\nTEST 1: TOKEN LIMIT")

    token_result = run_agent(
        user_message=message_s1,
        scenario_id="LIMIT_TOKEN_S1",
        max_iterations=6,
        token_limit=200,
        time_limit_seconds=10
    )

    print("Expected: FALLBACK_TOKEN_LIMIT")
    print("Actual:", token_result.get("outcome"))
    print("Result:", "PASS" if
          token_result.get("outcome") == "FALLBACK_TOKEN_LIMIT"
          else "FAIL")

    # TEST 2: TIMEOUT
    print("\nTEST 2: TIMEOUT")

    timeout_result = run_agent(
        user_message=message_s1,
        scenario_id="LIMIT_TIMEOUT_S1",
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=0
    )

    print("Expected: FALLBACK_TIMEOUT")
    print("Actual:", timeout_result.get("outcome"))
    print("Result:", "PASS" if
          timeout_result.get("outcome") == "FALLBACK_TIMEOUT"
          else "FAIL")

    # TEST 3: MAXIMUM ITERATIONS
    print("\nTEST 3: MAXIMUM ITERATIONS")

    loop_result = run_agent(
        user_message=message_s7,
        scenario_id="LIMIT_LOOP_S7",
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=10
    )

    print("Expected: FALLBACK_MAX_ITERATIONS")
    print("Actual:", loop_result.get("outcome"))
    print("Result:", "PASS" if
          loop_result.get("outcome") == "FALLBACK_MAX_ITERATIONS"
          else "FAIL")

    # TEST 4: CRASH AND RESUME
    print("\nTEST 4: CRASH AND CHECKPOINT RECOVERY")

    crash_id = "CRASH_RECOVERY_S1"

    try:
        run_agent(
            user_message=message_s1,
            scenario_id=crash_id,
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10,
            crash_after_tools=2
        )
    except SimulatedCrash:
        print("Simulated crash after 2 tool executions.")

    recovered = run_agent(
        user_message=message_s1,
        scenario_id=crash_id,
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=10,
        resume=True
    )

    history = recovered.get("tool_history", [])
    tool_names = [item.get("tool") for item in history]

    expected_tools = [
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund"
    ]

    recovery_pass = (
        recovered.get("outcome") == "REFUND"
        and len(history) == 4
        and tool_names == expected_tools
    )

    print("Expected outcome: REFUND")
    print("Actual outcome:", recovered.get("outcome"))
    print("Tool executions:", len(history))
    print("Tool order:", tool_names)
    print("Expected refund: INR 128.25")
    print("Actual answer:", recovered.get("answer"))
    print("Result:", "PASS" if recovery_pass else "FAIL")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
