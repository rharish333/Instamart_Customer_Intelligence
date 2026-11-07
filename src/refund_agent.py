
"""ReAct refund agent with limits and checkpoint/resume support."""

import json
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mock_llm
from refund_tools import TOOLS_SPEC, TOOL_MAP


STATE_DIR = ROOT / "state"

SYSTEM_PROMPT = """
You are the INSTAMART REFUND AGENT.

Follow the refund policy and use the available tools.
Always retrieve the order before deciding.
Search the policy before calculating a refund.
Only completed orders are eligible.
A delivery taking more than 45 minutes qualifies for a full refund.
A refund up to INR 1000 can be issued automatically.
A refund above INR 1000 must be escalated to a human.
Never claim a refund was issued unless the issue_refund tool succeeds.
Never claim an escalation was sent unless the escalation tool succeeds.
If an order does not exist, report NOT_FOUND.
If the order is cancelled or not eligible, report DENIED.
Be concise and use the tools in the correct order.
"""


class SimulatedCrash(Exception):
    """Used to test checkpoint recovery."""


def checkpoint_path(scenario_id):
    safe_id = "".join(
        character for character in str(scenario_id)
        if character.isalnum() or character in ("-", "_")
    )
    return STATE_DIR / f"{safe_id}.json"


def save_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, default=str)


def load_checkpoint(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_agent(
    user_message,
    scenario_id="manual",
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None,
    resume=False
):
    """Run the agent or continue from an existing checkpoint."""

    path = checkpoint_path(scenario_id)

    if resume:
        if not path.exists():
            return {
                "scenario_id": scenario_id,
                "outcome": "FALLBACK_NO_CHECKPOINT",
                "answer": "No checkpoint is available to resume."
            }

        state = load_checkpoint(path)

        if state.get("completed"):
            return state["result"]

    else:
        state = {
            "scenario_id": scenario_id,
            "user_message": user_message,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "iterations": 0,
            "tool_executions": 0,
            "total_tokens": 0,
            "tool_history": [],
            "completed": False
        }

        save_checkpoint(path, state)

    start_time = time.monotonic()

    def finish(outcome, answer):
        result = {
            "scenario_id": scenario_id,
            "outcome": outcome,
            "answer": answer,
            "iterations": state["iterations"],
            "tool_executions": state["tool_executions"],
            "total_tokens": state["total_tokens"],
            "tool_history": state["tool_history"]
        }

        state["completed"] = True
        state["result"] = result
        save_checkpoint(path, state)

        return result

    while True:

        if time.monotonic() - start_time >= time_limit_seconds:
            return finish(
                "FALLBACK_TIMEOUT",
                "The agent reached its execution time limit."
            )

        if state["iterations"] >= max_iterations:
            return finish(
                "FALLBACK_MAX_ITERATIONS",
                "The agent reached its maximum iteration limit."
            )

        try:
            response = mock_llm.chat(
                state["messages"],
                tools=TOOLS_SPEC
            )
        except Exception as error:
            return finish(
                "FALLBACK_ERROR",
                f"The agent could not complete the request: {error}"
            )

        usage = response.usage.total_tokens
        state["total_tokens"] += usage
        state["iterations"] += 1

        if state["total_tokens"] > token_limit:
            return finish(
                "FALLBACK_TOKEN_LIMIT",
                "The agent reached its token budget."
            )

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            answer = message.content or "The agent did not return an answer."

            if answer.startswith("REFUND:"):
                outcome = "REFUND"
            elif answer.startswith("ESCALATE:"):
                outcome = "ESCALATE"
            elif answer.startswith("DENIED:"):
                outcome = "DENIED"
            elif answer.startswith("NOT_FOUND:"):
                outcome = "NOT_FOUND"
            else:
                outcome = "COMPLETED"

            return finish(outcome, answer)

        assistant_calls = []

        for call in tool_calls:
            arguments = call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            assistant_calls.append({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(arguments)
                }
            })

        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_calls
        })

        for call in tool_calls:
            name = call.name
            arguments = call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            function = TOOL_MAP.get(name)

            if function is None:
                observation = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    observation = function(**arguments)
                except Exception as error:
                    observation = {"error": str(error)}

            observation_text = json.dumps(
                observation,
                default=str
            )

            state["messages"].append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": observation_text
            })

            state["tool_executions"] += 1

            state["tool_history"].append({
                "tool": name,
                "arguments": arguments,
                "observation": observation
            })

            # Save after every tool observation.
            save_checkpoint(path, state)

            if (
                crash_after_tools is not None
                and state["tool_executions"] == crash_after_tools
            ):
                raise SimulatedCrash(
                    f"Simulated crash after {crash_after_tools} tools."
                )


def resume_agent(run_id):
    """Resume an agent using its saved checkpoint."""

    path = checkpoint_path(run_id)

    if not path.exists():
        return {
            "scenario_id": run_id,
            "outcome": "FALLBACK_NO_CHECKPOINT",
            "answer": "No checkpoint is available to resume."
        }

    state = load_checkpoint(path)

    if state.get("completed"):
        return state["result"]

    return run_agent(
        user_message=state["user_message"],
        scenario_id=run_id,
        resume=True
    )


def simulate_crash_and_resume(
    user_message,
    scenario_id,
    crash_after_tools=2
):
    """Simulate a crash and resume from the saved checkpoint."""

    try:
        run_agent(
            user_message=user_message,
            scenario_id=scenario_id,
            crash_after_tools=crash_after_tools
        )
    except SimulatedCrash as error:
        print(error)
        print("Checkpoint saved. Resuming agent...")

    return resume_agent(scenario_id)


if __name__ == "__main__":
    print("REACT REFUND AGENT TEST")

    test_message = (
        "My order ORD-100048 reached me very late. "
        "I would like a refund."
    )

    result = run_agent(
        user_message=test_message,
        scenario_id="TEST_S1"
    )

    print(json.dumps(result, indent=2))
    print("\nREACT AGENT TEST COMPLETED")
