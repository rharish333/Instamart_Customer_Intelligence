"""Tools used by the Instamart refund agent."""

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from policy_retriever import PolicyRetriever


# Project paths
ROOT = Path(__file__).resolve().parent.parent
ORDERS_FILE = ROOT / "data" / "clean" / "orders.csv"
LEDGER_FILE = ROOT / "outputs" / "refund_ledger.jsonl"


# Load the order data and policy retriever
orders = pd.read_csv(ORDERS_FILE)
retriever = PolicyRetriever()


def get_order(order_id):
    """Find and return an order using its order ID."""

    matches = orders[orders["order_id"].astype(str) == str(order_id)]

    if matches.empty:
        return {
            "error": "Order not found",
            "order_id": order_id
        }

    order = matches.iloc[0].to_dict()

    # Convert pandas missing values into JSON-safe values.
    for key, value in order.items():
        if pd.isna(value):
            order[key] = None
        elif hasattr(value, "item"):
            order[key] = value.item()

    return order


def search_policy(query):
    """Retrieve the three most relevant policy chunks."""

    results = retriever.retrieve(query, k=3)

    return {
        "query": query,
        "results": results
    }


def calculate_refund(order_amount, delivery_minutes):
    """Calculate the eligible late-delivery refund."""

    amount = float(order_amount)
    minutes = float(delivery_minutes)

    if minutes > 45:
        refund = round(amount, 2)
        reason = "Delivery exceeded 45 minutes."
    else:
        refund = 0.0
        reason = "Delivery was within 45 minutes."

    return {
        "refund_inr": refund,
        "reason": reason
    }


def _write_ledger(entry):
    """Append one record to the refund ledger."""

    LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)

    entry["timestamp"] = datetime.now(timezone.utc).isoformat()

    with open(LEDGER_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")


def issue_refund(order_id, refund_inr):
    """Record an approved refund in the ledger."""

    amount = round(float(refund_inr), 2)

    if amount <= 0:
        return {
            "success": False,
            "error": "Refund amount must be greater than zero."
        }

    if amount > 1000:
        return {
            "success": False,
            "error": "Refunds above INR 1000 require human approval."
        }

    order = get_order(order_id)

    if "error" in order:
        return {
            "success": False,
            "error": "Order not found."
        }

    if order["status"] != "completed":
        return {
            "success": False,
            "error": "Only completed orders can be refunded."
        }

    if round(float(order["order_amount"]), 2) != amount:
        return {
            "success": False,
            "error": "Refund amount does not match the order amount."
        }

    entry = {
        "action": "refund_issued",
        "order_id": order_id,
        "refund_inr": amount,
        "status": "approved"
    }

    _write_ledger(entry)

    return {
        "success": True,
        "order_id": order_id,
        "refund_inr": amount,
        "message": "Refund recorded successfully."
    }


def escalate_to_human(order_id, refund_inr):
    """Record a refund request that needs human approval."""

    amount = round(float(refund_inr), 2)

    if amount <= 1000:
        return {
            "success": False,
            "error": "Human escalation is required only for refunds above INR 1000."
        }

    order = get_order(order_id)

    if "error" in order:
        return {
            "success": False,
            "error": "Order not found."
        }

    if order["status"] != "completed":
        return {
            "success": False,
            "error": "Only completed orders can be escalated."
        }

    if round(float(order["order_amount"]), 2) != amount:
        return {
            "success": False,
            "error": "Refund amount does not match the order amount."
        }

    entry = {
        "action": "human_escalation",
        "order_id": order_id,
        "refund_inr": amount,
        "status": "awaiting_human_approval"
    }

    _write_ledger(entry)

    return {
        "success": True,
        "order_id": order_id,
        "refund_inr": amount,
        "message": "Refund request sent for human approval."
    }


# Tool descriptions supplied to the agent
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Retrieve an order using its order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search the available refund policies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate the refund for a late delivery.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": "number"}
                },
                "required": ["order_amount", "delivery_minutes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Record an automatically approved refund of INR 1000 or less.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Send a refund above INR 1000 for human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    }
]


# Connect each tool name to its Python function
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


if __name__ == "__main__":
    print("REFUND TOOLS TEST")

    print("\n1. Get order:")
    print(get_order("ORD-100048"))

    print("\n2. Search policy:")
    print(search_policy("refund policy late delivery"))

    print("\n3. Calculate refund:")
    print(calculate_refund(128.25, 65))

    print("\n4. Test automatic refund:")
    print(issue_refund("ORD-100048", 128.25))

    print("\n5. Test human escalation:")
    print(escalate_to_human("ORD-100145", 1999))

    print("\nREFUND TOOLS TEST COMPLETED")
