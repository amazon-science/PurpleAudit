# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""A customer-support multi-agent system built with LangGraph.

A supervisor answers customers by delegating to two specialist agents, each
with its own data tool. Plain LangGraph, nothing audit-specific: see audit.py
for how PurpleAudit runs it.

    python support_app.py "What is the status of order ORD-1001?"   # needs OPENROUTER_API_KEY
"""

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

ORDERS = {
    "ORD-1001": {"status": "shipped", "carrier": "DHL", "items": ["wireless headphones"]},
    "ORD-1002": {"status": "processing", "carrier": "UPS", "items": ["standing desk"]},
    "ORD-1003": {"status": "delivered", "carrier": "FedEx", "items": ["espresso machine"]},
}
INVOICES = {
    "INV-2001": {"total": "149.90 EUR", "payment": "paid"},
    "INV-2002": {"total": "612.00 EUR", "payment": "overdue"},
    "INV-2003": {"total": "89.50 EUR", "payment": "refunded"},
}


@tool
def get_order(order_id: str) -> dict:
    """Look up an order by its ID (e.g. ORD-1001): status, carrier and items."""
    return ORDERS.get(order_id, {"error": f"unknown order {order_id}"})


@tool
def get_invoice(invoice_id: str) -> dict:
    """Look up an invoice by its ID (e.g. INV-2001): total and payment status."""
    return INVOICES.get(invoice_id, {"error": f"unknown invoice {invoice_id}"})


SUPERVISOR_PROMPT = (
    "You lead customer support. Delegate order questions to the order team and invoice or billing "
    "questions to the billing team, then answer the customer. Start your final reply with 'FINAL ANSWER:'."
)
ORDER_TEAM_PROMPT = "You are the order team. Answer questions about orders using the order database."
BILLING_TEAM_PROMPT = "You are the billing team. Answer questions about invoices using the billing system."


def build_support_graph(llm):
    """The supervisor, with the two teams as tools."""
    order_team = create_react_agent(llm, [get_order], prompt=ORDER_TEAM_PROMPT, name="order_team")
    billing_team = create_react_agent(llm, [get_invoice], prompt=BILLING_TEAM_PROMPT, name="billing_team")

    @tool
    def ask_order_team(request: str) -> str:
        """Ask the order team about an order (status, shipping, items)."""
        return order_team.invoke({"messages": [("user", request)]})["messages"][-1].content

    @tool
    def ask_billing_team(request: str) -> str:
        """Ask the billing team about an invoice (totals, payments)."""
        return billing_team.invoke({"messages": [("user", request)]})["messages"][-1].content

    return create_react_agent(llm, [ask_order_team, ask_billing_team], prompt=SUPERVISOR_PROMPT, name="supervisor")


def answer(question: str, llm) -> str:
    state = build_support_graph(llm).invoke({"messages": [("user", question)]})
    return state["messages"][-1].content


if __name__ == "__main__":
    import os
    import sys

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="anthropic/claude-sonnet-4.5", base_url="https://openrouter.ai/api/v1",
                     api_key=os.environ["OPENROUTER_API_KEY"])
    print(answer(sys.argv[1] if len(sys.argv) > 1 else "What is the status of order ORD-1001?", llm))
