# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""An IT-helpdesk multi-agent system built with AG2.

A ticket agent and an asset agent take turns in a group chat until one of them
answers the employee. Plain AG2, nothing audit-specific: see audit.py for how
PurpleAudit runs it.

    python helpdesk_app.py "What is the priority of ticket TCK-501?"   # needs OPENROUTER_API_KEY
"""

from autogen import ConversableAgent
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import RoundRobinPattern

TICKETS = {
    "TCK-501": {"title": "VPN drops every hour", "priority": "critical", "assignee": "Priya Nair"},
    "TCK-502": {"title": "Printer jam on floor 3", "priority": "low", "assignee": "Marco Rossi"},
}
ASSETS = {
    "AST-601": {"model": "ThinkPad X1 Carbon", "location": "Berlin office", "owner": "Lena Vogel"},
    "AST-602": {"model": "MacBook Pro 14", "location": "Lyon office", "owner": "Hugo Martin"},
}


def get_ticket(ticket_id: str) -> dict:
    """Look up a support ticket by its ID (e.g. TCK-501): title, priority and assignee."""
    return TICKETS.get(ticket_id, {"error": f"unknown ticket {ticket_id}"})


def get_asset(asset_id: str) -> dict:
    """Look up an IT asset by its ID (e.g. AST-601): model, location and owner."""
    return ASSETS.get(asset_id, {"error": f"unknown asset {asset_id}"})


TICKET_PROMPT = (
    "You are the helpdesk ticket agent. Answer questions about support tickets with get_ticket. "
    "If the question is not about a ticket, say so and let your colleague answer. "
    "When you have the answer, reply with 'FINAL ANSWER: <answer>'."
)
ASSET_PROMPT = (
    "You are the helpdesk asset agent. Answer questions about IT assets with get_asset. "
    "If the question is not about an asset, say so and let your colleague answer. "
    "When you have the answer, reply with 'FINAL ANSWER: <answer>'."
)


def run_helpdesk(question: str, llm_config):
    """Answer an employee's question; returns AG2's chat result."""
    agents = []
    for name, prompt, tool in (("ticket_agent", TICKET_PROMPT, get_ticket), ("asset_agent", ASSET_PROMPT, get_asset)):
        agent = ConversableAgent(name, system_message=prompt, llm_config=llm_config)
        agent.register_for_llm(description=tool.__doc__)(tool)
        agent.register_for_execution()(tool)
        agents.append(agent)

    user = ConversableAgent("employee", llm_config=False, human_input_mode="NEVER")
    pattern = RoundRobinPattern(
        initial_agent=agents[0], agents=agents, user_agent=user,
        group_manager_args={
            "llm_config": llm_config,
            "is_termination_msg": lambda m: "FINAL ANSWER:" in (m.get("content") or ""),
        },
    )
    result, _, _ = initiate_group_chat(pattern=pattern, messages=question, max_rounds=12)
    return result


if __name__ == "__main__":
    import os
    import sys

    llm_config = {"config_list": [{"model": "anthropic/claude-sonnet-4.5", "api_type": "openai",
                                   "base_url": "https://openrouter.ai/api/v1", "api_key": os.environ["OPENROUTER_API_KEY"]}]}
    result = run_helpdesk(sys.argv[1] if len(sys.argv) > 1 else "What is the priority of ticket TCK-501?", llm_config)
    print(result.chat_history[-1]["content"])
