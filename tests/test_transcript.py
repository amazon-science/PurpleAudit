# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Whatever a bring-your-own MAS returns becomes a transcript the lab can score."""

from types import SimpleNamespace

import pytest

from purpleaudit.byo import AuditContext
from purpleaudit.byo.transcript import build_transcript

LLM = SimpleNamespace(model_id="m", temperature=0.0, region_name="us-west-2")


def status(order_id: str) -> str:
    """Order status."""
    return "shipped"


def test_plain_string_result_uses_recorded_tool_outputs():
    ctx = AuditContext("Status of ORD-1?", llm_config=LLM)
    (tool,) = ctx.wrap_tools([status], agent="clerk")
    tool("ORD-1")
    final, messages = build_transcript("FINAL ANSWER: shipped", ctx)
    assert final == "FINAL ANSWER: shipped"
    assert [(m["role"], m["name"]) for m in messages] == [("user", "user"), ("tool", "clerk/status"), ("assistant", "final_answer")]


def test_recorded_messages_keep_their_order():
    ctx = AuditContext("Status of ORD-1?", llm_config=LLM)
    (tool,) = ctx.wrap_tools([status], agent="clerk")
    ctx.record_message("planner", "Asking the clerk")
    tool("ORD-1")
    ctx.record_message("clerk", "FINAL ANSWER: shipped")
    final, messages = build_transcript(None, ctx)
    assert final == "FINAL ANSWER: shipped"
    assert [m["name"] for m in messages] == ["user", "planner", "clerk/status", "clerk"]


def test_ag2_chat_result():
    ctx = AuditContext("Status of ORD-1?", llm_config=LLM)
    (tool,) = ctx.wrap_tools([status], agent="clerk")
    tool("ORD-1")
    chat = SimpleNamespace(chat_history=[
        {"role": "assistant", "name": "user", "content": "Status of ORD-1?"},
        {"role": "assistant", "name": "clerk", "content": "None",
         "tool_calls": [{"id": "c", "type": "function", "function": {"name": "status", "arguments": "{\"order_id\": \"ORD-1\"}"}}]},
        {"role": "tool", "name": "_Group_Tool_Executor", "content": "shipped"},
        {"role": "user", "name": "clerk", "content": "FINAL ANSWER: shipped"},
    ])
    final, messages = build_transcript((chat, {}, None), ctx)  # initiate_group_chat returns a tuple
    assert final == "FINAL ANSWER: shipped"
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "user"]  # task not duplicated, tool output not re-added
    assert messages[1]["content"] == '[calls status({"order_id": "ORD-1"})]'


def test_crewai_output():
    ctx = AuditContext("Gate of FLT-1?", llm_config=LLM)
    output = SimpleNamespace(raw="FINAL ANSWER: B12", tasks_output=[
        SimpleNamespace(agent="Flight specialist", raw="Gate B12"),
        SimpleNamespace(agent="Concierge", raw="FINAL ANSWER: B12"),
    ])
    final, messages = build_transcript(output, ctx)
    assert final == "FINAL ANSWER: B12"
    assert [m["name"] for m in messages] == ["user", "Flight specialist", "Concierge"]


def test_langgraph_state():
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    ctx = AuditContext("Status of ORD-1?", llm_config=LLM)
    state = {"messages": [
        HumanMessage("Status of ORD-1?"),
        AIMessage("", name="clerk", tool_calls=[{"name": "status", "args": {"order_id": "ORD-1"}, "id": "c1"}]),
        ToolMessage("shipped", name="status", tool_call_id="c1"),
        AIMessage("FINAL ANSWER: shipped", name="clerk"),
    ]}
    final, messages = build_transcript(state, ctx)
    assert final == "FINAL ANSWER: shipped"
    assert [(m["role"], m["name"]) for m in messages] == [("user", "user"), ("assistant", "clerk"), ("tool", "status"), ("assistant", "clerk")]


def test_ag2_joined_tool_outputs_are_not_duplicated():
    ctx = AuditContext("Status of ORD-1 and ORD-2?", llm_config=LLM)
    lookup = ctx.wrap_tools([lambda order_id: {"id": order_id, "paid": None}])[0]
    lookup("ORD-1"), lookup("ORD-2")
    chat = SimpleNamespace(chat_history=[
        {"role": "tool", "name": "_Group_Tool_Executor",
         "content": '{"id": "ORD-1", "paid": null}\n\n{"id": "ORD-2", "paid": null}'},
        {"role": "user", "name": "clerk", "content": "FINAL ANSWER: both unpaid"},
    ])
    _, messages = build_transcript(chat, ctx)
    assert [m["role"] for m in messages] == ["user", "tool", "user"]
