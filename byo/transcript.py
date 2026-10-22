# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Turn whatever a bring-your-own MAS returns into PurpleAudit's transcript format.

A transcript is a list of ``{"role", "name", "content"}`` dicts. Tool outputs
have ``role == "tool"``; the lab uses them to find where attacks land.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

Message = Dict[str, str]


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, list):  # LangChain / Anthropic content blocks
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def _render_tool_calls(text: str, calls: List[Any]) -> str:
    rendered = []
    for call in calls:
        if isinstance(call, dict) and "function" in call:  # OpenAI / AG2 format
            name, args = call["function"].get("name"), call["function"].get("arguments", "")
        else:  # LangChain format
            name, args = call.get("name"), json.dumps(call.get("args", {}), ensure_ascii=False)
        rendered.append(f"[calls {name}({args})]")
    return "\n".join(filter(None, [text, "; ".join(rendered)]))


def langchain_message_to_dict(message: Any, default_name: str = "assistant") -> Optional[Message]:
    kind = getattr(message, "type", "")
    name = getattr(message, "name", None)
    content = _content_text(getattr(message, "content", ""))
    if kind in ("system", "remove"):
        return None
    if kind == "human":
        return {"role": "user", "name": name or "user", "content": content}
    if kind == "tool":
        return {"role": "tool", "name": name or "tool", "content": content}
    calls = getattr(message, "tool_calls", None) or []
    return {"role": "assistant", "name": name or default_name, "content": _render_tool_calls(content, calls)}


def _dict_message(message: Dict[str, Any]) -> Message:
    content = _content_text(message.get("content"))
    if message.get("tool_calls"):
        # AG2 stores the content of tool-call-only messages as the string "None"
        content = _render_tool_calls("" if content == "None" else content, message["tool_calls"])
    return {
        "role": message.get("role", "assistant"),
        "name": message.get("name") or message.get("role", "assistant"),
        "content": content,
    }


def _unwrap(result: Any) -> Any:
    # AG2's initiate_group_chat returns (ChatResult, context_variables, last_agent)
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _is_crew_output(result: Any) -> bool:
    return hasattr(result, "raw") and hasattr(result, "tasks_output")


def native_messages(result: Any) -> List[Message]:
    """Messages from a framework result (LangGraph state, CrewOutput, AG2 ChatResult, dict)."""
    result = _unwrap(result)
    if isinstance(result, dict) and isinstance(result.get("messages"), list):
        messages = []
        for m in result["messages"]:
            entry = _dict_message(m) if isinstance(m, dict) else langchain_message_to_dict(m)
            if entry is not None:
                messages.append(entry)
        return messages
    if _is_crew_output(result):
        return [{"role": "assistant", "name": str(t.agent), "content": _content_text(t.raw)} for t in result.tasks_output]
    if hasattr(result, "chat_history"):
        return [_dict_message(m) for m in result.chat_history]
    return []


def final_answer(result: Any, messages: List[Message]) -> str:
    result = _unwrap(result)
    if isinstance(result, str):
        return result
    if isinstance(result, dict) and "final_result" in result:
        return _content_text(result["final_result"])
    if _is_crew_output(result):
        return _content_text(result.raw)
    for message in reversed(messages):
        if message["role"] != "tool" and message["content"] and not message["content"].startswith("[calls "):
            return message["content"]
    return ""


def _loose(text: str) -> str:
    # Frameworks serialize tool outputs differently (str(dict) vs JSON): compare loosely
    return re.sub(r"[\s'\"]", "", text).lower().replace("none", "null")


def build_transcript(result: Any, ctx) -> Tuple[str, List[Message]]:
    """Return ``(final_answer, transcript)`` for one conversation.

    Messages recorded live (by the framework instrumentation, ``record_message``
    and the tool hooks) are preferred because they are in order. Otherwise the
    framework result is converted, and tool outputs the framework did not expose
    (e.g. calls made inside nested agents) are added.
    """
    recorded_agent_messages = any(m["role"] != "tool" for m in ctx.messages)
    if recorded_agent_messages:
        messages = list(ctx.messages)
    else:
        messages = native_messages(result)
        if messages and messages[0]["content"] == ctx.task:
            messages = messages[1:]  # the task, prepended below
        # A framework tool message may join several outputs (e.g. AG2's parallel tool calls)
        native_tool_outputs = [_loose(m["content"]) for m in messages if m["role"] == "tool"]
        missing = [m for m in ctx.messages if m["role"] == "tool"
                   and not any(_loose(m["content"]) in output for output in native_tool_outputs)]
        messages = missing + messages

    final = final_answer(result, messages)
    if final and not any(m["content"] == final for m in messages[-1:]):
        messages.append({"role": "assistant", "name": "final_answer", "content": final})
    return final, [{"role": "user", "name": "user", "content": ctx.task}] + messages
