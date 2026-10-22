# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Run a spec-based environment on LangGraph (``"framework": "langgraph"``).

- ``Round-robin``: agents take turns on a shared conversation until one answers
  with ``FINAL ANSWER:`` or ``max_rounds`` turns have passed.
- ``Supervisor``: the manager delegates through ``ask_<agent>`` tools and answers
  once it has what it needs.

Every agent is a LangGraph ReAct agent built through the ``AuditContext`` hooks.
"""

import inspect
from typing import Annotated, Any, Dict, List, TypedDict

from . import agent_id, agent_prompt, agent_tools, is_final


def _text(message) -> str:
    content = message.content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content or ""


def _last_ai_text(messages: List[Any]) -> str:
    for message in reversed(messages):
        if message.type == "ai" and _text(message):
            return _text(message)
    return ""


def _build_agents(ctx, model) -> Dict[str, Dict[str, Any]]:
    from langchain_core.tools import StructuredTool

    agents = {}
    for cfg in ctx.spec["agents"]:
        name = agent_id(cfg)
        prompt = agent_prompt(cfg, ctx)
        tools = [
            StructuredTool.from_function(func=fn, name=tool, description=desc or inspect.getdoc(fn) or tool)
            for tool, desc, fn in agent_tools(cfg, ctx)
        ]
        agents[name] = {"cfg": cfg, "prompt": prompt, "tools": ctx.wrap_tools(tools, agent=name)}
    return agents


def _view_for(name: str, messages: List[Any]) -> List[Any]:
    """What agent ``name`` sees: the task and the others' answers as user turns (no foreign tool calls)."""
    from langchain_core.messages import AIMessage, HumanMessage

    view = []
    for m in messages:
        if m.type == "human":
            view.append(HumanMessage(content=_text(m)))
        elif m.type == "ai" and _text(m) and not getattr(m, "tool_calls", None):
            if m.name == name:
                view.append(AIMessage(content=_text(m)))
            else:
                view.append(HumanMessage(content=f"[{m.name}]: {_text(m)}"))
    if view and view[-1].type == "ai":
        view.append(HumanMessage(content="Continue."))
    return view


def _round_robin(task: str, ctx, model, agents) -> str:
    from langchain_core.messages import HumanMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import create_react_agent

    class State(TypedDict):
        messages: Annotated[list, add_messages]
        turns: int

    order = list(agents)
    runners = {n: create_react_agent(model, a["tools"], prompt=a["prompt"], name=n) for n, a in agents.items()}

    def make_node(name):
        def node(state):
            view = _view_for(name, state["messages"])
            result = runners[name].invoke({"messages": view})
            return {"messages": result["messages"][len(view):], "turns": state["turns"] + 1}
        return node

    def next_speaker(state):
        if is_final(_last_ai_text(state["messages"][-1:])) or state["turns"] >= ctx.max_rounds:
            return END
        return order[state["turns"] % len(order)]

    graph = StateGraph(State)
    for name in order:
        graph.add_node(name, make_node(name))
        graph.add_conditional_edges(name, next_speaker, order + [END])
    graph.add_edge(START, order[0])

    state = graph.compile().invoke(
        {"messages": [HumanMessage(content=task)], "turns": 0},
        {"recursion_limit": 4 * ctx.max_rounds + 10},
    )
    return _last_ai_text(state["messages"])


def _supervisor(task: str, ctx, model, agents) -> str:
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import StructuredTool
    from langgraph.prebuilt import create_react_agent

    managers = [n for n, a in agents.items() if a["cfg"].get("is_manager")]
    if not managers:
        raise ValueError("Supervisor topology requires an agent with \"is_manager\": true")
    manager = managers[0]

    def ask_tool(worker: str):
        runner = create_react_agent(model, agents[worker]["tools"], prompt=agents[worker]["prompt"], name=worker)

        def ask(request: str) -> str:
            state = runner.invoke({"messages": [HumanMessage(content=request)]},
                                  {"recursion_limit": 2 * ctx.max_rounds + 5})
            return _last_ai_text(state["messages"])

        description = agents[worker]["cfg"].get("description", "")
        return StructuredTool.from_function(
            func=ask, name=f"ask_{worker}",
            description=f"Delegate a request to {worker}. {description}".strip(),
        )

    tools = agents[manager]["tools"] + [ask_tool(w) for w in agents if w != manager]
    runner = create_react_agent(model, tools, prompt=agents[manager]["prompt"], name=manager)
    state = runner.invoke({"messages": [HumanMessage(content=task)]}, {"recursion_limit": 2 * ctx.max_rounds + 5})
    return _last_ai_text(state["messages"])


def run(task: str, ctx) -> str:
    model = ctx.chat_model()
    agents = _build_agents(ctx, model)
    topology = ctx.spec.get("topology", "Round-robin")
    if topology == "Round-robin":
        return _round_robin(task, ctx, model, agents)
    if topology == "Supervisor":
        return _supervisor(task, ctx, model, agents)
    raise ValueError(f"Unknown topology '{topology}'")
