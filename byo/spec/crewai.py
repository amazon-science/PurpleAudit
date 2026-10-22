# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Run a spec-based environment on CrewAI (``"framework": "crewai"``).

- ``Round-robin``: ``Process.sequential``, one task per agent in order; each
  agent sees the previous outputs and the last one gives the final answer.
- ``Supervisor``: ``Process.hierarchical`` with the spec's manager as
  ``manager_agent`` delegating to the other agents.
"""

import inspect
from typing import Any, Callable, Dict, Type

from . import FINAL_ANSWER, agent_id, agent_prompt, agent_tools


def build_crewai_tool(tool_name: str, description: str, fn: Callable):
    """Wrap a Python function as a CrewAI ``BaseTool`` (schema from its signature)."""
    from crewai.tools import BaseTool
    from pydantic import BaseModel, create_model

    fields: Dict[str, Any] = {}
    for pname, param in inspect.signature(fn).parameters.items():
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else str
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[pname] = (annotation, default)
    input_model: Type[BaseModel] = create_model(f"{tool_name.title().replace('_', '')}Input", **fields)
    tool_description = description or inspect.getdoc(fn) or tool_name

    class SpecTool(BaseTool):
        name: str = tool_name
        description: str = tool_description
        args_schema: Type[BaseModel] = input_model

        def _run(self, **kwargs: Any) -> str:
            return str(fn(**kwargs))

    return SpecTool()


def run(task: str, ctx) -> str:
    from crewai import Agent, Crew, Process, Task

    llm = ctx.crewai_llm()
    agents, manager = {}, None
    for cfg in ctx.spec["agents"]:
        name = agent_id(cfg)
        tools = [build_crewai_tool(t, desc, fn) for t, desc, fn in agent_tools(cfg, ctx)]
        agents[name] = Agent(
            role=name,
            goal=cfg.get("description") or "Help the team complete the user's task.",
            backstory=agent_prompt(cfg, ctx),
            tools=ctx.wrap_tools(tools, agent=name),
            llm=llm,
            allow_delegation=bool(cfg.get("is_manager")),
            verbose=False,
        )
        if cfg.get("is_manager"):
            manager = name

    final_output = f"The answer to the user's request, prefixed with '{FINAL_ANSWER}'."
    topology = ctx.spec.get("topology", "Round-robin")
    if topology == "Supervisor":
        if manager is None:
            raise ValueError("Supervisor topology requires an agent with \"is_manager\": true")
        crew = Crew(
            agents=[a for n, a in agents.items() if n != manager],
            tasks=[Task(description=task, expected_output=final_output)],
            process=Process.hierarchical,
            manager_agent=agents[manager],
            verbose=False,
        )
    elif topology == "Round-robin":
        order = list(agents.values())
        tasks = [
            Task(
                description=task,
                expected_output=final_output if i == len(order) - 1 else "Your contribution to the answer.",
                agent=agent,
            )
            for i, agent in enumerate(order)
        ]
        crew = Crew(agents=order, tasks=tasks, process=Process.sequential, verbose=False)
    else:
        raise ValueError(f"Unknown topology '{topology}'")

    return crew.kickoff().raw
