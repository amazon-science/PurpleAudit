# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Built-in entrypoints that run spec-based environments on other frameworks.

A spec-based environment describes its agents in ``environment.json`` (roles,
system prompts, tools from ``tools.py``, topology). ``"framework": "ag2"`` runs
it with the original AG2 implementation; ``"langgraph"`` and ``"crewai"`` use
these entrypoints, which build the MAS through the same ``AuditContext`` hooks a
bring-your-own MAS uses.
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

from purpleaudit.utils.logger import log_warning

FINAL_ANSWER = "FINAL ANSWER:"


def agent_id(agent_cfg: Dict[str, Any]) -> str:
    """Same normalization of an agent role as the AG2 MAS."""
    return agent_cfg["role"].replace(" ", "_").lower().replace("(", "").replace(")", "")


def agent_prompt(agent_cfg: Dict[str, Any], ctx) -> str:
    # Same order as the AG2 MAS: defense, then current date
    prompt = ctx.system_prompt(agent_id(agent_cfg), agent_cfg["system_prompt"])
    return prompt + f"\n\nCURRENT DATE: {datetime.now():%B %d, %Y}"


def agent_tools(agent_cfg: Dict[str, Any], ctx) -> List[Tuple[str, str, Callable]]:
    """``(name, description, function)`` of an agent's tools found in ``tools.py``."""
    tools = []
    for entry in agent_cfg.get("tools", []):
        name = entry.get("name") if isinstance(entry, dict) else str(entry)
        description = entry.get("description", "") if isinstance(entry, dict) else ""
        if name not in ctx.spec_tools:
            log_warning(f"Tool '{name}' of agent '{agent_cfg['role']}' is missing from tools.py, skipping")
            continue
        tools.append((name, description, ctx.spec_tools[name]))
    return tools


def is_final(text: str) -> bool:
    return FINAL_ANSWER in (text or "").upper()
