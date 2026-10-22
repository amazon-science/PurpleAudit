# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""The context PurpleAudit hands to a bring-your-own MAS for one conversation."""

import importlib.util
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional

from purpleaudit.defenses import ToolFilteringDefense
from purpleaudit.utils.defense_utils import create_defense_wrapper_from_config
from purpleaudit.utils.llm_provider import openrouter_api_key, openrouter_base_url, openrouter_model, use_openrouter

from .tools import tool_description, tool_name, wrap_tool

DISABLED_TOOL_OUTPUT = "This tool is disabled for the current task."


# Objects of these packages returned by a tool are framework control flow or status
# (LangGraph ``Command``, AG2 ``ReplyResult``, CrewAI ``ToolFailure``...), never rewritten
_FRAMEWORK_MODULES = ("langgraph.", "langchain", "autogen.", "crewai.")


def _is_tool_message(output: Any) -> bool:
    return getattr(output, "type", None) == "tool" and type(output).__module__.startswith("langchain_core.")


def _is_framework_object(output: Any) -> bool:
    return type(output).__module__.startswith(_FRAMEWORK_MODULES)


class AuditContext:
    """Passed to your entrypoint ``run(task, ctx)``, which runs your MAS on ``task``.

    MASs built with LangChain / LangGraph, CrewAI or AG2 need nothing else: PurpleAudit
    instruments these frameworks while the entrypoint runs. The attack is planted in
    the first tool output, defenses are applied to tool outputs and system prompts,
    and the conversation is recorded.

    ``ctx.chat_model()`` / ``ctx.crewai_llm()`` / ``ctx.ag2_llm_config()`` give the
    model under audit (``--model``), to run your MAS on it.

    For code outside these frameworks, route tools and system prompts through
    ``ctx.wrap_tools`` / ``ctx.system_prompt`` and report agent turns with
    ``ctx.record_message``.
    """

    def __init__(self, task: str, *, llm_config, attack=None, defense_config: Optional[dict] = None,
                 spec: Optional[dict] = None, spec_tools: Optional[dict] = None, max_rounds: int = 15):
        self.task = task
        self.llm_config = llm_config
        self.max_rounds = max_rounds
        # Only set for spec-based environments (agents described in environment.json)
        self.spec = spec or {}
        self.spec_tools = spec_tools or {}

        self.messages: List[Dict[str, str]] = []
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.injected = False
        self.tool_calls = 0
        self.system_prompts_seen = 0

        self._attack = attack
        self._defense_wrapper = create_defense_wrapper_from_config(defense_config)
        defense_config = defense_config or {}
        self.defense_prompt = defense_config.get("defense_prompt") or defense_config.get("system_prompt_addition") or ""
        self._tool_filter = next(
            (d for d in defense_config.get("defenses", []) if isinstance(d, ToolFilteringDefense)), None
        )
        self._kept_tools, self._removed_tools = set(), set()
        self._paused = threading.local()  # per thread: other agents keep being audited meanwhile
        self._lock = threading.RLock()

    # ---------- Hooks for code outside the supported frameworks ----------

    def wrap_tools(self, tools: Iterable[Any], agent: Optional[str] = None) -> List[Any]:
        """Wrap an agent's tools (functions, LangChain or CrewAI tools). Unneeded ones are
        dropped under the tool_filtering defense."""
        tools = list(tools)
        if self._tool_filter is not None:
            self._decide_tools(agent or "agent", [{"name": tool_name(t), "description": tool_description(t)} for t in tools])
            tools = [t for t in tools if not self.is_disabled(tool_name(t))]
        return [self.wrap_tool(tool, agent) for tool in tools]

    def wrap_tool(self, tool: Any, agent: Optional[str] = None) -> Any:
        name = tool_name(tool)
        self._agent(agent)["tools"].add(name)
        return wrap_tool(tool, lambda output: DISABLED_TOOL_OUTPUT if self.is_disabled(name)
                         else self.on_tool_output(name, agent, output, serialize=True))

    def system_prompt(self, agent: str, prompt: str) -> str:
        """Return ``prompt`` with the active defense appended."""
        return self.agent_prompt(agent, prompt)

    def record_message(self, agent: str, content: Any, role: str = "assistant") -> None:
        """Add an agent turn to the transcript."""
        with self._lock:
            self.messages.append({"role": role, "name": agent, "content": "" if content is None else str(content)})

    # ---------- Model under audit ----------

    @property
    def model(self) -> str:
        """Model name as the provider API expects it (OpenRouter slug or Bedrock ID)."""
        model_id = self.llm_config.model_id
        return openrouter_model(model_id) if use_openrouter() else model_id

    def chat_model(self, **kwargs):
        """LangChain chat model for the model under audit."""
        kwargs.setdefault("temperature", self.llm_config.temperature)
        if use_openrouter():
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=self.model, base_url=openrouter_base_url(), api_key=openrouter_api_key(), **kwargs)
        from langchain_aws import ChatBedrockConverse
        return ChatBedrockConverse(model=self.model, region_name=self.llm_config.region_name, **kwargs)

    def crewai_llm(self, **kwargs):
        """CrewAI ``LLM`` for the model under audit."""
        from crewai import LLM

        kwargs.setdefault("temperature", self.llm_config.temperature)
        if use_openrouter():
            # CrewAI reads OPENROUTER_BASE_URL itself
            return LLM(model=f"openrouter/{self.model}", api_key=openrouter_api_key(), **kwargs)
        # CrewAI's native Bedrock provider (1.x) takes region_name, its LiteLLM fallback aws_region_name
        native = importlib.util.find_spec("crewai.llms.providers.bedrock") is not None
        kwargs["region_name" if native else "aws_region_name"] = self.llm_config.region_name
        return LLM(model=f"bedrock/{self.model}", **kwargs)

    def ag2_llm_config(self) -> dict:
        """AG2 ``llm_config`` for the model under audit."""
        return self.llm_config.get_llm_config()

    # ---------- Called by the instrumentation (purpleaudit.byo.instrument) ----------

    def agent_prompt(self, agent: str, prompt: str, tools: Iterable[dict] = ()) -> str:
        """Record an agent's system prompt and tools; return the prompt with the active defense appended.

        Idempotent: a prompt that already carries the defense is returned as is.
        """
        defense = self.defense_prompt
        original = prompt.replace(defense, "") if defense else prompt
        tools = [t for t in tools if t.get("name")]
        with self._lock:
            self.system_prompts_seen += 1
            info = self._agent(agent)
            info["system_prompt"] = info["system_prompt"] or original
            info["tools"].update(t["name"] for t in tools)
        if self._tool_filter is not None and tools:
            self._decide_tools(agent, tools)
        if not defense or defense.strip() in prompt:
            return prompt
        return prompt + defense

    def on_tool_output(self, name: str, agent: Optional[str], output: Any, serialize: bool = False) -> Any:
        """Plant the attack in the first tool output, apply tool-level defenses, record the output.

        Outputs are only changed when an attack or defense rewrites them. Text (and the
        content of a LangChain ``ToolMessage``) can be rewritten; other data (dicts,
        lists, models...) only with ``serialize``, for calls whose output the framework
        hands to the LLM as text anyway. Framework objects are never touched, so code
        reading them keeps working.
        """
        if _is_tool_message(output):
            return output.model_copy(update={"content": self._rewrite_content(name, agent, output.content)})
        if isinstance(output, str) or (serialize and not _is_framework_object(output)):
            return self._rewrite(name, agent, output)
        return output

    def _rewrite_content(self, name: str, agent: Optional[str], content: Any) -> Any:
        if isinstance(content, str):
            return self._rewrite(name, agent, content)
        # Content blocks (e.g. from MCP tools): rewrite the first text block, keep the others
        blocks = list(content or [])
        for i, block in enumerate(blocks):
            if isinstance(block, dict) and block.get("type") == "text":
                blocks[i] = dict(block, text=self._rewrite(name, agent, block.get("text", "")))
                return blocks
        return content

    def _rewrite(self, name: str, agent: Optional[str], output: Any) -> Any:
        with self._lock:
            self.tool_calls += 1
            text, rewritten = str(output), False
            if self._attack is not None and not self.injected:
                self.injected, text, rewritten = True, self._attack(text), True
            if self._defense_wrapper is not None:
                text, rewritten = self._defense_wrapper(text), True
            self.messages.append({"role": "tool", "name": f"{agent}/{name}" if agent else name, "content": text})
        return text if rewritten else output

    def is_disabled(self, tool: str) -> bool:
        """Whether the tool_filtering defense removed this tool from every agent that has it."""
        return tool in self._removed_tools and tool not in self._kept_tools

    @property
    def is_paused(self) -> bool:
        return getattr(self._paused, "depth", 0) > 0

    @contextmanager
    def paused(self):
        """Stop reporting to this context while PurpleAudit itself calls an LLM framework (the tool filter)."""
        self._paused.depth = getattr(self._paused, "depth", 0) + 1
        try:
            yield
        finally:
            self._paused.depth -= 1

    # ---------- Internals ----------

    def _agent(self, agent: Optional[str]) -> Dict[str, Any]:
        return self.agents.setdefault(agent or "agent", {"system_prompt": "", "tools": set()})

    def _decide_tools(self, agent: str, tools: List[dict]) -> None:
        names = {t["name"] for t in tools}
        if names <= self._kept_tools | self._removed_tools:
            return
        with self.paused():
            kept = self._tool_filter.filter_tools(
                agent_role=agent, available_tools=tools,
                system_prompt=self.agents.get(agent, {}).get("system_prompt", ""), user_task=self.task,
            )
        kept_names = {t["name"] if isinstance(t, dict) else str(t) for t in kept}
        with self._lock:
            self._kept_tools |= names & kept_names
            self._removed_tools |= names - kept_names
