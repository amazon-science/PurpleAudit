# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Adapter that audits a bring-your-own MAS through its entrypoint."""

import asyncio
import importlib
import importlib.util
import inspect
import json
import os
import re
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

from purpleaudit.core.adapters.base import BaseMASAdapter
from purpleaudit.defenses import ToolFilteringDefense
from purpleaudit.utils.logger import log_error, log_info, log_json_conversation, log_warning

from . import instrument
from .context import AuditContext
from .transcript import build_transcript

# Bugs in the entrypoint: fail right away instead of retrying like an API error
_NOT_RETRIED = (ImportError, NameError, SyntaxError, AttributeError, TypeError, KeyError, NotImplementedError)

# Folders of the environments loaded in this process
_ENV_DIRS = set()


def _activate_env_dir(env_dir: str) -> None:
    """Make ``env_dir`` the first place to import from, and unload other environments'
    modules, so that environments can each have their own ``app.py``."""
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None) or ""
        if any(path.startswith(d + os.sep) for d in _ENV_DIRS if d != env_dir):
            del sys.modules[name]
    if env_dir in sys.path:
        sys.path.remove(env_dir)
    sys.path.insert(0, env_dir)
    _ENV_DIRS.add(env_dir)


def load_entrypoint(entrypoint: str, env_dir: str, env_name: str) -> Callable:
    """Resolve ``"module:function"``, looking for ``module`` in the environment folder first."""
    module_name, _, attr = entrypoint.partition(":")
    attr = attr or "run"
    module_file = os.path.join(env_dir, *module_name.split(".")) + ".py"
    if os.path.exists(module_file):
        # Lets the entrypoint import its sibling modules (your MAS code)
        _activate_env_dir(os.path.abspath(env_dir))
        unique_name = "_purpleaudit_env_" + re.sub(r"\W", "_", f"{env_name}_{module_name}")
        spec = importlib.util.spec_from_file_location(unique_name, module_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    fn = getattr(module, attr, None)
    if not callable(fn):
        raise AttributeError(f"Entrypoint '{entrypoint}': '{attr}' is not a function of module '{module_name}'")
    return fn


class BYOMASAdapter(BaseMASAdapter):
    """Runs ``entrypoint(task, ctx)`` for each conversation.

    Used for environments whose ``environment.json`` names an ``"entrypoint"``, and
    (with a built-in entrypoint) for spec-based LangGraph / CrewAI environments.
    """

    framework_name = "custom"
    supports_replay = False
    max_attempts = 3

    def __init__(self, *args, entrypoint: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.entrypoint = entrypoint or self.environment_config.get("entrypoint")
        if not self.entrypoint:
            raise ValueError(f"{self.env_path} has no \"entrypoint\" (expected \"module:function\")")
        self._run_fn = load_entrypoint(self.entrypoint, self.env_dir, self.environment_name)
        instrument.install()  # patch the frameworks your code imports
        self._attack: Optional[Callable] = None
        self._warned_no_tools = False

        # System prompts / tools seen at runtime, shown to the co-evolution blue team
        self.agents_path = f"{self.folder}/{self.model_id}/{self.environment_name}/agents.json"
        self.known_agents: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.agents_path):
            with open(self.agents_path, "r", encoding="utf-8") as f:
                self.known_agents = json.load(f)
        self.agents = [SimpleNamespace(name=name) for name in self.known_agents]

    def load(self, attack_to_inject: Optional[Callable] = None) -> None:
        # The MAS is rebuilt by the entrypoint on every run; only the attack is kept
        self._attack = attack_to_inject

    def run(self, prompt, filepath: Optional[str] = None) -> Dict[str, Any]:
        task = prompt if isinstance(prompt, str) else next(
            (m.get("content", "") for m in prompt if isinstance(m, dict)), str(prompt)
        )
        self.current_prompt = task
        for defense in (self.defense_config or {}).get("defenses", []):
            if hasattr(defense, "set_original_prompt"):
                defense.set_original_prompt(task)
            if isinstance(defense, ToolFilteringDefense):
                ToolFilteringDefense.configure(self.bedrock_config)

        ctx, result = self._call_entrypoint(task)
        final_result, chat_history = build_transcript(result, ctx)
        self._check_hooks(ctx)
        self._remember_agents(ctx)

        if filepath:
            for message in chat_history:
                log_json_conversation(message["name"], message["role"], message["content"], json_file=filepath)

        return {
            "final_result": final_result,
            "chat_history": chat_history,
            "prompt": prompt,
            "total_cost": 0.0,
            "usage_summary": {},
            "injected": ctx.injected if self._attack is not None else None,
            "tool_calls": ctx.tool_calls,
        }

    # ---------- Internals ----------

    def _new_context(self, task: str) -> AuditContext:
        return AuditContext(
            task,
            llm_config=self.bedrock_config,
            attack=self._attack,
            defense_config=self.defense_config,
            spec=self.environment_config if "agents" in self.environment_config else None,
            spec_tools=self.function_tools,
            max_rounds=self.max_rounds,
        )

    def _call_entrypoint(self, task: str):
        attempt = 0
        while attempt < self.max_attempts:
            attempt += 1
            ctx = self._new_context(task)  # fresh context: the attack must hit the first tool call again
            result, error = None, None
            try:
                with instrument.audit(ctx):
                    result = self._run_fn(task, ctx)
                    if inspect.isawaitable(result):
                        result = asyncio.run(result)
            except Exception as e:
                error = e
            if instrument.install():
                # The entrypoint imported an agent framework for the first time: its first
                # calls were not instrumented, so run the conversation again
                log_info(f"{self.entrypoint} imported an agent framework; running the conversation again")
                attempt -= 1
                continue
            if error is None:
                return ctx, result
            if type(error).__name__ == "GraphRecursionError":
                # Like reaching max_rounds in AG2: score the conversation as it stands
                log_warning(f"{self.entrypoint} hit LangGraph's recursion limit; scoring the conversation so far")
                return ctx, None
            if attempt == self.max_attempts or isinstance(error, _NOT_RETRIED):
                details = "".join(traceback.format_exception(type(error), error, error.__traceback__))
                log_error(f"{self.entrypoint} failed {attempt} times on task: {task[:80]}\n{details}")
                raise error
            log_warning(f"{self.entrypoint} raised {error!r} (attempt {attempt}/{self.max_attempts}), retrying...")
            time.sleep(5 * attempt)

    def _check_hooks(self, ctx: AuditContext) -> None:
        if ctx.defense_prompt and not ctx.system_prompts_seen:
            raise RuntimeError(
                f"The active defense adds text to the agents' system prompts, but no agent of {self.entrypoint} "
                f"was seen: PurpleAudit instruments LangChain/LangGraph, CrewAI and AG2; other code must pass its "
                f"prompts through ctx.system_prompt(agent, prompt)."
            )
        if ctx.tool_calls == 0 and not self._warned_no_tools:
            self._warned_no_tools = True
            log_warning(
                f"No tool call was seen while running {self.entrypoint}: attacks are planted in tool outputs. "
                f"PurpleAudit instruments LangChain/LangGraph, CrewAI and AG2 tools; wrap others with ctx.wrap_tools()."
            )

    def _remember_agents(self, ctx: AuditContext) -> None:
        # Record base prompts without the active defense so evolved defenses start from the original MAS
        updated = dict(self.known_agents)
        for name, info in ctx.agents.items():
            # Union: the tool_filtering defense only exposes a subset of the tools per run
            tools = set(info["tools"]) | set(updated.get(name, {}).get("tools", []))
            updated[name] = {"system_prompt": info["system_prompt"], "tools": sorted(tools)}
        if updated != self.known_agents:
            self.known_agents = updated
            self.agents = [SimpleNamespace(name=name) for name in updated]
            os.makedirs(os.path.dirname(self.agents_path), exist_ok=True)
            with open(self.agents_path, "w", encoding="utf-8") as f:
                json.dump(updated, f, indent=2, ensure_ascii=False)
            log_info(f"Agents seen in {self.environment_name}: {', '.join(updated)}")
