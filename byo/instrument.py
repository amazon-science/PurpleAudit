# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Automatic instrumentation of agent frameworks for bring-your-own MAS audits.

While a conversation is audited, the tool calls and LLM requests that
LangChain / LangGraph, CrewAI and AG2 make go through the active ``AuditContext``:

- tool outputs: the attack is planted in the first one, tool-level defenses wrap
  every one, and they are recorded in the transcript;
- LLM requests: the active defense is appended to the agent's system prompt, the
  prompt and tools are recorded for the co-evolution blue team, and the reply is
  recorded in the transcript;
- the tool_filtering defense disables the tools an agent does not need.

The patches are installed once, on the frameworks already imported, and hook
methods looked up at call time, so agents created before the audit are covered
too. Outside an audited conversation they do nothing.
"""

import inspect
import sys
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterable, List, Optional

from .context import DISABLED_TOOL_OUTPUT
from .tools import acall_once, call_once
from .transcript import _dict_message, langchain_message_to_dict

_active = None  # AuditContext of the conversation being audited
_patched = set()
_install_lock = threading.Lock()


def active():
    """The context to report to, or None outside an audited conversation (or while PurpleAudit calls an LLM itself)."""
    ctx = _active
    return None if ctx is None or ctx.is_paused else ctx


@contextmanager
def audit(ctx):
    """Route the instrumented frameworks to ``ctx`` for the duration of one conversation."""
    global _active
    install()
    token = _RUN_TREE.set(_new_run_tree())
    restore_crewai_prompts = _silence_crewai_prompts()
    _active = ctx
    try:
        yield ctx
    finally:
        _active = None
        _RUN_TREE.reset(token)
        restore_crewai_prompts()


def install() -> bool:
    """Patch the supported frameworks imported so far. Returns True if one was newly patched."""
    patches = (("langchain_core", _patch_langchain), ("crewai", _patch_crewai), ("autogen", _patch_ag2))
    newly_patched = False
    with _install_lock:
        for module, patch in patches:
            if module in sys.modules and module not in _patched:
                patch()
                _patched.add(module)
                newly_patched = True
    return newly_patched


def _intercept_tool(ctx, name: str, agent: Optional[str], call: Callable, disabled: Callable[[], Any],
                    serialize: bool = False):
    if ctx.is_disabled(name):
        return disabled()
    return call_once(name, call, lambda output: ctx.on_tool_output(name, agent, output, serialize=serialize))


async def _aintercept_tool(ctx, name: str, agent: Optional[str], call: Callable, disabled: Callable[[], Any],
                           serialize: bool = False):
    if ctx.is_disabled(name):
        return disabled()
    return await acall_once(name, call, lambda output: ctx.on_tool_output(name, agent, output, serialize=serialize))


def _schema_name_and_description(tool: Any) -> dict:
    if isinstance(tool, dict):  # OpenAI, Anthropic or Bedrock tool schema
        spec = tool.get("function") or tool.get("toolSpec") or tool
        return {"name": spec.get("name", ""), "description": spec.get("description", "")}
    return {"name": getattr(tool, "name", ""), "description": getattr(tool, "description", "")}


def _text_of(content: Any) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content or ""


def _defended_content(content: Any, prompt: str, defended: str) -> Any:
    """``content`` (text or content blocks) with the defense that ``defended`` adds to ``prompt``."""
    if isinstance(content, list):
        return [*content, {"type": "text", "text": defended[len(prompt):]}]
    return defended


# ---------- LangChain / LangGraph ----------

# Run tree of the audited conversation, used to name the agent behind each call:
# the closest enclosing run with a meaningful name (e.g. the graph passed name="researcher")
_RUN_TREE: ContextVar = ContextVar("purpleaudit_run_tree", default=None)
_GENERIC_RUN_NAMES = {"agent", "tools", "model", "call_model", "LangGraph", "should_continue",
                      "ChatPromptTemplate", "PromptTemplate"}
_new_run_tree: Callable = lambda: None


def _named_run(run_id) -> Optional[str]:
    """Name of ``run_id`` or of its closest ancestor with a meaningful name."""
    tree = _RUN_TREE.get()
    while tree is not None and run_id in tree.runs:
        name, run_id = tree.runs[run_id]
        if name and name not in _GENERIC_RUN_NAMES and not name.startswith(("Runnable", "Channel", "_")):
            return name
    return None


def _parent_run(run_id):
    tree = _RUN_TREE.get()
    return tree.runs.get(run_id, (None, None))[1] if tree is not None else None


def _patch_langchain():
    global _new_run_tree
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import SystemMessage, ToolMessage
    from langchain_core.runnables.config import ensure_config
    from langchain_core.tools import BaseTool
    from langchain_core.tracers.context import register_configure_hook

    class RunTree(BaseCallbackHandler):
        run_inline = True  # record runs before their children start, also in async code

        def __init__(self):
            self.runs = {}

        def _start(self, serialized, run_id, parent_run_id, name):
            self.runs[run_id] = (name or (serialized or {}).get("name"), parent_run_id)

        def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, name=None, **kwargs):
            self._start(serialized, run_id, parent_run_id, name)

        def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, name=None, **kwargs):
            self._start(serialized, run_id, parent_run_id, name)

        def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, name=None, **kwargs):
            self._start(serialized, run_id, parent_run_id, name)

    register_configure_hook(_RUN_TREE, True)
    _new_run_tree = RunTree

    # ----- Tools: BaseTool.run / arun -----
    # A tool called by a model returns a ToolMessage (rewritten); called directly by
    # code, its output keeps its type unless it is text

    def tool_agent(kwargs) -> Optional[str]:
        # ``callbacks`` is the manager of the run that called the tool (e.g. a ToolNode)
        return _named_run(getattr(kwargs.get("callbacks"), "parent_run_id", None))

    def disabled_output(tool, kwargs):
        tool_call_id = kwargs.get("tool_call_id")
        if tool_call_id:
            return ToolMessage(DISABLED_TOOL_OUTPUT, name=tool.name, tool_call_id=tool_call_id)
        return DISABLED_TOOL_OUTPUT

    original_run, original_arun = BaseTool.run, BaseTool.arun

    def run(self, tool_input, *args, **kwargs):
        ctx = active()
        if ctx is None:
            return original_run(self, tool_input, *args, **kwargs)
        return _intercept_tool(ctx, self.name, tool_agent(kwargs), lambda: original_run(self, tool_input, *args, **kwargs),
                               lambda: disabled_output(self, kwargs))

    async def arun(self, tool_input, *args, **kwargs):
        ctx = active()
        if ctx is None:
            return await original_arun(self, tool_input, *args, **kwargs)
        return await _aintercept_tool(ctx, self.name, tool_agent(kwargs),
                                      lambda: original_arun(self, tool_input, *args, **kwargs),
                                      lambda: disabled_output(self, kwargs))

    BaseTool.run, BaseTool.arun = run, arun

    # ----- Chat models: invoke / batch go through _generate_with_cache, streaming through stream -----

    def defend(ctx, agent: str, messages: List[Any], tools: Optional[Iterable[Any]]) -> List[Any]:
        index = next((i for i, m in enumerate(messages) if m.type == "system"), None)
        content = messages[index].content if index is not None else ""
        prompt = _text_of(content)
        defended = ctx.agent_prompt(agent, prompt, [_schema_name_and_description(t) for t in tools or []])
        if defended == prompt:
            return messages
        messages = list(messages)
        if index is None:
            messages.insert(0, SystemMessage(defended))
        else:
            messages[index] = messages[index].model_copy(update={"content": _defended_content(content, prompt, defended)})
        return messages

    def record(ctx, agent: str, message) -> None:
        entry = langchain_message_to_dict(message, agent)
        if entry is not None:
            ctx.record_message(agent, entry["content"])

    def model_agent(run_manager) -> str:
        return _named_run(_parent_run(getattr(run_manager, "run_id", None))) or "agent"

    def config_agent(config) -> str:
        # The callbacks of a run's config belong to the run that started it (e.g. a graph node)
        return _named_run(getattr(ensure_config(config).get("callbacks"), "parent_run_id", None)) or "agent"

    def recorded_since(ctx, count: int) -> bool:
        return any(m["role"] == "assistant" for m in ctx.messages[count:])

    original_generate = BaseChatModel._generate_with_cache
    original_agenerate = BaseChatModel._agenerate_with_cache
    original_stream, original_astream = BaseChatModel.stream, BaseChatModel.astream

    def generate(self, messages, stop=None, run_manager=None, **kwargs):
        ctx = active()
        if ctx is None:
            return original_generate(self, messages, stop=stop, run_manager=run_manager, **kwargs)
        agent = model_agent(run_manager)
        result = original_generate(self, defend(ctx, agent, messages, kwargs.get("tools")),
                                   stop=stop, run_manager=run_manager, **kwargs)
        for generation in result.generations[:1]:
            record(ctx, agent, generation.message)
        return result

    async def agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        ctx = active()
        if ctx is None:
            return await original_agenerate(self, messages, stop=stop, run_manager=run_manager, **kwargs)
        agent = model_agent(run_manager)
        result = await original_agenerate(self, defend(ctx, agent, messages, kwargs.get("tools")),
                                          stop=stop, run_manager=run_manager, **kwargs)
        for generation in result.generations[:1]:
            record(ctx, agent, generation.message)
        return result

    def stream(self, input, config=None, *, stop=None, **kwargs):
        ctx = active()
        if ctx is None:
            yield from original_stream(self, input, config, stop=stop, **kwargs)
            return
        agent = config_agent(config)
        messages = defend(ctx, agent, self._convert_input(input).to_messages(), kwargs.get("tools"))
        count, reply = len(ctx.messages), None
        for chunk in original_stream(self, messages, config, stop=stop, **kwargs):
            reply = chunk if reply is None else reply + chunk
            yield chunk
        # Models that cannot stream fall back to invoke, which already recorded the reply
        if reply is not None and not recorded_since(ctx, count):
            record(ctx, agent, reply)

    async def astream(self, input, config=None, *, stop=None, **kwargs):
        ctx = active()
        if ctx is None:
            async for chunk in original_astream(self, input, config, stop=stop, **kwargs):
                yield chunk
            return
        agent = config_agent(config)
        messages = defend(ctx, agent, self._convert_input(input).to_messages(), kwargs.get("tools"))
        count, reply = len(ctx.messages), None
        async for chunk in original_astream(self, messages, config, stop=stop, **kwargs):
            reply = chunk if reply is None else reply + chunk
            yield chunk
        if reply is not None and not recorded_since(ctx, count):
            record(ctx, agent, reply)

    BaseChatModel._generate_with_cache = generate
    BaseChatModel._agenerate_with_cache = agenerate
    BaseChatModel.stream, BaseChatModel.astream = stream, astream


# ---------- CrewAI ----------

# CrewAI agents running, innermost last (a crew runs its tasks one at a time, and runs
# tools in worker threads, so this is process-wide)
_crewai_agents: List[str] = []


def _crewai_agent() -> Optional[str]:
    return _crewai_agents[-1] if _crewai_agents else None


def _silence_crewai_prompts() -> Callable[[], None]:
    """Skip CrewAI's interactive "share this execution trace?" prompt, which waits 20 s after runs.

    CrewAI's event handlers run in a copy of this context, so the flag reaches them.
    """
    if "crewai" not in _patched:
        return lambda: None
    try:
        from crewai.events.listeners.tracing.utils import set_suppress_tracing_messages, should_suppress_tracing_messages
    except ImportError:
        return lambda: None
    previous = should_suppress_tracing_messages()
    set_suppress_tracing_messages(True)
    return lambda: set_suppress_tracing_messages(previous)


def _patch_crewai():
    from crewai import Agent
    from crewai.agents.cache.cache_handler import CacheHandler
    from crewai.tools.base_tool import BaseTool, Tool
    from crewai.tools.structured_tool import CrewStructuredTool

    disabled = lambda: DISABLED_TOOL_OUTPUT

    def is_internal(tool) -> bool:
        # CrewAI's own delegation tools ("Delegate work to coworker", ...) are not data sources
        return type(tool).__module__.startswith("crewai.tools.agent_tools")

    # ----- Tools: BaseTool.run, Tool.run (``@tool``) or CrewStructuredTool.invoke -----
    # CrewAI hands tool outputs to the LLM as text, so data outputs are serialized

    def patch_tool(cls, method_name):
        original = vars(cls).get(method_name)
        if original is None:
            return

        def method(self, *args, **kwargs):
            ctx = active()
            if ctx is None or is_internal(self):
                return original(self, *args, **kwargs)
            return _intercept_tool(ctx, self.name, _crewai_agent(), lambda: original(self, *args, **kwargs),
                                   disabled, serialize=True)

        async def amethod(self, *args, **kwargs):
            ctx = active()
            if ctx is None or is_internal(self):
                return await original(self, *args, **kwargs)
            return await _aintercept_tool(ctx, self.name, _crewai_agent(),
                                          lambda: original(self, *args, **kwargs), disabled, serialize=True)
        setattr(cls, method_name, amethod if inspect.iscoroutinefunction(original) else method)

    for cls in (BaseTool, Tool):
        patch_tool(cls, "run")
        patch_tool(cls, "arun")
    patch_tool(CrewStructuredTool, "invoke")
    patch_tool(CrewStructuredTool, "ainvoke")

    # ----- Tool cache: off during audits (a cached output would replay an earlier injection) -----

    original_read, original_add = CacheHandler.read, CacheHandler.add
    CacheHandler.read = lambda self, *a, **kw: None if active() else original_read(self, *a, **kw)
    CacheHandler.add = lambda self, *a, **kw: None if active() else original_add(self, *a, **kw)

    # ----- Agents: defense in the backstory (the system prompt is built from it), output recorded -----

    def start(ctx, agent, tools):
        specs = [{"name": t.name, "description": t.description} for t in tools or [] if not is_internal(t)]
        backstory = agent.backstory
        agent.backstory = ctx.agent_prompt(agent.role, backstory, specs)
        _crewai_agents.append(agent.role)
        return backstory

    def finish(ctx, agent, backstory, result):
        _crewai_agents.pop()
        agent.backstory = backstory
        if result is not None:
            output = str(getattr(result, "raw", result))
            last = ctx.messages[-1] if ctx.messages else {}
            if (last.get("name"), last.get("content")) != (agent.role, output):  # CrewAI's retries re-run the task
                ctx.record_message(agent.role, output)

    def patch_agent(method_name, tools_of):
        original = getattr(Agent, method_name, None)
        if original is None:
            return

        def method(self, *args, **kwargs):
            ctx = active()
            if ctx is None:
                return original(self, *args, **kwargs)
            backstory, result = start(ctx, self, tools_of(self, args, kwargs)), None
            try:
                result = original(self, *args, **kwargs)
                return result
            finally:
                finish(ctx, self, backstory, result)

        async def amethod(self, *args, **kwargs):
            ctx = active()
            if ctx is None:
                return await original(self, *args, **kwargs)
            backstory, result = start(ctx, self, tools_of(self, args, kwargs)), None
            try:
                result = await original(self, *args, **kwargs)
                return result
            finally:
                finish(ctx, self, backstory, result)
        setattr(Agent, method_name, amethod if inspect.iscoroutinefunction(original) else method)

    def task_tools(agent, args, kwargs):  # execute_task(task, context=None, tools=None)
        tools = kwargs.get("tools", args[2] if len(args) > 2 else None)
        return tools if tools is not None else agent.tools

    patch_agent("execute_task", task_tools)
    patch_agent("aexecute_task", task_tools)
    # Standalone agents and Flows
    patch_agent("kickoff", lambda agent, args, kwargs: agent.tools)
    patch_agent("kickoff_async", lambda agent, args, kwargs: agent.tools)


# ---------- AG2 ----------

_AG2_INTERNAL_AGENTS = {"checking_agent", "speaker_selection_agent"}


def _ag2_tool_specs(agent) -> List[dict]:
    config = agent.llm_config
    if not config:
        return []
    tools = config.get("tools") if isinstance(config, dict) else getattr(config, "tools", None)
    specs = []
    for tool in tools or []:
        tool = tool if isinstance(tool, dict) else getattr(tool, "model_dump", lambda: {})()
        specs.append(_schema_name_and_description(tool))
    return specs


def _patch_ag2():
    from autogen import ConversableAgent
    from autogen.agentchat.groupchat import GroupChatManager

    def is_internal(agent) -> bool:
        return isinstance(agent, GroupChatManager) or agent.name in _AG2_INTERNAL_AGENTS or agent.name.startswith("_")

    # ----- Tools: execute_function / a_execute_function -----
    # AG2 hands tool outputs to the LLM as text, so data outputs are serialized (ReplyResults are left alone)

    def tool_call(self, func_call):
        name = func_call.get("name", "")
        originator = getattr(self, "get_tool_call_originator", lambda: None)() or self.name
        # Group handoffs are control flow, not data
        is_handoff = getattr(self, "function_is_agent_llm_handoff", lambda *a: False)(originator, name)
        return name, originator, is_handoff

    def post(ctx, name, originator):
        def process(result):
            ok, message = result
            return ok, dict(message, content=ctx.on_tool_output(name, originator, message.get("content"), serialize=True))
        return process

    def disabled(name):
        return True, {"name": name, "role": "function", "content": DISABLED_TOOL_OUTPUT}

    original_execute = ConversableAgent.execute_function
    original_aexecute = ConversableAgent.a_execute_function

    def execute_function(self, func_call, *args, **kwargs):
        ctx = active()
        name, originator, is_handoff = tool_call(self, func_call)
        if ctx is None or is_handoff:
            return original_execute(self, func_call, *args, **kwargs)
        if ctx.is_disabled(name):
            return disabled(name)
        return call_once(name, lambda: original_execute(self, func_call, *args, **kwargs), post(ctx, name, originator))

    async def a_execute_function(self, func_call, *args, **kwargs):
        ctx = active()
        name, originator, is_handoff = tool_call(self, func_call)
        if ctx is None or is_handoff:
            return await original_aexecute(self, func_call, *args, **kwargs)
        if ctx.is_disabled(name):
            return disabled(name)
        return await acall_once(name, lambda: original_aexecute(self, func_call, *args, **kwargs),
                                post(ctx, name, originator))

    ConversableAgent.execute_function = execute_function
    ConversableAgent.a_execute_function = a_execute_function

    # ----- LLM calls: every reply (sync or async) goes through _generate_oai_reply_from_client,
    # looked up on the agent at call time, with the system message first in ``messages`` -----

    original_from_client = ConversableAgent._generate_oai_reply_from_client

    def generate_oai_reply_from_client(self, llm_client, messages, cache, **kwargs):
        ctx = active()
        if ctx is None or is_internal(self):
            return original_from_client(self, llm_client, messages, cache, **kwargs)
        index = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
        content = messages[index].get("content") if index is not None else ""
        prompt = _text_of(content)
        defended = ctx.agent_prompt(self.name, prompt, _ag2_tool_specs(self))
        if defended != prompt:
            messages = list(messages)
            if index is None:
                messages.insert(0, {"role": "system", "content": defended})
            else:
                messages[index] = dict(messages[index], content=_defended_content(content, prompt, defended))
        reply = original_from_client(self, llm_client, messages, cache, **kwargs)
        if reply is not None:
            ctx.record_message(self.name, _dict_message(reply)["content"] if isinstance(reply, dict) else str(reply))
        return reply

    ConversableAgent._generate_oai_reply_from_client = generate_oai_reply_from_client
