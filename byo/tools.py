# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Wrap a MAS's tools so every output goes through the audit context.

Supports plain (sync or async) functions and the tool objects of LangChain /
LangGraph and CrewAI. The wrapped tool keeps its name, description and
argument schema, so the framework sees the same tool as before.
"""

import contextvars
import copy
import functools
import inspect
import typing
from typing import Any, Callable

PostProcess = Callable[[Any], Any]

# Tools whose call is in progress in this context. A wrapper running inside another
# wrapper of the same tool (an ``_arun`` calling ``_run``, a tool wrapped twice) leaves
# the output to the outer one, so every output is post-processed exactly once.
_ACTIVE_TOOLS: contextvars.ContextVar = contextvars.ContextVar("purpleaudit_active_tools", default=frozenset())


def call_once(key: str, call: Callable, post: PostProcess):
    active = _ACTIVE_TOOLS.get()
    if key in active:
        return call()
    token = _ACTIVE_TOOLS.set(active | {key})
    try:
        output = call()
    finally:
        _ACTIVE_TOOLS.reset(token)
    return post(output)


async def acall_once(key: str, call: Callable, post: PostProcess):
    active = _ACTIVE_TOOLS.get()
    if key in active:
        return await call()
    token = _ACTIVE_TOOLS.set(active | {key})
    try:
        output = await call()
    finally:
        _ACTIVE_TOOLS.reset(token)
    return post(output)


def _is_framework_tool(tool: Any) -> bool:
    return any(
        cls.__name__ == "BaseTool" and cls.__module__.startswith(("langchain_core.", "crewai."))
        for cls in type(tool).__mro__
    )


def tool_name(tool: Any) -> str:
    if _is_framework_tool(tool):
        return tool.name
    return getattr(tool, "__name__", None) or type(tool).__name__


def tool_description(tool: Any) -> str:
    if _is_framework_tool(tool):
        return tool.description or ""
    return inspect.getdoc(tool) or ""


def wrap_function(fn: Callable, post: PostProcess, key: str = None) -> Callable:
    """Wrap a function, keeping its signature (frameworks build tool schemas from it)."""
    key = key or getattr(fn, "__qualname__", repr(fn))
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            return await acall_once(key, lambda: fn(*args, **kwargs), post)
    else:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return call_once(key, lambda: fn(*args, **kwargs), post)

    # Resolve string annotations (``from __future__ import annotations``) in the tool's
    # own module: frameworks would otherwise evaluate them in this module's globals
    try:
        wrapper.__signature__ = inspect.signature(fn, eval_str=True)
        wrapper.__annotations__ = typing.get_type_hints(fn, include_extras=True)
    except Exception:
        pass
    return wrapper


def _wrap_method(method: Callable, post: PostProcess, key: str) -> Callable:
    if inspect.iscoroutinefunction(method):
        @functools.wraps(method)
        async def async_method(self, *args, **kwargs):
            return await acall_once(key, lambda: method(self, *args, **kwargs), post)
        return async_method

    @functools.wraps(method)
    def sync_method(self, *args, **kwargs):
        return call_once(key, lambda: method(self, *args, **kwargs), post)
    return sync_method


def wrap_framework_tool(tool: Any, post: PostProcess) -> Any:
    """Copy of a LangChain / CrewAI ``BaseTool`` whose outputs go through ``post``."""
    key = tool.name
    if getattr(tool, "response_format", None) == "content_and_artifact":
        # The tool returns (content, artifact): only the content reaches the model
        process = post
        post = lambda output: (process(output[0]), output[1]) if isinstance(output, tuple) and len(output) == 2 else process(output)

    wrapped = copy.copy(tool)

    # Function-backed tools (``@tool`` in both frameworks) call ``func`` directly
    if callable(getattr(tool, "func", None)):
        object.__setattr__(wrapped, "func", wrap_function(tool.func, post, key))
        if callable(getattr(tool, "coroutine", None)):
            object.__setattr__(wrapped, "coroutine", wrap_function(tool.coroutine, post, key))
        return wrapped

    # Subclassed tools: override the ``_run`` / ``_arun`` the subclass defines.
    # Framework defaults (e.g. ``_arun`` delegating to ``_run``) are left alone so
    # an output is never post-processed twice.
    cls = type(tool)
    overrides = {}
    for name in ("_run", "_arun"):
        owner = next((c for c in cls.__mro__ if name in vars(c)), None)
        if owner is None or owner.__name__ == "BaseTool" or getattr(vars(owner)[name], "__isabstractmethod__", False):
            continue
        overrides[name] = _wrap_method(vars(owner)[name], post, key)
    object.__setattr__(wrapped, "__class__", type(cls.__name__, (cls,), overrides))
    return wrapped


def wrap_tool(tool: Any, post: PostProcess) -> Any:
    if _is_framework_tool(tool):
        return wrap_framework_tool(tool, post)
    if callable(tool):
        return wrap_function(tool, post, tool_name(tool))
    raise TypeError(
        f"Cannot wrap tool {tool!r}: expected a function or a LangChain/CrewAI BaseTool. "
        f"Wrap the underlying Python function instead."
    )
