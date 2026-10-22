# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
MAS adapter layer.

The lab, evolution pipeline and evaluator run against ``BaseMASAdapter``. The
adapter is picked from the environment's ``environment.json``:

- ``"entrypoint": "module:function"``: a bring-your-own MAS (``purpleaudit.byo``).
- ``"framework": "ag2"`` (default): the original AG2 implementation of a
  spec-based environment (agents described in ``environment.json``).
- ``"framework": "langgraph" | "crewai"``: the same spec run on that framework
  by a built-in entrypoint (``purpleaudit.byo.spec``).
"""

import importlib
import json
import os
from typing import Any, Dict, Type

from purpleaudit.utils.logger import log_info
from purpleaudit.utils.bedrock_config import BedrockLLMConfig

from .base import BaseMASAdapter


_REGISTRY: Dict[str, str] = {
    # framework name -> "module:class" lazy import path
    "ag2": "purpleaudit.core.adapters.ag2:AG2MASAdapter",
    "autogen": "purpleaudit.core.adapters.ag2:AG2MASAdapter",  # alias
}

# Spec-based environments on other frameworks: built-in bring-your-own entrypoints
_SPEC_ENTRYPOINTS: Dict[str, str] = {
    "langgraph": "purpleaudit.byo.spec.langgraph:run",
    "crewai": "purpleaudit.byo.spec.crewai:run",
}
_SPEC_REQUIREMENTS: Dict[str, str] = {
    "langgraph": "langgraph",
    "crewai": "crewai",
}


def _resolve_adapter_class(framework: str) -> Type[BaseMASAdapter]:
    framework = (framework or "ag2").lower()
    if framework not in _REGISTRY:
        raise ValueError(
            f"Unknown MAS framework '{framework}'. "
            f"Supported: {sorted(set(_REGISTRY) | set(_SPEC_ENTRYPOINTS))}, "
            f"or set \"entrypoint\" in environment.json to audit your own MAS"
        )
    module_path, class_name = _REGISTRY[framework].split(":")
    return getattr(importlib.import_module(module_path), class_name)


def _peek_environment(folder: str, model_id: str, environment_name: str) -> Dict[str, Any]:
    """Read the environment.json that the adapter will load (model-specific first)."""
    for path in (
        f"{folder}/{model_id}/{environment_name}/environment.json",
        f"{folder}/default/{environment_name}/environment.json",
    ):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def create_mas(
    environment_name: str,
    bedrock_config: BedrockLLMConfig,
    folder: str = "environments",
    framework: str = None,
    **kwargs: Any,
) -> BaseMASAdapter:
    """
    Factory that returns the right adapter for an environment.

    Args:
        environment_name: Name of the environment directory.
        bedrock_config: LLM config of the model under audit (also used by the judge).
        folder: Root environments folder.
        framework: Force a specific framework for a spec-based environment. If None,
                   use the ``"framework"`` field of ``environment.json`` (default ``"ag2"``).
        **kwargs: Forwarded to the adapter constructor.
    """
    from purpleaudit.byo.adapter import BYOMASAdapter

    env_cfg = _peek_environment(folder, kwargs.get("model_id", "default"), environment_name)
    framework = (framework or env_cfg.get("framework") or "ag2").lower()

    if env_cfg.get("entrypoint"):
        log_info(f"Using bring-your-own MAS: {env_cfg['entrypoint']} (framework={framework})")
        return BYOMASAdapter(environment_name=environment_name, bedrock_config=bedrock_config, folder=folder, **kwargs)

    if framework in _SPEC_ENTRYPOINTS:
        requirement = _SPEC_REQUIREMENTS[framework]
        try:
            importlib.import_module(requirement)
        except ImportError as e:
            raise ImportError(f"'{requirement}' is not installed. Install with: pip install -e \".[{framework}]\"") from e
        log_info(f"Using spec-based MAS on {framework}")
        return BYOMASAdapter(environment_name=environment_name, bedrock_config=bedrock_config, folder=folder,
                             entrypoint=_SPEC_ENTRYPOINTS[framework], **kwargs)

    adapter_cls = _resolve_adapter_class(framework)
    log_info(f"Using MAS adapter: {adapter_cls.__name__} (framework={framework})")
    return adapter_cls(
        environment_name=environment_name,
        bedrock_config=bedrock_config,
        folder=folder,
        **kwargs,
    )


__all__ = ["BaseMASAdapter", "create_mas"]
