# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Base class shared by all MAS adapters.

Handles the parts of the lifecycle that are framework-independent:
    - Resolving where ``environment.json`` / ``tools.py`` / ``prompts.json`` live
    - Loading + variable-substituting the environment config
    - Importing the tools module of spec-based environments

Subclasses implement ``load()`` and ``run()``.
"""

from __future__ import annotations

import os
import json
import importlib.util
import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from purpleaudit.utils.logger import log_error, log_info, log_warning
from purpleaudit.utils.bedrock_config import BedrockLLMConfig


class BaseMASAdapter(ABC):
    """Common interface the lab, evolution pipeline and evaluator run against."""

    framework_name: str = "base"
    # Whether attack runs can resume the benign conversation right before its first
    # tool call. Adapters that cannot restart from the original task instead.
    supports_replay: bool = False

    def __init__(
        self,
        environment_name: str,
        bedrock_config: BedrockLLMConfig,
        folder: str = "environments",
        system_prompt_variables: Optional[dict] = None,
        defense_config: Optional[dict] = None,
        model_id: str = "default",
        max_rounds: int = 15,
    ):
        self.environment_name = environment_name
        self.bedrock_config = bedrock_config
        self.folder = folder
        self.model_id = model_id
        self.system_prompt_variables = system_prompt_variables or {}
        self.defense_config = defense_config
        self.max_rounds = max_rounds

        self.agents: List[Any] = []
        self.tools_module = None
        self.function_tools: Dict[str, Callable] = {}
        self.prompts: Optional[Dict[str, Any]] = None
        self.current_prompt = None
        self.stored_attack_to_inject: Optional[Callable] = None

        self.effective_model_id = self._resolve_model_id()
        self.env_dir = f"{self.folder}/{self.effective_model_id}/{self.environment_name}"
        self.env_path = f"{self.env_dir}/environment.json"
        self.prompts_path = f"{self.env_dir}/prompts.json"
        self.environment_config = self.load_environment(self.system_prompt_variables)

        random.seed(42)

    # ---------- Path resolution ----------

    def _resolve_model_id(self) -> str:
        """Mirror MultiAgentSystem._resolve_model_id so paths stay identical."""
        if self.folder == "factories":
            if os.path.exists(f"factories/{self.model_id}/factory"):
                return self.model_id
            if os.path.exists("factories/default/factory"):
                log_info(
                    f"Factory directory 'factories/{self.model_id}/factory' not found, "
                    f"using default factory"
                )
                return "default"
        else:
            if os.path.exists(f"{self.folder}/{self.model_id}/{self.environment_name}/environment.json"):
                return self.model_id
            if os.path.exists(f"{self.folder}/default/{self.environment_name}"):
                log_info(
                    f"Environment directory for model '{self.model_id}' not found, using default"
                )
                return "default"
        return self.model_id

    # ---------- Environment / tools / prompts ----------

    def load_environment(self, system_prompt_variables: Optional[dict] = None) -> Dict[str, Any]:
        if not os.path.exists(self.env_path):
            model_folder = f"{self.folder}/{self.effective_model_id}"
            available = (
                [d for d in os.listdir(model_folder) if os.path.isdir(f"{model_folder}/{d}")]
                if os.path.exists(model_folder)
                else []
            )
            raise FileNotFoundError(
                f"Environment '{self.environment_name}' not found.\n"
                f"Path checked: {self.env_path}\n"
                f"Available: {', '.join(available) if available else 'None'}"
            )

        with open(self.env_path, "r", encoding="utf-8") as f:
            env_cfg = json.load(f)

        if system_prompt_variables:
            blob = json.dumps(env_cfg)
            for k, v in system_prompt_variables.items():
                blob = blob.replace(k, str(v))
            env_cfg = json.loads(blob)

        # Only spec-based environments (agents described in environment.json) ship a tools.py
        if "agents" in env_cfg:
            self._load_tools_module()
        self._load_prompts()
        return env_cfg

    def _load_tools_module(self):
        tools_path = f"{self.env_dir}/tools.py"
        if not os.path.exists(tools_path):
            log_warning(f"No tools.py found for environment {self.environment_name}")
            return
        try:
            spec = importlib.util.spec_from_file_location(
                f"{self.environment_name}_tools", tools_path
            )
            self.tools_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.tools_module)
            for name in dir(self.tools_module):
                obj = getattr(self.tools_module, name)
                if callable(obj) and not name.startswith("_") and hasattr(obj, "__call__"):
                    if not (hasattr(obj, "__module__") and "typing" in str(obj.__module__)):
                        self.function_tools[name] = obj
        except Exception as e:
            log_error(f"Failed to load tools module: {e}")
            self.tools_module = None

    def _load_prompts(self):
        if not os.path.exists(self.prompts_path):
            log_warning(f"No prompts.json found for environment {self.environment_name}")
            return
        with open(self.prompts_path, "r", encoding="utf-8") as f:
            self.prompts = json.load(f)

    # ---------- Abstract framework hooks ----------

    @abstractmethod
    def load(self, attack_to_inject: Optional[Callable] = None) -> None:
        """Set up the next runs, injecting ``attack_to_inject`` into the first tool output."""

    @abstractmethod
    def run(self, prompt, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute one conversation.

        Returns a dict with:
            - final_result (str)
            - chat_history (list[dict] with {role, name, content})
            - prompt
            - total_cost (float)
            - usage_summary (dict)
            - injected (bool | None): whether the attack reached a tool output
              (None when no attack was set)
        Tool messages must have ``"tool"`` somewhere in ``role`` so the lab can
        locate the first tool call.
        """

    # ---------- Cost (default no-op so the lab keeps working) ----------

    def get_cost(self) -> Dict[str, Any]:
        return {"total_cost": 0.0, "usage_summary": {}}
