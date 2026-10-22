# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Factory tools — read/write the three files that define a generated MAS.

The factory pipeline (MAS Builder → State Builder → Prompt Builder) collaborates
through these tools. Output paths are derived from the environment variables
``PURPLEAUDIT_OUTPUT_FOLDER`` (default ``environments``) and ``PURPLEAUDIT_MODEL_ID``
(default ``default``), which framework.py sets before invoking the factory.

All write_* functions return strings containing ``file_name=<path>/environment.json``
when applicable, which is how ``utils.utils.extract_project_name`` recovers the
generated env name from the factory's chat history.
"""

import json
import os


def _resolve_out_dir(env_name: str) -> str:
    """Build (and create) the output directory for a generated MAS."""
    folder = os.environ.get("PURPLEAUDIT_OUTPUT_FOLDER", "environments")
    model = os.environ.get("PURPLEAUDIT_MODEL_ID", "default")
    out = os.path.join(folder, model, env_name)
    os.makedirs(out, exist_ok=True)
    return out


def write_environment_json(env_name: str, content: str) -> str:
    """Persist environment.json for a new MAS. ``content`` must be valid JSON."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        return f"ERROR: invalid JSON for environment.json: {e}"
    if "agents" not in parsed:
        return "ERROR: environment.json must contain an 'agents' key"

    out = _resolve_out_dir(env_name)
    path = os.path.join(out, "environment.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)
    n_agents = len(parsed.get("agents", []))
    return f"OK wrote {n_agents} agents — file_name={path}"


def write_tools_py(env_name: str, content: str) -> str:
    """Persist tools.py for a generated MAS."""
    if "import" not in content and "def " not in content:
        return "ERROR: tools.py must contain at least one function definition"

    out = _resolve_out_dir(env_name)
    path = os.path.join(out, "tools.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    n_funcs = content.count("\ndef ")
    return f"OK wrote {n_funcs} tool functions — file_path={path}"


def write_prompts_json(env_name: str, content: str) -> str:
    """Persist prompts.json for a generated MAS."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        return f"ERROR: invalid JSON for prompts.json: {e}"

    for k, v in parsed.items():
        if not isinstance(v, dict) or "prompt" not in v or "ground_truth" not in v:
            return f"ERROR: task '{k}' must have both 'prompt' and 'ground_truth' fields"

    out = _resolve_out_dir(env_name)
    path = os.path.join(out, "prompts.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)
    return f"OK wrote {len(parsed)} tasks — file_path={path}"


def read_environment_json(env_name: str) -> str:
    """Return the contents of a previously-written environment.json."""
    path = os.path.join(_resolve_out_dir(env_name), "environment.json")
    if not os.path.exists(path):
        return f"ERROR: {path} not found — MAS Builder has not run yet."
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_tools_py(env_name: str) -> str:
    """Return the contents of a previously-written tools.py."""
    path = os.path.join(_resolve_out_dir(env_name), "tools.py")
    if not os.path.exists(path):
        return f"ERROR: {path} not found — State Builder has not run yet."
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
