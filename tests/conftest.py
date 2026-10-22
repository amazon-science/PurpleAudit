# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from stub_llm import StubServer

REPO = Path(__file__).resolve().parent.parent
MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


@pytest.fixture(scope="session")
def stub():
    with StubServer() as server:
        yield server


@pytest.fixture(autouse=True)
def crewai_storage(tmp_path, monkeypatch):
    """Keep CrewAI's app data (named after the working directory by default) out of the user's home."""
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path / "crewai_storage"))


@pytest.fixture
def cli(stub, tmp_path):
    """Run PurpleAudit commands in ``tmp_path`` with every LLM call going to the stub."""
    env = dict(
        os.environ,
        LLM_PROVIDER="openrouter",
        OPENROUTER_API_KEY="sk-or-v1-test",
        OPENROUTER_BASE_URL=stub.url,
        AWS_CONFIG_FILE="/nonexistent",
        AWS_SHARED_CREDENTIALS_FILE="/nonexistent",
        CREWAI_STORAGE_DIR=str(tmp_path / "crewai_storage"),
        CREWAI_DISABLE_TELEMETRY="true",
        OTEL_SDK_DISABLED="true",
    )

    class Cli:
        workdir = tmp_path

        def copy_envs(self, source: Path, name: str = "envs", framework: dict = None) -> str:
            # Only the definitions: results of local runs (<source>/<model>/...) must not leak in
            shutil.copytree(source / "default", tmp_path / name / "default")
            for env_name, fw in (framework or {}).items():
                path = tmp_path / name / "default" / env_name / "environment.json"
                cfg = json.loads(path.read_text())
                cfg["framework"] = fw
                path.write_text(json.dumps(cfg))
            return name

        def run(self, module: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
            proc = subprocess.run(
                [sys.executable, "-m", f"purpleaudit.{module}", *args],
                cwd=tmp_path, env=env, capture_output=True, text=True, timeout=900,
            )
            proc.log = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout + proc.stderr)
            if check and proc.returncode != 0:
                raise AssertionError(f"purpleaudit.{module} {' '.join(args)} failed:\n{proc.log[-4000:]}")
            return proc

        def asr_utility(self, folder: str, env_name: str, defense: str = None):
            args = ["per-env", "--folder", folder, "--model", MODEL] + (["--defense-id", defense] if defense else [])
            out = self.run("evaluate", *args).log
            match = re.search(rf"^{env_name}\s+([\d.]+)%\s+([\d.]+)%\s+(\d+)", out, re.M)
            assert match, f"no results for {env_name}:\n{out}"
            return float(match.group(1)), float(match.group(2)), int(match.group(3))

    return Cli()
