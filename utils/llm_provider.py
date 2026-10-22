# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
LLM provider selection: AWS Bedrock (default, used for the paper) or OpenRouter.

OpenRouter is used when ``LLM_PROVIDER=openrouter``, or when ``OPENROUTER_API_KEY``
is set and ``LLM_PROVIDER`` is not. Bedrock model IDs (e.g. the
``us.anthropic.claude-sonnet-4-5-20250929-v1:0`` defaults) are translated to their
OpenRouter slug, so CLI defaults and experiment folder names stay unchanged. Any
ID containing a ``/`` is treated as an OpenRouter slug and passed through as is.
"""

import json
import os
import re
from typing import Optional

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Bedrock model ID (without cross-region prefix) -> OpenRouter slug
BEDROCK_TO_OPENROUTER = {
    "anthropic.claude-sonnet-4-20250514-v1:0": "anthropic/claude-sonnet-4",
    "anthropic.claude-sonnet-4-5-20250929-v1:0": "anthropic/claude-sonnet-4.5",
    "anthropic.claude-opus-4-5-20251101-v1:0": "anthropic/claude-opus-4.5",
    "anthropic.claude-opus-4-7": "anthropic/claude-opus-4.7",
    "qwen.qwen3-235b-a22b-2507-v1:0": "qwen/qwen3-235b-a22b-2507",
    "meta.llama3-3-70b-instruct-v1:0": "meta-llama/llama-3.3-70b-instruct",
    "openai.gpt-oss-120b-1:0": "openai/gpt-oss-120b",
}


def get_provider() -> str:
    """Return the active LLM provider: ``"bedrock"`` or ``"openrouter"``."""
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if not provider:
        return "openrouter" if os.environ.get("OPENROUTER_API_KEY") else "bedrock"
    if provider not in ("bedrock", "openrouter"):
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (expected 'bedrock' or 'openrouter')")
    return provider


def use_openrouter() -> bool:
    return get_provider() == "openrouter"


def openrouter_model(model_id: str) -> str:
    """Translate a Bedrock model ID to its OpenRouter slug (slugs pass through)."""
    if "/" in model_id:
        return model_id
    base_id = re.sub(r"^(us|eu|apac|global)\.", "", model_id)
    if base_id not in BEDROCK_TO_OPENROUTER:
        raise ValueError(
            f"No OpenRouter equivalent known for Bedrock model '{model_id}'. "
            f"Pass an OpenRouter model slug instead (e.g. anthropic/claude-sonnet-4.5)."
        )
    return BEDROCK_TO_OPENROUTER[base_id]


def openrouter_base_url() -> str:
    """OpenRouter endpoint; override with OPENROUTER_BASE_URL (e.g. for a proxy)."""
    return os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL


def openrouter_api_key() -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_PROVIDER is 'openrouter' but OPENROUTER_API_KEY is not set")
    return api_key


class LLMClient:
    """Single-turn text completion against Bedrock or OpenRouter.

    Used by the agents that call the LLM directly rather than through a MAS
    framework (red/blue team, injection and defense agents).
    """

    def __init__(self, region_name: str, model_id: str):
        self.model_id = model_id
        if use_openrouter():
            from openai import OpenAI

            self.provider = "openrouter"
            self.api_model = openrouter_model(model_id)
            self.client = OpenAI(base_url=openrouter_base_url(), api_key=openrouter_api_key())
        else:
            import boto3

            self.provider = "bedrock"
            self.api_model = model_id
            self.client = boto3.client(service_name="bedrock-runtime", region_name=region_name)

    def complete(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 1000, temperature: float = 0.0) -> str:
        if self.provider == "openrouter":
            messages = [{"role": "system", "content": system}] if system else []
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                model=self.api_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        response = self.client.invoke_model(modelId=self.api_model, body=json.dumps(body))
        return json.loads(response["body"].read())["content"][0]["text"]
