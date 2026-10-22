# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import sys
from typing import Dict, Any

import boto3

from .logger import log_success, log_error, log_info
from .llm_provider import openrouter_api_key, openrouter_base_url, openrouter_model, use_openrouter

# (1,000 input tokens; 1,000 output tokens) https://aws.amazon.com/bedrock/pricing/
BEDROCK_PRICING = {
    "qwen.qwen3-235b-a22b-2507-v1:0": (0.00022, 0.00088),
    "us.anthropic.claude-opus-4-5-20251101-v1:0": (0.005, 0.025),
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": (0.003, 0.015),
    "us.anthropic.claude-sonnet-4-20250514-v1:0": (0.003, 0.015),
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": (0.003, 0.015),
    "us.meta.llama3-3-70b-instruct-v1:0": (0.00072, 0.00072),
    "openai.gpt-oss-120b-1:0": (0.00015, 0.0006),
}

def reply_text(reply) -> str:
    """Text of an AG2 ``generate_reply()`` result.

    AG2 returns a message dict with the Bedrock client but a plain string with
    OpenAI-compatible APIs (OpenRouter).
    """
    if isinstance(reply, dict):
        return reply.get("content") or ""
    return reply or ""


class BedrockLLMConfig:
    """LLM configuration for AG2 agents.

    Targets AWS Bedrock, or OpenRouter when enabled (see ``llm_provider``).
    """

    def __init__(self, region_name: str, model_id: str):
        self.region_name = region_name
        self.model_id = model_id
        self.temperature = 0.0
        if use_openrouter():
            log_success("Using OpenRouter")
            log_info(f"Using model: {openrouter_model(self.model_id)}")
        else:
            self._initialize_bedrock()

    def _initialize_bedrock(self):
        """Initialize AWS Bedrock client"""
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=self.region_name
            )
            log_success(f"Connected to AWS Bedrock in {self.region_name}")
            log_info(f"Using model: {self.model_id}")
        except Exception as e:
            log_error(f"Failed to connect to AWS Bedrock: {e}")
            sys.exit(1)

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration for AG2 agents"""
        if use_openrouter():
            entry = {
                "model": openrouter_model(self.model_id),
                "api_type": "openai",
                "base_url": openrouter_base_url(),
                "api_key": openrouter_api_key(),
            }
        else:
            entry = {
                "model": self.model_id,
                "api_type": "bedrock",
                "aws_region": self.region_name,
            }
        entry["temperature"] = self.temperature
        entry["price"] = BEDROCK_PRICING.get(self.model_id, (0, 0))
        return {
            "config_list": [entry],
            "cache_seed": None,  # Disable caching for testing
        }

    def get_manager_llm_config(self) -> Dict[str, Any]:
        cfg = self.get_llm_config()
        manager_model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        cfg["config_list"][0]["model"] = openrouter_model(manager_model) if use_openrouter() else manager_model
        return cfg
