# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from purpleaudit.utils.logger import log_info, log_separator

from .mas import MultiAgentSystem
from purpleaudit.utils.bedrock_config import BedrockLLMConfig


class MultiAgentSystemBuilder(MultiAgentSystem):
    def __init__(self, environment_name: str, bedrock_config: BedrockLLMConfig, **kwargs):
        # Factories also use model_id in paths
        super().__init__(environment_name, bedrock_config, **kwargs)


    def run(self, prompt=None):
        """Build a multi-agent system"""
        
        log_info("Starting environment generation...")
        
        results = []
        prompts = [prompt] if prompt else self.prompts["prompts"]
        for prompt in prompts:
            log_separator(f"GENERATE: {prompt}")
        
            self.load() # Need a fresh state for each scenario
            res = super().run(
                prompt=prompt,
            )
            results.append(res)
        
        return results
