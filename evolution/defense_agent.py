# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Defense Agent for applying safeguards to system prompts.
"""

import json
from typing import Dict, List

from purpleaudit.utils.llm_provider import LLMClient
from purpleaudit.utils.retry_utils import bedrock_retry


class DefenseAgent:
    """
    Agent that applies defense strategies to system prompts.
    
    This agent receives:
    - An original system prompt
    - A defense strategy prompt describing how to apply safeguards
    - Few-shot examples showing the expected behavior
    
    It outputs the safeguarded system prompt.
    """
    
    def __init__(self,
                 region_name: str = "us-west-2",
                 model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"):
        """
        Initialize Defense Agent.
        
        Args:
            region_name: AWS region for Bedrock
            model_id: Bedrock model ID
        """
        self.region_name = region_name
        self.model_id = model_id
        
        self.llm = LLMClient(region_name=self.region_name, model_id=self.model_id)
    
    @bedrock_retry()
    def apply_defense(self,
                     system_prompt: str,
                     defense_prompt: str,
                     few_shot_examples: List[Dict] = None) -> str:
        """
        Apply a defense strategy to a system prompt.
        
        Args:
            system_prompt: The original system prompt
            defense_prompt: Instructions describing how to apply safeguards
            few_shot_examples: List of example transformations
            
        Returns:
            Safeguarded system prompt
        """
        # Build the system prompt for the defense agent
        agent_system_prompt = f"""{defense_prompt}

IMPORTANT: You must output ONLY the safeguarded system prompt. Do not add explanations or commentary.
"""
        
        # Build few-shot examples if provided
        examples_text = ""
        if few_shot_examples:
            examples_text = "\n\nEXAMPLES:\n"
            for i, example in enumerate(few_shot_examples, 1):
                original = example.get('original', '')
                safeguarded = example.get('safeguarded', '')
                examples_text += f"\nExample {i}:\n"
                examples_text += f"Original: {original}\n"
                examples_text += f"Safeguarded: {safeguarded}\n"
        
        # Build the user prompt
        user_prompt = f"""SYSTEM PROMPT TO SAFEGUARD:
{system_prompt}
{examples_text}

Now apply the same safeguarding strategy to the system prompt above. Output ONLY the safeguarded system prompt:"""
        
        try:
            return self.llm.complete(
                user_prompt,
                system=agent_system_prompt,
                max_tokens=3000,
                temperature=0.0,  # Deterministic for consistency
            ).strip()
        
        except Exception as e:
            print(f"Error applying defense: {e}")
            return system_prompt  # Return original if error
