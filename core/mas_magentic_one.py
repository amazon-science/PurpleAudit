# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
MagenticOne MultiAgentSystem implementation for AssistantBench and GAIA benchmarks.
This is a separate implementation that uses the MagenticOne framework from autogen_ext.
"""

import os
import json
import asyncio
import copy
from typing import Dict, List, Any, Callable
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from purpleaudit.utils.logger import log_error, log_warning, log_info
from purpleaudit.utils.bedrock_config import BedrockLLMConfig
from purpleaudit.utils.llm_provider import openrouter_api_key, openrouter_base_url, openrouter_model, use_openrouter
from purpleaudit.utils.retry_utils import bedrock_retry
from purpleaudit.defenses import ToolFilteringDefense

# MagenticOne specific imports
from autogen_core.models import ModelFamily, ModelInfo
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.anthropic import AnthropicBedrockChatCompletionClient, BedrockInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor


class MagenticOneMultiAgentSystem:
    """MagenticOne-based multi-agent system for benchmark datasets"""
    
    def __init__(self, environment_name: str, bedrock_config: BedrockLLMConfig, 
                 folder="environments", 
                 defense_config: dict=None, 
                 model_id: str="default", per_task_timeout: int=600,
                 max_turns: int=None):
        """
        Initialize MagenticOne MAS
        
        Args:
            environment_name: Name of the environment (assistant_bench or gaia)
            bedrock_config: Bedrock configuration for LLM
            folder: Base folder for environments
            system_prompt_variables: Variables for system prompts (unused in MagenticOne)
            defense_config: Defense config (limited support in MagenticOne)
            model_id: Model ID for organizing experiments
            per_task_timeout: Timeout per task in seconds
            max_turns: Maximum number of turns for the conversation (default: 20)
        """
        self.environment_name = environment_name
        self.bedrock_config = bedrock_config
        self.folder = folder
        self.model_id = model_id
        self.defense_config = defense_config
        self.per_task_timeout = per_task_timeout
        self.max_turns = max_turns or 20  # Store max_turns with default of 20
        
        # Check for tool filtering defense (limited support in MagenticOne)
        self._check_tool_filtering_defense()
        
        # MagenticOne team will be created on load()
        self.team = None
        self.prompts = {}
        self.attack_to_inject = None
        
        # Load dataset-specific configuration
        self._load_dataset_config()
        
        # Prompts will be loaded in load_prompts() method based on mode
    
    def _check_tool_filtering_defense(self):
        """Check if tool filtering defense is enabled and log warning about limited support"""
        if self.defense_config and 'defenses' in self.defense_config:
            for defense in self.defense_config['defenses']:
                if isinstance(defense, ToolFilteringDefense):
                    raise NotImplementedError(
                        "Tool filtering defense has limited support in MagenticOne. "
                        "MagenticOne uses a pre-built team structure with fixed tool assignments. "
                        "The defense will not be able to dynamically filter tools per task."
                    )
    
    def _load_dataset_config(self):
        """Load configuration for the specific dataset"""
        if self.environment_name == "assistant_bench":
            # AssistantBench specific config
            self.dataset_name = "AssistantBench/AssistantBench"
            self.dataset_split = "validation"
            self.task_field = "task"
            self.answer_field = "answer"
            self.id_field = "id"
            self.level_field = None
        elif self.environment_name == "gaia":
            # GAIA specific config
            self.dataset_name = "gaia-benchmark/GAIA"
            self.dataset_config = "2023_level1"  # Can be parameterized
            self.dataset_split = "validation"
            self.task_field = "Question"
            self.answer_field = "Final answer"
            self.id_field = "task_id"
            self.level_field = "Level"
        else:
            raise ValueError(f"Unsupported environment for MagenticOne: {self.environment_name}")
    
    def load(self, attack_to_inject: Callable=None):
        """
        Create the MagenticOne team
        
        Args:
            attack_to_inject: Optional attack function to inject
        """
        self.attack_to_inject = attack_to_inject
        # Team will be created fresh for each run to avoid event loop issues
    
    def load_prompts(self, mode: str):
        """
        Load prompts based on the mode (benign or attack)
        
        Args:
            mode: "benign" to load from online dataset, "attack" to load from benign files
        """
        if mode == "benign":
            # Load from online dataset
            self._load_prompts_from_dataset()
        elif mode == "attack":
            # Load from benign JSON files
            self._load_prompts_from_benign_files()
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'benign' or 'attack'")
    
    def _load_prompts_from_benign_files(self):
        """Load prompts from benign JSON files for attack mode"""
        base_benign_path = f"{self.folder}/{self.model_id}/{self.environment_name}/iterations/iter_0/discussions/benign/"
        
        # Clear existing prompts
        self.prompts = {}
        
        # Load prompts from benign files
        if os.path.exists(base_benign_path):
            for filename in sorted(os.listdir(base_benign_path)):
                if filename.endswith('.json'):
                    prompt_key = filename.replace('.json', '')
                    filepath = os.path.join(base_benign_path, filename)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            benign_data = json.load(f)
                            
                            # Extract task and ground truth from benign file
                            task = benign_data.get('task', '')
                            
                            # Try to get ground truth from results or from the file directly
                            ground_truth = None
                            if 'results' in benign_data and len(benign_data['results']) > 0:
                                ground_truth = benign_data['results'][0].get('ground_truth', '')
                            
                            # Store in prompts dictionary
                            self.prompts[prompt_key] = {
                                'prompt': task,
                                'ground_truth': ground_truth or '',
                                'metadata': {
                                    'loaded_from': filepath,
                                    'dataset': self.environment_name
                                }
                            }
                    except Exception as e:
                        log_warning(f"Failed to load benign file {filepath}: {e}")
                        continue
        else:
            log_warning(f"Benign path does not exist: {base_benign_path}")
        
        log_info(f"Loaded {len(self.prompts)} prompts from benign files for attack mode")
        
    def _load_prompts_from_dataset(self):
        """Load prompts from the benchmark dataset"""
        try:
            from datasets import load_dataset
            
            if self.environment_name == "assistant_bench":
                dataset = load_dataset(self.dataset_name, split=self.dataset_split)
            else:  # gaia
                dataset = load_dataset(
                    self.dataset_name,
                    self.dataset_config,
                    split=self.dataset_split
                )
            
            # Convert dataset to prompts format with simple indexing for file names
            for i, example in enumerate(dataset):
                prompt_key = f"prompt_{i}"
                level = example.get(self.level_field, "")
                self.prompts[prompt_key] = {
                    "prompt": example[self.task_field],
                    "ground_truth": example.get(self.answer_field, ""),
                    "metadata": {
                        "id": example.get(self.id_field, i),
                        "dataset": self.environment_name,
                        "level": level,
                    }
                }
            
            log_info(f"Loaded {len(self.prompts)} prompts from {self.environment_name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
    
    async def _run_async(self, task: Any) -> TaskResult:
        """Run MagenticOne asynchronously with optional attack injection"""
        
        # If task is a string, convert to expected format
        if isinstance(task, str):
            task_text = task
        elif isinstance(task, list):
            # Handle pre-built message list (for attack mode)
            return await self.team.run(task=task)
        else:
            task_text = str(task)
        
        # Apply attack injection if configured
        if self.attack_to_inject:
            # For MagenticOne, we need to inject into the first tool response
            # This is handled differently than standard MAS
            # We'll modify the task to include the injection
            task_text = self.attack_to_inject(task_text)
        
        # Run the team
        if self.per_task_timeout:
            result = await asyncio.wait_for(
                self.team.run(task=task_text),
                timeout=self.per_task_timeout
            )
        else:
            result = await self.team.run(task=task_text)
        
        return result
    
    def _create_team(self, max_turns: int = None):
        """Create a fresh MagenticOne team instance
        
        Args:
            max_turns: Maximum number of turns for the conversation (default: 20)
        """
        if use_openrouter():
            model = openrouter_model(self.bedrock_config.model_id)
            model_client = OpenAIChatCompletionClient(
                model=model,
                base_url=openrouter_base_url(),
                api_key=openrouter_api_key(),
                temperature=self.bedrock_config.temperature,
                model_info=ModelInfo(
                    vision=False,
                    function_calling=True,
                    json_output=False,
                    # A Claude family enables autogen-ext's Claude message fixes (no whitespace-only content)
                    family=ModelFamily.CLAUDE_4_SONNET if "claude" in model else ModelFamily.UNKNOWN,
                    structured_output=False,
                ),
            )
        else:
            model_client = AnthropicBedrockChatCompletionClient(
                model=self.bedrock_config.model_id,
                temperature=self.bedrock_config.temperature,
                model_info=ModelInfo(
                    vision=False,
                    function_calling=True,
                    json_output=False,
                    family="claude",
                    structured_output=False,  # Must be False for Anthropic
                ),
                bedrock_info=BedrockInfo(
                    aws_region=self.bedrock_config.region_name,
                ),
            )
        
        # TODO: use Docker instead. But it does not work for some reason with MagenticOne
        import tempfile
        work_dir = tempfile.mkdtemp(prefix="magentic_one_")
        code_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
        log_info(f"Using local code executor for MagenticOne in {work_dir}")
        
        # Same team as autogen_ext.teams.magentic_one.MagenticOne, built from its
        # parts because MagenticOne does not expose max_turns
        from autogen_agentchat.agents import CodeExecutorAgent
        from autogen_agentchat.teams import MagenticOneGroupChat
        from autogen_ext.agents.file_surfer import FileSurfer
        from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
        from autogen_ext.agents.web_surfer import MultimodalWebSurfer
        agents = [
            FileSurfer("FileSurfer", model_client=model_client),
            MultimodalWebSurfer("WebSurfer", model_client=model_client),
            MagenticOneCoderAgent("Coder", model_client=model_client),
            CodeExecutorAgent("ComputerTerminal", code_executor=code_executor),
        ]
        return MagenticOneGroupChat(agents, model_client=model_client, max_turns=max_turns)
    
    def _run_with_retry(self, prompt):
        """Run async MagenticOne with retry logic for Bedrock API calls
        
        After max retries are exhausted, restarts conversation with only the first message.
        """
        @bedrock_retry()
        def _run_with_bedrock_retry(prompt_to_use):
            # Create a fresh team for each run to avoid event loop issues
            self.team = self._create_team(max_turns=self.max_turns)
            
            # Use nest_asyncio to handle nested event loops
            import nest_asyncio
            nest_asyncio.apply()
            
            # Run with asyncio.run to get a fresh event loop
            return asyncio.run(self._run_async(prompt_to_use))

        try:
            # First attempt with full prompt
            return _run_with_bedrock_retry(prompt)
        except Exception as e:
            # If all retries failed, restart with only the first message
            log_warning(f"All retries exhausted. Restarting conversation with only the first message.")
            
            # Extract the first message
            first_message_only = prompt
            if isinstance(prompt, list) and len(prompt) > 0:
                # Check if it's a list of TextMessage objects or dicts
                from autogen_agentchat.messages import TextMessage
                if isinstance(prompt[0], TextMessage):
                    # Keep only the first TextMessage
                    first_message_only = [prompt[0]]
                else:
                    # It's a list of dicts/other, take only the first one
                    first_message_only = [prompt[0]]
            elif isinstance(prompt, str):
                # If it's a string, keep it as is
                first_message_only = prompt
            
            # Try again with only the first message (with retry logic)
            try:
                return _run_with_bedrock_retry(first_message_only)
            except Exception as final_error:
                log_error(f"Failed even with first message only: {final_error}")
                raise
    
    def run(self, prompt, filepath=None):
        """
        Run the MagenticOne system and return the result
        
        Args:
            prompt: The task/prompt to run
            filepath: Optional filepath to save the discussion
            
        Returns:
            Dictionary with results matching the standard MAS format
        """
        try:
            # Run asynchronously with retry logic
            result = self._run_with_retry(prompt)
            
            # Extract messages and convert to standard format
            messages = []
            for msg in result.messages:
                msg_dict = msg.model_dump()
                # Convert created_at to string if it exists
                if 'created_at' in msg_dict:
                    msg_dict['created_at'] = str(msg_dict['created_at'])
                
                # Map MagenticOne message format to standard format
                content = msg_dict.get('content', '')
                source = msg_dict.get('source', 'Unknown')

                if isinstance(content, list):
                    content = content[0] # remove data if present
                
                # Map source to role for compatibility
                if source == "User":
                    role = "user"
                elif "WebSurfer" in source:
                    role = "tool"
                else:
                    role = "assistant"
                
                messages.append({
                    "role": role,
                    "name": source,
                    "content": content
                })
            
            # Extract final result
            final_result = ""
            if messages:
                # Look for the last non-tool message
                for msg in reversed(messages):
                    if msg["role"] != "tool":
                        final_result = msg["content"]
                        break
            
            prompt_str = prompt if isinstance(prompt, str) else prompt[0].content \
                if (isinstance(prompt, list) and isinstance(prompt[0], TextMessage)) else None

            # Save to file if requested
            if filepath:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                discussion_data = {
                    "messages": messages,
                    "stop_reason": result.stop_reason,
                    "task": prompt_str,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Note: results field will be added by the lab's evaluation logic
                # to avoid duplication
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(discussion_data, f, indent=2, ensure_ascii=False)
                
                log_info(f"Saved MagenticOne discussion to {filepath}")
            
            # Get cost metrics from the team
            cost_metrics = self.get_cost()
            
            # Return in format compatible with standard MAS
            return {
                'final_result': final_result,
                'chat_history': messages,
                'prompt': prompt_str,
                'total_cost': cost_metrics.get('total_cost', 0),
                'usage_summary': cost_metrics.get('usage_summary', {}),
                'stop_reason': result.stop_reason
            }
            
        except asyncio.TimeoutError:
            log_error(f"Task timed out after {self.per_task_timeout}s")
            return {
                'final_result': "Task timed out",
                'chat_history': [],
                'prompt': prompt,
                'total_cost': 0,
                'usage_summary': {},
                'stop_reason': 'timeout'
            }
        except Exception as e:
            log_error(f"Failed to run MagenticOne: {e}")
            raise
    
    def get_cost(self):
        """Get cost metrics from MagenticOne agents"""
        total_cost = 0
        usage_summary = {}
        
        # If team exists, try to get cost information from the model client
        if hasattr(self, 'team') and self.team is not None:
            # MagenticOne uses a shared model client for all agents
            # We need to access it through the team's agents
            try:
                # Access the agents from the team
                if hasattr(self.team, '_agents') and self.team._agents:
                    for agent in self.team._agents:
                        # Check if agent has a model_client with usage tracking
                        if hasattr(agent, '_model_client'):
                            model_client = agent._model_client
                            
                            # For AnthropicBedrockChatCompletionClient, use total_usage() method
                            if hasattr(model_client, 'total_usage'):
                                usage = model_client.total_usage()
                                if usage:
                                    # Calculate cost based on Anthropic Claude 3.5 Sonnet pricing for AWS Bedrock
                                    # Input: $0.003 per 1K tokens, Output: $0.015 per 1K tokens
                                    input_cost = (usage.prompt_tokens / 1000) * 0.003
                                    output_cost = (usage.completion_tokens / 1000) * 0.015
                                    agent_cost = input_cost + output_cost
                                    
                                    agent_name = getattr(agent, 'name', agent.__class__.__name__)
                                    total_cost += agent_cost
                                    usage_summary[agent_name] = {
                                        "total_cost": agent_cost,
                                        "prompt_tokens": usage.prompt_tokens,
                                        "completion_tokens": usage.completion_tokens,
                                        "total_tokens": usage.prompt_tokens + usage.completion_tokens
                                    }
                                    # Since MagenticOne shares the same client, we only need to get it once
                                    break
                            # Fallback to total_usage_summary if available (for other client types)
                            elif hasattr(model_client, 'total_usage_summary'):
                                summary = model_client.total_usage_summary or {}
                                agent_name = getattr(agent, 'name', agent.__class__.__name__)
                                agent_cost = summary.get("total_cost", 0)
                                total_cost += agent_cost
                                usage_summary[agent_name] = {
                                    "total_cost": agent_cost,
                                    "prompt_tokens": summary.get("prompt_tokens", 0),
                                    "completion_tokens": summary.get("completion_tokens", 0),
                                    "total_tokens": summary.get("total_tokens", 0)
                                }
                                # Since MagenticOne shares the same client, we only need to get it once
                                break
            except Exception as e:
                log_warning(f"Could not extract cost metrics from MagenticOne: {e}")
        
        return {
            'total_cost': total_cost,
            'usage_summary': usage_summary
        }
    
    def apply_tool_injection(self, messages: List[Dict], injection: str, flag: str) -> List[TextMessage]:
        """
        Apply injection to the first tool message in a discussion
        
        Args:
            messages: List of message dictionaries from a benign discussion
            injection: The injection text to add
            flag: The flag/answer for the injection
            
        Returns:
            List of TextMessage objects with injection applied
        """
        
        # Find first WebSurfer/tool message
        first_tool_idx = None
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            name = msg.get("name", "")
            if "WebSurfer" in name or role == "tool":
                first_tool_idx = i
                break
        
        if first_tool_idx is None:
            log_warning("No tool message found for injection")
            return []
        
        # Build prefix up to and including the modified tool message
        # Deep copy to avoid modifying original messages
        prefix = copy.deepcopy(messages[:first_tool_idx + 1])
        
        # Apply defense if configured (only for defenses with system prompt additions)
        if self.defense_config:
            defense_addition = self._get_defense_system_prompt_addition()
            if defense_addition:
                # Find the second message (orchestrator guidelines) and append defense
                if len(prefix) > 1 and prefix[1].get("name") == "MagenticOneOrchestrator":
                    prefix[1]["content"] = prefix[1]["content"] + "\n\n" + defense_addition
                    log_info(f"Applied defense to orchestrator guidelines in MagenticOne")
        
        # Inject into the tool message
        tool_msg = prefix[-1]

        # Apply injection using the loaded attack function if available
        if self.attack_to_inject:
            # Use the loaded attack function (could be generated or hardcoded)
            injected_content = self.attack_to_inject("")
        else:
            raise ValueError(f"Attacks not found.")
        
        tool_msg["content"] = injected_content
        
        # Convert to TextMessage objects
        text_messages = []
        for msg in prefix:
            content = msg.get("content", "")
            name = msg.get("name", "")
            
            # Map to source names that MagenticOne expects
            if name == "user" or msg.get("role") == "user":
                source = "User"
            else:
                source = name or "Unknown"
            
            text_messages.append(TextMessage(content=content, source=source))
        
        return text_messages
    
    def _get_defense_system_prompt_addition(self) -> str:
        """
        Get the system prompt addition for the configured defense.
        Only supports defenses with non-empty get_system_prompt_addition().
        
        Returns:
            str: Defense system prompt addition, or empty string if none
        """
        # Check for simplified defense format (from evolved defenses)
        if self.defense_config and 'defense_prompt' in self.defense_config:
            return self.defense_config['defense_prompt']
        
        # Check hardcoded defense (legacy format)
        if self.defense_config and 'defenses' in self.defense_config:
            for defense in self.defense_config['defenses']:
                addition = defense.get_system_prompt_addition()
                if addition:
                    return addition
                else:
                    # Raise error for defenses that don't have system prompt additions
                    from purpleaudit.defenses import ToolFilteringDefense
                    if isinstance(defense, ToolFilteringDefense):
                        raise NotImplementedError(
                            "Tool filtering defense is not supported for MagenticOne. "
                            "MagenticOne has a fixed tool assignment structure."
                        )
                    elif defense.name == "prompt_sandwiching":
                        raise NotImplementedError(
                            "Prompt sandwiching defense is not supported for MagenticOne. "
                            "This defense requires tool output wrapping which is not compatible with MagenticOne's architecture."
                        )
        
        return ""
