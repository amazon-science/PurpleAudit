# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import json
import tempfile
import random
from typing import Dict, List, Any, Callable
import importlib.util
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from botocore.exceptions import ClientError

from purpleaudit.utils.logger import log_error, log_warning, log_info, log_json_conversation
from purpleaudit.utils.bedrock_config import BedrockLLMConfig
from purpleaudit.utils.attack_utils import tool_injection_wrapper
from purpleaudit.utils.defense_utils import defense_tool_wrapper, create_defense_wrapper_from_config
from purpleaudit.utils.retry_utils import bedrock_retry
from purpleaudit.defenses import ToolFilteringDefense

from autogen import ConversableAgent
from autogen.coding import DockerCommandLineCodeExecutor, LocalCommandLineCodeExecutor
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern, RoundRobinPattern


class MultiAgentSystem:
    """Production-ready multi-agent system generator using AG2"""

    # Attack runs can resume the benign conversation right before its first tool call
    supports_replay = True
    
    def __init__(self, environment_name: str, bedrock_config: BedrockLLMConfig, folder="environments", system_prompt_variables: dict={}, defense_config: dict=None, model_id: str="default", max_rounds: int=15):
        self.environment_name = environment_name
        self.bedrock_config = bedrock_config
        self.folder = folder
        self.model_id = model_id
        self.system_prompt_variables = system_prompt_variables  # Store for use in create_agents
        self.defense_config = defense_config  # Optional hardcoded defense config
        self.agents = []
        self.tools_module = None
        self.function_tools = {}
        self.prompts = None
        self.pattern = None
        self.current_prompt = None  # Store current prompt for defense mechanisms
        self.tool_filtering_defense = None  # Store tool filtering defense if enabled
        self.stored_attack_to_inject = None  # Store attack for reloading agents
        self.injection_state = None  # Tracks whether the attack reached a tool output
        self.max_rounds = max_rounds
        
        # Determine effective model_id for path resolution (fallback to default if needed)
        self.effective_model_id = self._resolve_model_id()
        
        # All paths use effective_model_id (which may be 'default' as fallback)
        self.env_path = f"{self.folder}/{self.effective_model_id}/{self.environment_name}/environment.json"
        self.prompts_path = f"{self.folder}/{self.effective_model_id}/{self.environment_name}/prompts.json"
        self.environment_config = self.load_environment(system_prompt_variables)

        random.seed(42)
        

    def _resolve_model_id(self) -> str:
        """
        Resolve the model_id to use for file paths.
        
        For factory generation (folder='factories'):
        - Check if factories/{model_id}/factory/ exists
        - If not, fall back to factories/default/factory/
        
        For regular environments (folder='environments'):
        - Check if environments/{model_id}/{environment_name}/environment.json exists
        - If not, fall back to environments/default (environments/{model_id}/ may
          hold only run outputs, e.g. after a benign pass on a default environment)
        
        This allows model-specific factories/environments to override default.
        
        Returns:
            The model_id to use for constructing file paths
        """
        # Check if model-specific directory exists
        if self.folder == "factories":
            # For factory generation, check factories/{model_id}/factory/
            model_specific_path = f"factories/{self.model_id}/factory"
            
            if os.path.exists(model_specific_path):
                return self.model_id
            
            # Fall back to default factory
            default_path = "factories/default/factory"
            if os.path.exists(default_path):
                log_info(f"Factory directory 'factories/{self.model_id}/factory' not found, using default factory")
                return "default"
        else:
            # For regular environments, check environments/{model_id}/{environment_name}
            model_specific_path = f"{self.folder}/{self.model_id}/{self.environment_name}/environment.json"

            if os.path.exists(model_specific_path):
                return self.model_id

            # Fall back to default if model-specific path doesn't exist
            default_path = f"{self.folder}/default/{self.environment_name}"
            if os.path.exists(default_path):
                log_info(f"Environment directory for model '{self.model_id}' not found, using default")
                return "default"
        
        # If neither exists, return the original model_id (will error later with helpful message)
        return self.model_id
    
    def load(self, attack_to_inject: Callable=None):
        """
        Create the MAS architecture from generated environment.json, tools.py and prompts.py files.
        Can be also called between calls to different prompts to start the MAS with a fresh context.
        """
        # Store the attack if provided for future reloads
        if attack_to_inject is not None:
            self.stored_attack_to_inject = attack_to_inject
        
        # Use stored attack if no new attack provided but we have a stored one
        if attack_to_inject is None and self.stored_attack_to_inject is not None:
            attack_to_inject = self.stored_attack_to_inject
        
        self.create_agents(attack_to_inject)
        self.create_topology()
    
    def load_environment(self, system_prompt_variables: dict={}) -> Dict[str, Any]:
        """Load environment configuration"""
        
        if not os.path.exists(self.env_path):
            model_folder = f"{self.folder}/{self.effective_model_id}"
            if os.path.exists(model_folder):
                available_envs = [d for d in os.listdir(model_folder) if os.path.isdir(f"{model_folder}/{d}")]
            else:
                available_envs = []
            
            fallback_msg = f" (using default factory)" if self.effective_model_id == "default" and self.model_id != "default" else ""
            raise FileNotFoundError(
                f"Environment '{self.environment_name}' not found for model '{self.model_id}'{fallback_msg}.\n"
                f"Path checked: {self.env_path}\n"
                f"Available environments: {', '.join(available_envs) if available_envs else 'None'}"
            )
        
        with open(self.env_path, 'r') as f:
            environment_config = json.load(f)
        
        # Apply variable substitutions to the JSON string representation
        if system_prompt_variables:
            config_str = json.dumps(environment_config)
            for k, v in system_prompt_variables.items():
                config_str = config_str.replace(k, str(v))
            environment_config = json.loads(config_str)
        
        self._load_tools_module()
        self._load_prompts()
        
        return environment_config
    
    def _load_tools_module(self):
        """Load tools module for the environment"""
        if not self.environment_name:
            return
        
        tools_path = f"{self.folder}/{self.effective_model_id}/{self.environment_name}/tools.py"
        
        if not os.path.exists(tools_path):
            log_warning(f"No tools.py found for environment {self.environment_name}")
            return
        
        try:
            spec = importlib.util.spec_from_file_location(
                f"{self.environment_name}_tools", 
                tools_path
            )
            self.tools_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.tools_module)
            self._register_functions()
            
        except Exception as e:
            log_error(f"Failed to load tools module: {e}")
            self.tools_module = None
    
    def _load_prompts(self):
        """Load prompts module for the environment"""
        if not self.environment_name:
            return
        
        if not os.path.exists(self.prompts_path):
            log_warning(f"No prompts.json found for environment {self.environment_name}")
            return
        
        with open(self.prompts_path, 'r', encoding='utf-8') as f:
            self.prompts = json.load(f)
    
    def _register_functions(self):
        """Register functions from tools module for AG2"""
        if not self.tools_module:
            return
        
        for name in dir(self.tools_module):
            obj = getattr(self.tools_module, name)
            if callable(obj) and not name.startswith('_') and hasattr(obj, '__call__'):
                if not (hasattr(obj, '__module__') and 'typing' in str(obj.__module__)):
                        self.function_tools[name] = obj
    
    def _apply_multi_agent_tool_filtering(self) -> Dict[str, List]:
        """
        Apply multi-agent tool filtering defense if enabled.
        
        Returns:
            Dictionary mapping agent roles to their filtered list of tools.
            Returns empty dict if tool filtering is not enabled.
        """
        # Check if tool filtering defense is enabled and configure it
        self.tool_filtering_defense = None
        if self.defense_config and 'defenses' in self.defense_config:
            for defense in self.defense_config['defenses']:
                if isinstance(defense, ToolFilteringDefense):
                    # Configure the tool filtering defense with the bedrock config
                    ToolFilteringDefense.configure(self.bedrock_config)
                    self.tool_filtering_defense = defense
                    log_info("Tool filtering defense enabled and configured")
                    break
        
        # If tool filtering is not enabled or no prompt is available, return empty dict
        if not self.tool_filtering_defense or not self.current_prompt:
            return {}
        
        # Extract prompt string if it's a list
        prompt_str = self.current_prompt
        if isinstance(self.current_prompt, list) and len(self.current_prompt) > 0:
            prompt_str = self.current_prompt[0].get("content", "")
        
        # Prepare all agents information for multi-agent filtering
        agents_info = []
        for agent_config in self.environment_config['agents']:
            agent_name = agent_config['role'].replace(' ', '_').lower().replace('(', '').replace(')', '')
            
            system_message = agent_config['system_prompt']
            
            agents_info.append({
                'role': agent_name,
                'description': agent_config.get('description', ''),
                'system_prompt': system_message,
                'tools': agent_config.get('tools', [])
            })
        
        # Get filtered tools for all agents at once
        filtered_tools_map = self.tool_filtering_defense.filter_tools_multi_agent(
            agents_info=agents_info,
            user_task=prompt_str
        )
        
        return filtered_tools_map
                        
    def create_agents(self, attack_to_inject=None) -> List[ConversableAgent]:
        """Create agents based on environment configuration using AG2"""
        agents = []
        llm_config = self.bedrock_config.get_llm_config()
        
        # Apply multi-agent tool filtering if enabled
        filtered_tools_map = self._apply_multi_agent_tool_filtering()
        
        # Shared state for tool injection - ensures injection only on first tool call
        injection_state = {'is_first_call': True} if attack_to_inject else None
        self.injection_state = injection_state
        
        def is_termination_msg(msg):
            content = msg.get("content", "")
            # Only terminate the whole conversation when manager provides final answer
            if "FINAL ANSWER:" in content:
                return True
            return False
        
        for agent_config in self.environment_config['agents']:
            # Get tools for this agent
            agent_tool_names = agent_config.get('tools', [])
            is_manager = agent_config.get('is_manager', False)
            
            agent_name = agent_config['role'].replace(' ', '_').lower().replace('(', '').replace(')', '')
            system_message = agent_config['system_prompt']
            
            # Apply defense prompt if available: 'defense_prompt' for evolved defenses,
            # 'system_prompt_addition' for hardcoded ones
            if self.defense_config:
                system_message += (self.defense_config.get('defense_prompt')
                                   or self.defense_config.get('system_prompt_addition') or '')
            
            # Add current date
            current_date = datetime.now().strftime("%B %d, %Y")
            system_message += f"\n\nCURRENT DATE: {current_date}"
            
            code_execution_config = agent_config.get('code_execution_config', None)
            if code_execution_config:
                if code_execution_config.get('use_docker', False):
                    try:
                        temp_dir = tempfile.mkdtemp(dir="/tmp")
                        executor = DockerCommandLineCodeExecutor(
                            image="python:3.12-slim",
                            work_dir=temp_dir
                        )
                    except Exception:
                        raise ValueError(f"Docker not running, run it or set `use_docker=false`")
                else:
                    temp_dir = tempfile.mkdtemp(dir="/tmp")
                    executor = LocalCommandLineCodeExecutor(work_dir=temp_dir)
                
                # Create agent with code execution capability
                agent = ConversableAgent(
                    name=agent_name,
                    system_message=system_message,
                    llm_config=llm_config,
                    human_input_mode="NEVER",
                    is_termination_msg=is_termination_msg,
                    code_execution_config={"executor": executor}
                )
            else:
                # Create regular agent without code execution
                agent = ConversableAgent(
                    name=agent_name,
                    system_message=system_message,
                    llm_config=llm_config,
                    human_input_mode="NEVER",
                    is_termination_msg=is_termination_msg,
                )

            # Apply tool filtering if defense is enabled
            if filtered_tools_map and agent_name in filtered_tools_map:
                agent_tool_names = filtered_tools_map[agent_name]
            
            # Get defense wrapper if defense_config is provided
            defense_wrapper = create_defense_wrapper_from_config(self.defense_config)
            
            # Register tools for this agent using the new method
            for tool_entry in agent_tool_names:
                # Handle both string and dict formats for tools
                if isinstance(tool_entry, dict):
                    tool_name = tool_entry.get('name')
                    tool_description = tool_entry.get('description', '')
                else:
                    tool_name = tool_entry
                    tool_description = ''
                
                if tool_name and tool_name in self.function_tools:
                    tool_function = self.function_tools[tool_name]

                    # Apply attack injection if provided
                    if attack_to_inject is not None:
                        tool_function = tool_injection_wrapper(
                            tool_function, 
                            attack_to_inject,
                            shared_state=injection_state
                        )
                    
                    # Apply defense wrapper if provided (wraps the output)
                    if defense_wrapper is not None:
                        tool_function = defense_tool_wrapper(
                            tool_function,
                            [defense_wrapper]
                        )
                    
                    # Use function docstring as fn description
                    description = tool_description
                    # if hasattr(tool_function, '__doc__') and tool_function.__doc__:
                    #     description = tool_function.__doc__.strip()
                    
                    # Register the tool function for both LLM and execution
                    agent.register_for_execution()(tool_function)
                    agent.register_for_llm(description=description)(tool_function)
            
            # Store metadata
            agent._tool_names = agent_tool_names
            agent._description = agent_config['description']
            agent.is_manager = is_manager
            
            agents.append(agent)
        
        self.agents = agents
        return agents
    
    def create_topology(self):
        topology = self.environment_config.get('topology', None)
        
        if topology == 'Supervisor':
            PatternTopology = AutoPattern

            manager_candidates = [a for a in self.agents if a.is_manager]
            if not any(manager_candidates):
                raise ValueError(f"Manager topology requires one manager.")
            initial_agent = manager_candidates[0]

        elif topology == 'Round-robin':
            PatternTopology = RoundRobinPattern

            initial_agent = self.agents[0]

        else:
            raise ValueError(f"Unknown topology '{topology}'")
    
        user_proxy = ConversableAgent(
            name="user_proxy",
            llm_config=False,
            human_input_mode="NEVER",
            default_auto_reply="I cannot provide additional information. "
                "Please forward the request to the @ of the manager.",
        )

        self.pattern = PatternTopology(
            initial_agent=initial_agent,
            agents=self.agents,
            user_agent=user_proxy,
            group_manager_args={
                "llm_config": self.bedrock_config.get_manager_llm_config(),
                "is_termination_msg": lambda x: "FINAL ANSWER:" in (x.get("content", "") or "").upper(),
            },
        )

    @bedrock_retry()
    def _run_with_retry(self, prompt):
        """Run group chat with retry logic for all errors
        
        Retries on any exception including:
        - ClientError: All Bedrock API errors
        - ConnectionError: Network issues
        - TimeoutError: Request timeouts
        - Any other exceptions
        
        Uses constant backoff with 10 second intervals between retries.
        Logs retry attempts to console for visibility.
        """
        # A retry restarts the conversation: the attack must hit its first tool call again
        if self.injection_state is not None:
            self.injection_state['is_first_call'] = True
        result, context, last_agent = initiate_group_chat(
            pattern=self.pattern,
            messages=prompt,
            max_rounds=self.max_rounds,
        )
        return result, context, last_agent

    def run(self, prompt, filepath=None):
        """Run the multi-agent system and return the final result"""
        # Store current prompt for tool filtering
        self.current_prompt = prompt
        
        # If tool filtering is enabled and agents need to be recreated with filtered tools
        if self.tool_filtering_defense:
            # Reload the agents with filtered tools based on the current prompt
            # Will use the stored attack if one was previously injected
            self.load()  # Recreate agents with filtered tools (uses stored attack if any)
        
        # Set the original prompt for prompt_sandwiching defense if it's being used
        if self.defense_config and 'defenses' in self.defense_config:
            for defense in self.defense_config['defenses']:
                if hasattr(defense, 'set_original_prompt'):
                    prompt_str = prompt if isinstance(prompt, str) else prompt[0]["content"] if isinstance(prompt, list) else None
                    defense.set_original_prompt(prompt_str)
        
        try:
            result, context, last_agent = self._run_with_retry(prompt)
        except ClientError as e:
            log_error(f"Failed to run multi-agent system after retries: {e}")
            raise

        # OpenAI-compatible APIs (e.g. OpenRouter) return content=None for tool-call-only messages
        for message in result.chat_history:
            if message.get("content") is None:
                message["content"] = ""

        final_result = result.chat_history[-1].get("content", "")

        # Log conversation if filepath is provided
        if filepath:
            for message in result.chat_history:
                name = message.get("name", "user")
                role = message.get("role", "user")
                content = message.get("content", "")
                log_json_conversation(name, role, content, json_file=filepath)

        cost_metrics = self.get_cost()

        return {
            'final_result': final_result,
            'chat_history': result.chat_history,
            'prompt': prompt,
            # Whether the attack reached a tool output (None when no attack is set)
            'injected': None if self.injection_state is None else not self.injection_state['is_first_call'],
            **cost_metrics,
        }
    
    def get_cost(self):
        """Calculate total cost and usage statistics"""
        total_cost = 0
        usage_summary = {}
        
        for agent in self.agents:
            if hasattr(agent, 'client') and hasattr(agent.client, 'total_usage_summary'):
                summary = agent.client.total_usage_summary or {}
                total_cost += summary.get("total_cost", 0)
                usage_summary[agent.name] = {
                    "total_cost": summary.get("total_cost", 0),
                    "prompt_tokens": summary.get("prompt_tokens", 0),
                    "completion_tokens": summary.get("completion_tokens", 0),
                    "total_tokens": summary.get("total_tokens", 0)
                }
        return {
            'total_cost': total_cost,
            'usage_summary': usage_summary,
        }
