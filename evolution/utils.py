# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Utility functions for the evolution module.
"""

import os
import json
import glob
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime


def get_evolution_path_prefix(iteration: int, evolution_name: str = "default") -> str:
    """
    Get the evolution path prefix based on iteration.
    
    For baseline (iteration 0), returns empty string (shared across all evolutions).
    For iterations > 0, returns evolution_name subfolder path.
    
    Args:
        iteration: Iteration number
        evolution_name: Name of the evolution experiment (default: "default")
        
    Returns:
        Path prefix string (either "" or "{evolution_name}/")
    """
    if iteration == 0:
        return ""
    else:
        return f"{evolution_name}/"


def get_evolution_path(folder: str, iteration: int, filename: str, evolution_name: str = "default") -> str:
    """
    Get the path for evolution files in the shared _evolution folder.
    
    This folder is model-independent and stores evolved attacks/defenses
    that can be used across different models. Evolution experiments are organized
    by evolution_name, allowing multiple parallel evolution strategies.
    
    Args:
        folder: Base folder (e.g., 'environments')
        iteration: Iteration number
        filename: Name of file (e.g., 'attacks.json', 'defenses.json')
        evolution_name: Name of the evolution experiment (default: "default")
        
    Returns:
        Full path to the evolution file
    """
    return os.path.join(folder, "_evolution", evolution_name, "iterations", f"iter_{iteration}", filename)

def load_evolution_file(folder: str, iteration: int, filename: str, 
                       model_id: str = None, fallback_to_legacy: bool = True,
                       evolution_name: str = "default") -> Optional[List[Dict]]:
    """
    Load an evolution file (attacks.json or defenses.json) from shared folder.
    
    Checks new location first, then falls back to legacy location if needed.
    
    Args:
        folder: Base folder
        iteration: Iteration number
        filename: Name of file to load
        model_id: Model ID (required for legacy fallback)
        fallback_to_legacy: Whether to check legacy location if new location not found
        evolution_name: Name of the evolution experiment (default: "default")
        
    Returns:
        List of dictionaries from the file, or None if not found
    """
    # Try new location first with evolution_name
    new_path = get_evolution_path(folder, iteration, filename, evolution_name)
    if os.path.exists(new_path):
        with open(new_path, 'r') as f:
            return json.load(f)
    
    return None


def save_evolution_file(folder: str, iteration: int, filename: str, data: List[Dict],
                       evolution_name: str = "default") -> str:
    """
    Save an evolution file to the shared _evolution folder.
    
    Args:
        folder: Base folder
        iteration: Iteration number
        filename: Name of file to save
        data: List of dictionaries to save
        evolution_name: Name of the evolution experiment (default: "default")
        
    Returns:
        Path where file was saved
    """
    filepath = get_evolution_path(folder, iteration, filename, evolution_name)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return filepath

def load_attack_results(folder: str, environments: List[str] = None, model_id: str = "default", iteration: int = None, evolution_name: str = "default") -> Dict[str, List[Dict]]:
    """
    Load attack results from all environments.
    
    Args:
        folder: Base folder containing environments (e.g., 'environments', 'old_environments')
        environments: List of specific environments to load, or None for all
        model_id: Model ID subdirectory within folder
        iteration: Iteration number to filter results (None for all iterations)
        evolution_name: Name of the evolution experiment (default: "default")
        
    Returns:
        Dictionary mapping environment names to list of attack results
    """
    results_by_env = {}
    
    model_folder = os.path.join(folder, model_id)
    if not os.path.exists(model_folder):
        return results_by_env
    
    # Get list of environments
    if environments is None:
        environments = [d for d in os.listdir(model_folder) 
                       if os.path.isdir(os.path.join(model_folder, d)) and not d.startswith('_')]
    
    for env_name in environments:
        env_path = os.path.join(model_folder, env_name)
        
        # All results are stored in iteration-specific directories
        if iteration is None:
            raise ValueError("iteration parameter is required - all results are stored in iterations/iter_X/ directories")
        
        # Get evolution path prefix (empty for iter_0, evolution_name/ for iter_1+)
        evolution_path = get_evolution_path_prefix(iteration, evolution_name)
        # Build the base iteration path
        iter_base_path = os.path.join(env_path, evolution_path + "iterations", f"iter_{iteration}")
        
        env_results = []
        
        # Try to load from defense directories first (blue-team-only mode)
        defenses_dir = os.path.join(iter_base_path, "defenses")
        if os.path.exists(defenses_dir):
            # In blue-team-only mode, attacks are in defense-specific directories
            for defense_item in os.listdir(defenses_dir):
                defense_path = os.path.join(defenses_dir, defense_item)
                # Only evolved defenses (defense_0, defense_1, ...): hardcoded baselines such as
                # defense_spotlighting are evaluations, not part of the co-evolution
                if os.path.isdir(defense_path) and defense_item.startswith('defense_') and defense_item[len('defense_'):].isdigit():
                    attack_dir = os.path.join(defense_path, "discussions", "attack")
                    if os.path.exists(attack_dir):
                        env_results.extend(_load_attack_files_from_dir(attack_dir, env_name))
        
        # If no results found in defense directories, try the standard location (red-team-only mode)
        if not env_results:
            attack_dir = os.path.join(iter_base_path, "discussions", "attack")
            if os.path.exists(attack_dir):
                env_results.extend(_load_attack_files_from_dir(attack_dir, env_name))
        
        if env_results:
            results_by_env[env_name] = env_results
    
    return results_by_env


def _load_attack_files_from_dir(attack_dir: str, env_name: str) -> List[Dict]:
    """
    Helper function to load attack result files from a directory.
    
    Args:
        attack_dir: Directory containing attack result files
        env_name: Environment name to add to results
        
    Returns:
        List of attack results with metadata
    """
    results = []
    
    # Walk through attack directory structure
    for root, dirs, files in os.walk(attack_dir):
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        
                        # Extract messages from the discussion
                        messages = data.get('messages', [])
                        
                        # Extract results and attach messages to each
                        if 'results' in data:
                            for result in data['results']:
                                result['environment'] = env_name
                                result['filepath'] = filepath
                                # Attach the conversation messages to the result
                                result['messages'] = messages
                                results.append(result)
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
    
    return results


def compute_asr_by_attack(results: List[Dict]) -> Dict[str, Tuple[float, float, int, int]]:
    """
    Compute Attack Success Rate (ASR) and Utility for each attack type.
    
    Args:
        results: List of attack result dictionaries
        
    Returns:
        Dictionary mapping attack names to (ASR, Utility, num_success, total)
        where:
        - ASR: Attack Success Rate (% of attacks that succeeded)
        - Utility: Task completion rate (% of tasks that succeeded)
        - num_success: Number of successful attacks
        - total: Total number of attempts
    """
    attack_stats = defaultdict(lambda: {'attack_success': 0, 'task_success': 0, 'total': 0})
    
    for result in results:
        attack_name = result.get('attack_name', 'unknown')
        attack_success = result.get('attack_success', False)
        task_success = result.get('task_success', False)
        
        attack_stats[attack_name]['total'] += 1
        if attack_success:
            attack_stats[attack_name]['attack_success'] += 1
        if task_success:
            attack_stats[attack_name]['task_success'] += 1
    
    # Calculate ASR and Utility
    asr_by_attack = {}
    for attack_name, stats in attack_stats.items():
        total = stats['total']
        attack_success = stats['attack_success']
        task_success = stats['task_success']
        
        asr = attack_success / total if total > 0 else 0.0
        utility = task_success / total if total > 0 else 0.0
        
        asr_by_attack[attack_name] = (asr, utility, attack_success, total)
    
    return asr_by_attack


def get_attack_examples(results: List[Dict], 
                       attack_name: str, 
                       success_only: bool = False) -> List[Dict]:
    """
    Get examples for a specific attack.
    
    Args:
        results: List of attack result dictionaries
        attack_name: Name of the attack to filter
        success_only: If True, only return successful attacks
        
    Returns:
        List of attack examples
    """
    examples = []
    for result in results:
        if result.get('attack_name') == attack_name:
            if success_only and not result.get('attack_success', False):
                continue
            examples.append(result)
    return examples


def extract_system_prompts(env_folder: str, env_name: str, model_id: str = "default") -> Dict[str, str]:
    """
    Extract system prompts for all agents in an environment.
    
    Args:
        env_folder: Base folder containing environments
        env_name: Name of the environment
        model_id: Model ID subdirectory within folder
        
    Returns:
        Dictionary mapping agent names to system prompts
    """
    env_path = os.path.join(env_folder, model_id, env_name, "environment.json")
    if not os.path.exists(env_path):
        # Same fallback as MultiAgentSystem._resolve_model_id
        env_path = os.path.join(env_folder, "default", env_name, "environment.json")

    env_config = {}
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            env_config = json.load(f)

    if 'agents' not in env_config:
        # Bring-your-own MAS: prompts recorded at runtime through ctx.system_prompt()
        agents_path = os.path.join(env_folder, model_id, env_name, "agents.json")
        if not os.path.exists(agents_path):
            return {}
        with open(agents_path, 'r') as f:
            return {name: info.get('system_prompt', '') for name, info in json.load(f).items()}

    system_prompts = {}
    for agent in env_config.get('agents', []):
        agent_name = agent.get('role', '').replace(' ', '_').lower()
        system_prompt = agent.get('system_prompt', '')
        system_prompts[agent_name] = system_prompt
    
    return system_prompts




def get_latest_iteration(folder: str, env_name: str, model_id: str = "default", 
                         evolution_name: str = "default") -> int:
    """
    Get the latest iteration number for an environment.
    
    Args:
        folder: Base folder for environments
        env_name: Environment name
        model_id: Model ID subdirectory within folder
        evolution_name: Name of the evolution experiment (default: "default")
        
    Returns:
        Latest iteration number, or 0 if no iterations exist
    """
    env_base = os.path.join(folder, model_id, env_name)
    
    if not os.path.exists(env_base):
        return 0
    
    iter_nums = []
    
    # Check baseline iterations (always in root iterations/ folder)
    iter_base = os.path.join(env_base, "iterations")
    if os.path.exists(iter_base):
        iter_dirs = glob.glob(os.path.join(iter_base, "iter_*"))
        for iter_dir in iter_dirs:
            try:
                num = int(os.path.basename(iter_dir).split('_')[1])
                if num == 0:  # Baseline is always iteration 0
                    iter_nums.append(num)
            except (IndexError, ValueError):
                continue
    
    # Check evolution-specific iterations (in evolution_name/ subfolder)
    if evolution_name != "default":
        evolution_iter_base = os.path.join(env_base, evolution_name, "iterations")
    else:
        evolution_iter_base = os.path.join(env_base, "iterations")
    
    if os.path.exists(evolution_iter_base):
        iter_dirs = glob.glob(os.path.join(evolution_iter_base, "iter_*"))
        for iter_dir in iter_dirs:
            try:
                num = int(os.path.basename(iter_dir).split('_')[1])
                if num > 0:  # Evolution iterations are > 0
                    iter_nums.append(num)
            except (IndexError, ValueError):
                continue
    
    return max(iter_nums) if iter_nums else 0


def compute_metrics_from_attack_files(folder: str, model_id: str, env: str, 
                                       iteration: int = 0, 
                                       defense_id: Optional[int] = None,
                                       evolution_name: str = "default") -> Optional[Dict[str, float]]:
    """
    Compute ASR and Utility directly from attack result JSON files.
    
    This recursively scans the attack directory for JSON files and extracts
    attack_success and task_success from the results array.
    
    Args:
        folder: Base folder containing environments
        model_id: Model ID subdirectory
        env: Environment name
        iteration: Iteration number
        defense_id: Optional defense ID for defense-specific metrics
        evolution_name: Evolution experiment name
        
    Returns:
        Dict with 'asr', 'utility', and 'count', or None if no results found
    """
    # Build potential attack directory paths
    attack_dirs_to_check = []
    
    if defense_id is not None:
        attack_dirs_to_check.extend([
            # Path with evolution name
            os.path.join(folder, model_id, env, evolution_name, "iterations", 
                        f"iter_{iteration}", "defenses", f"defense_{defense_id}", "discussions", "attack"),
            # Path without evolution name
            os.path.join(folder, model_id, env, "iterations", 
                        f"iter_{iteration}", "defenses", f"defense_{defense_id}", "discussions", "attack"),
        ])
    else:
        attack_dirs_to_check.extend([
            # Path with evolution name
            os.path.join(folder, model_id, env, evolution_name, "iterations", 
                        f"iter_{iteration}", "discussions", "attack"),
            # Path without evolution name
            os.path.join(folder, model_id, env, "iterations", 
                        f"iter_{iteration}", "discussions", "attack"),
        ])
    
    # Find the first existing attack directory
    attack_dir = None
    for dir_path in attack_dirs_to_check:
        if os.path.exists(dir_path):
            attack_dir = dir_path
            break
    
    if attack_dir is None:
        return None
    
    # Recursively find all JSON files and extract metrics
    total = 0
    successes = 0
    task_success_sum = 0.0
    
    for root, dirs, files in os.walk(attack_dir):
        for filename in files:
            if filename.endswith('.json'):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # Handle both formats: direct result or nested in results array
                    if isinstance(data, dict):
                        if 'results' in data and isinstance(data['results'], list):
                            results = data['results']
                        elif 'attack_success' in data or 'task_success' in data:
                            results = [data]
                        else:
                            continue
                        
                        for result in results:
                            attack_success = result.get('attack_success', False)
                            task_success = result.get('task_success', False)
                            
                            total += 1
                            if attack_success:
                                successes += 1
                            if task_success:
                                task_success_sum += 1.0
                except Exception as e:
                    continue
    
    if total > 0:
        return {
            'asr': successes / total,
            'utility': task_success_sum / total,
            'count': total
        }
    
    return None
