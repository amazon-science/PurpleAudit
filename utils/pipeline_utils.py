# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Shared utilities for the clean evolution pipeline.
"""
import os
import json
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


def get_evolution_dir(base_folder: str, evolution_name: str = "default") -> str:
    """Get the evolution directory path."""
    return os.path.join(base_folder, "_evolution", evolution_name)


def get_iteration_dir(base_folder: str, iteration: int, evolution_name: str = "default") -> str:
    """Get the directory path for a specific iteration."""
    return os.path.join(get_evolution_dir(base_folder, evolution_name), "iterations", f"iter_{iteration}")


def get_env_iteration_dir(base_folder: str, model: str, env: str, iteration: int) -> str:
    """Get the environment-specific iteration directory."""
    return os.path.join(base_folder, model, env, "iterations", f"iter_{iteration}")


def load_json_file(filepath: str) -> Optional[Any]:
    """Load a JSON file, returning None if it doesn't exist."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def save_json_file(filepath: str, data: Any) -> None:
    """Save data to a JSON file, creating directories as needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_attacks(base_folder: str, iteration: int, evolution_name: str = "default") -> List[Dict]:
    """Load attacks from a specific iteration."""
    attacks_file = os.path.join(get_iteration_dir(base_folder, iteration, evolution_name), "attacks.json")
    return load_json_file(attacks_file) or []


def save_attacks(base_folder: str, iteration: int, attacks: List[Dict], evolution_name: str = "default") -> None:
    """Save attacks for a specific iteration."""
    attacks_file = os.path.join(get_iteration_dir(base_folder, iteration, evolution_name), "attacks.json")
    save_json_file(attacks_file, attacks)


def load_defenses(base_folder: str, iteration: int, evolution_name: str = "default") -> List[Dict]:
    """Load defenses from a specific iteration."""
    defenses_file = os.path.join(get_iteration_dir(base_folder, iteration, evolution_name), "defenses.json")
    return load_json_file(defenses_file) or []


def save_defenses(base_folder: str, iteration: int, defenses: List[Dict], evolution_name: str = "default") -> None:
    """Save defenses for a specific iteration."""
    defenses_file = os.path.join(get_iteration_dir(base_folder, iteration, evolution_name), "defenses.json")
    save_json_file(defenses_file, defenses)


def load_attack_results(base_folder: str, model: str, env: str, iteration: int, 
                        defense_id: Optional[int] = None, evolution_name: Optional[str] = None) -> List[Dict]:
    """Load attack results from discussion files."""
    results = []
    
    # Evolution name is always part of the path structure
    if not evolution_name:
        evolution_name = "default"
    
    # Build list of potential attack directories to check
    attack_dirs_to_check = []
    
    if defense_id is not None:
        # Defense-specific paths
        attack_dirs_to_check.extend([
            # Path with evolution name
            os.path.join(base_folder, model, env, evolution_name, "iterations", f"iter_{iteration}", 
                        "defenses", f"defense_{defense_id}", "discussions", "attack"),
            # Path without evolution name
            os.path.join(get_env_iteration_dir(base_folder, model, env, iteration), 
                        "defenses", f"defense_{defense_id}", "discussions", "attack"),
        ])
    else:
        # Non-defense paths
        attack_dirs_to_check.extend([
            # Path with evolution name
            os.path.join(base_folder, model, env, evolution_name, "iterations", f"iter_{iteration}", 
                        "discussions", "attack"),
            # Path without evolution name
            os.path.join(get_env_iteration_dir(base_folder, model, env, iteration), 
                        "discussions", "attack"),
        ])
    
    # Find the first existing attack directory
    attack_dir = None
    for dir_path in attack_dirs_to_check:
        if os.path.exists(dir_path):
            attack_dir = dir_path
            break
    
    if attack_dir is None:
        # Log for debugging
        print(f"[DEBUG] Attack directory not found. Checked: {attack_dirs_to_check}")
        return results
    
    # Recursively walk through the attack directory to find all JSON files
    for root, dirs, files in os.walk(attack_dir):
        for filename in files:
            if filename.endswith('.json'):
                filepath = os.path.join(root, filename)
                data = load_json_file(filepath)
                if data:
                    # Handle both old format (direct result) and new format (with results array)
                    if isinstance(data, dict):
                        if 'results' in data and isinstance(data['results'], list):
                            # New format: extract results from the array
                            results.extend(data['results'])
                        elif 'attack_success' in data or 'task_success' in data:
                            # Old format with direct attack result fields
                            results.append(data)
                        # Skip files that don't have expected result structure (e.g., config files)
    
    return results


def compute_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute ASR and utility metrics from attack results."""
    if not results:
        raise RuntimeError("Results are empty.")
    
    successful = sum(1 for r in results if r.get('attack_success', False))
    total = len(results)
    
    # Utility calculation based on task_success (whether the task was completed correctly)
    # task_success indicates whether the agent completed the intended task correctly
    utility_scores = []
    for r in results:
        # Check for task_success first (new format), fall back to benign_success (old format)
        if r.get('task_success', r.get('benign_success', False)):
            utility_scores.append(1.0)
        else:
            utility_scores.append(0.0)
    
    utility = sum(utility_scores) / len(utility_scores) if utility_scores else 0.0
    
    return {
        'asr': successful / total if total > 0 else 0.0,
        'utility': utility
    }


def save_metrics(base_folder: str, model: str, env: str, iteration: int, 
                 defense_id: Optional[int], metrics: Dict[str, float], evolution_name: Optional[str] = None) -> None:
    """Save metrics for an evaluation run."""
    # Evolution name is always part of the path structure
    if not evolution_name:
        evolution_name = "default"
        
    if defense_id is not None:
        metrics_file = os.path.join(base_folder, model, env, evolution_name, "iterations", f"iter_{iteration}",
                                   "defenses", f"defense_{defense_id}", "metrics.json")
    else:
        metrics_file = os.path.join(base_folder, model, env, evolution_name, "iterations", f"iter_{iteration}", "metrics.json")
    
    # Include overall metrics
    full_metrics = {
        'overall': metrics,
        'iteration': iteration,
        'defense_id': defense_id,
        'evolution_name': evolution_name
    }
    
    save_json_file(metrics_file, full_metrics)


def log_info(message: str) -> None:
    """Simple info logging."""
    print(f"[INFO] {message}")


def log_success(message: str) -> None:
    """Simple success logging."""
    print(f"[SUCCESS] {message}")


def log_error(message: str) -> None:
    """Simple error logging."""
    print(f"[ERROR] {message}")


def log_separator(title: str = "") -> None:
    """Print a separator line."""
    if title:
        print(f"\n{'='*10} {title} {'='*10}")
    else:
        print("="*50)
