#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
General evaluation script for computing various metrics from attack results.

Usage:
    # Per-attack metrics
    python evaluate.py per-attack --model us.meta.llama3-3-70b-instruct-v1:0
    
    # Per-environment metrics
    python evaluate.py per-env --model us.meta.llama3-3-70b-instruct-v1:0
    
    # Both analyses
    python evaluate.py all --model us.meta.llama3-3-70b-instruct-v1:0 --output results.json
"""

import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from purpleaudit.evolution.utils import get_evolution_path_prefix


def find_attack_files(base_folder: str, model_id: str, iteration: int = 0, evolution_name: str = "default", attacks_filter: Optional[str] = None, defense_id: Optional[str] = None, env_filter: Optional[str] = None, no_rw_attacks: bool = False) -> Dict[str, List[str]]:
    """
    Find all attack discussion files for a given model and iteration.
    
    Args:
        base_folder: Base folder containing environments
        model_id: Model ID
        iteration: Iteration number
        evolution_name: Evolution experiment name
        attacks_filter: Comma-separated list of attack names to filter (None = all attacks)
        defense_id: Defense variant ID (int for evolved, str for hardcoded, None = no defense)
        env_filter: Specific environment name to filter (None = all environments)
        no_rw_attacks: If True, exclude attacks starting with "evolved_rw"
    
    Returns:
        Dict mapping environment names to lists of attack file paths
    """
    model_folder = os.path.join(base_folder, model_id)
    env_attacks = {}
    
    if not os.path.exists(model_folder):
        print(f"Warning: Model folder not found: {model_folder}")
        return env_attacks
    
    # Parse attacks filter
    attacks_set = None
    if attacks_filter:
        attacks_set = set(attacks_filter.split(","))
        print(f"Filtering to attacks: {attacks_set}")
    
    # Iterate through environment directories
    for env_name in os.listdir(model_folder):
        env_path = os.path.join(model_folder, env_name)
        
        # Skip non-directories and shared folder
        if not os.path.isdir(env_path) or env_name.startswith('_'):
            continue
        
        # Filter by specific environment if requested
        if env_filter and env_name != env_filter:
            continue
        
        # Get evolution path prefix
        evolution_path = get_evolution_path_prefix(iteration, evolution_name)
        
        # Look for attack discussions in iterations folder
        # If defense_id is specified, look in defense variant subfolder
        if defense_id is not None:
            # Determine defense folder name based on type
            # If defense_id is numeric string, it's an evolved defense (defense_0, defense_1, etc.)
            # If defense_id is non-numeric string, it's a hardcoded defense (defense_spotlighting, etc.)
            try:
                # Try to parse as int - if successful, it's an evolved defense
                int(defense_id)
                defense_folder = f"defense_{defense_id}"
            except ValueError:
                # It's a hardcoded defense name
                defense_folder = f"defense_{defense_id}"
            
            attack_dir = os.path.join(env_path, evolution_path + "iterations", f"iter_{iteration}", "defenses", defense_folder, "discussions", "attack")
        else:
            attack_dir = os.path.join(env_path, evolution_path + "iterations", f"iter_{iteration}", "discussions", "attack")
        
        if not os.path.exists(attack_dir):
            continue
        
        attack_files = []
        
        # Iterate through attack subdirectories (prompt names)
        for prompt_name in os.listdir(attack_dir):
            prompt_path = os.path.join(attack_dir, prompt_name)
            if not os.path.isdir(prompt_path):
                continue
            
            # Iterate through vector subdirectories (tool, prompt, etc.)
            for vector in os.listdir(prompt_path):
                vector_path = os.path.join(prompt_path, vector)
                if not os.path.isdir(vector_path):
                    continue
                
                # Collect attack JSON files (with optional filtering)
                for attack_file in os.listdir(vector_path):
                    if attack_file.endswith('.json'):
                        attack_name = attack_file.replace('.json', '')
                        
                        if no_rw_attacks and attack_name.startswith('evolved_rw'):
                            continue
                        
                        # If attacks filter is provided, check if this attack matches
                        if attacks_set:
                            if attack_name not in attacks_set:
                                continue
                        attack_files.append(os.path.join(vector_path, attack_file))
        
        if attack_files:
            env_attacks[env_name] = attack_files
    
    return env_attacks


def compute_per_attack_metrics(attack_files: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute ASR and Utility for each attack type.
    
    Returns:
        Dict mapping attack names to {'asr': float, 'utility': float, 'count': int}
    """
    attack_metrics = defaultdict(lambda: {'successes': 0, 'total': 0, 'task_success_sum': 0.0})
    
    for file_path in attack_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Results is a list - iterate through each result
            results = data.get('results', [])
            if not isinstance(results, list):
                results = [results]
            
            for result in results:
                attack_name = result.get('attack_name', 'unknown')
                attack_success = result.get('attack_success', False)
                task_success = result.get('task_success', 0.0)
                
                attack_metrics[attack_name]['total'] += 1
                if attack_success:
                    attack_metrics[attack_name]['successes'] += 1
                attack_metrics[attack_name]['task_success_sum'] += (1.0 if task_success else 0.0)
            
        except Exception as e:
            print(f"Warning: Error reading {file_path}: {e}")
            continue
    
    # Compute ASR and average utility
    results = {}
    for attack_name, metrics in attack_metrics.items():
        total = metrics['total']
        if total > 0:
            asr = metrics['successes'] / total
            avg_utility = metrics['task_success_sum'] / total
            results[attack_name] = {
                'asr': asr,
                'utility': avg_utility,
                'count': total
            }
    
    return results


def compute_per_environment_metrics(env_attacks: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """
    Compute ASR and Utility for each environment.
    
    Returns:
        Dict mapping environment names to {'asr': float, 'utility': float, 'count': int}
    """
    env_metrics = {}
    
    for env_name, attack_files in env_attacks.items():
        total = 0
        successes = 0
        task_success_sum = 0.0
        
        for file_path in attack_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                results = data.get('results', [])
                if not isinstance(results, list):
                    results = [results]
                
                for result in results:
                    attack_success = result.get('attack_success', False)
                    task_success = result.get('task_success', 0.0)
                    
                    total += 1
                    if attack_success:
                        successes += 1
                    task_success_sum += (1.0 if task_success else 0.0)
                
            except Exception as e:
                print(f"Warning: Error reading {file_path}: {e}")
                continue
        
        if total > 0:
            env_metrics[env_name] = {
                'asr': successes / total,
                'utility': task_success_sum / total,
                'count': total
            }
    
    return env_metrics


def analyze_per_attack(env_attacks: Dict[str, List[str]], model_id: str) -> Dict[str, Dict[str, float]]:
    """Analyze ASR and Utility per attack averaged across environments."""
    print(f"\n{'='*80}")
    print(f"Per-Attack Analysis")
    print(f"Model: {model_id}")
    print(f"{'='*80}\n")
    
    # Compute metrics per environment
    env_metrics = {}
    for env_name, attack_files in env_attacks.items():
        env_metrics[env_name] = compute_per_attack_metrics(attack_files)
    
    # Average across environments
    all_attacks = set()
    for metrics in env_metrics.values():
        all_attacks.update(metrics.keys())
    
    averaged = {}
    for attack_name in all_attacks:
        asr_values = []
        utility_values = []
        total_count = 0
        
        for env_name, metrics in env_metrics.items():
            if attack_name in metrics:
                asr_values.append(metrics[attack_name]['asr'])
                utility_values.append(metrics[attack_name]['utility'])
                total_count += metrics[attack_name]['count']
        
        if asr_values:
            averaged[attack_name] = {
                'asr': sum(asr_values) / len(asr_values),
                'utility': sum(utility_values) / len(utility_values),
                'num_envs': len(asr_values),
                'total_count': total_count
            }
    
    # Print results
    sorted_attacks = sorted(averaged.items(), key=lambda x: x[1]['asr'], reverse=True)
    
    print(f"{'Attack Name':<40} {'ASR':<10} {'Utility':<10} {'Envs':<8} {'Total':<8}")
    print(f"{'-'*80}")
    
    for attack_name, metrics in sorted_attacks:
        asr_pct = metrics['asr'] * 100
        utility_pct = metrics['utility'] * 100
        num_envs = metrics['num_envs']
        total_count = metrics['total_count']
        
        print(f"{attack_name:<40} {asr_pct:>6.2f}%   {utility_pct:>6.2f}%   {num_envs:<8} {total_count:<8}")
    
    if sorted_attacks:
        avg_asr = sum(m['asr'] for _, m in sorted_attacks) / len(sorted_attacks)
        avg_utility = sum(m['utility'] for _, m in sorted_attacks) / len(sorted_attacks)
        
        print(f"{'-'*80}")
        print(f"{'AVERAGE':<40} {avg_asr*100:>6.2f}%   {avg_utility*100:>6.2f}%")
        print(f"\nTotal unique attacks: {len(sorted_attacks)}")
        print(f"Total attack instances: {sum(m['total_count'] for _, m in sorted_attacks)}")
    
    return averaged


def analyze_per_environment(env_attacks: Dict[str, List[str]], model_id: str) -> Dict[str, Dict[str, float]]:
    """Analyze ASR and Utility per environment."""
    print(f"\n{'='*80}")
    print(f"Per-Environment Analysis")
    print(f"Model: {model_id}")
    print(f"{'='*80}\n")
    
    env_metrics = compute_per_environment_metrics(env_attacks)
    
    # Sort by ASR descending
    sorted_envs = sorted(env_metrics.items(), key=lambda x: x[1]['asr'], reverse=True)
    
    print(f"{'Environment':<40} {'ASR':<10} {'Utility':<10} {'Total':<8}")
    print(f"{'-'*80}")
    
    for env_name, metrics in sorted_envs:
        asr_pct = metrics['asr'] * 100
        utility_pct = metrics['utility'] * 100
        total_count = metrics['count']
        
        print(f"{env_name:<40} {asr_pct:>6.2f}%   {utility_pct:>6.2f}%   {total_count:<8}")
    
    if sorted_envs:
        avg_asr = sum(m['asr'] for _, m in sorted_envs) / len(sorted_envs)
        avg_utility = sum(m['utility'] for _, m in sorted_envs) / len(sorted_envs)
        
        print(f"{'-'*80}")
        print(f"{'AVERAGE':<40} {avg_asr*100:>6.2f}%   {avg_utility*100:>6.2f}%")
        print(f"\nTotal environments: {len(sorted_envs)}")
        print(f"Total attack instances: {sum(m['count'] for _, m in sorted_envs)}")
    
    return env_metrics


def save_results_json(results: Dict, model_id: str, output_file: str):
    """Save results to JSON file."""
    output = {
        'model': model_id,
        **results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate attack results with various metrics')
    parser.add_argument('analysis', choices=['per-attack', 'per-env', 'all'],
                       help='Type of analysis to perform')
    parser.add_argument('--model', required=True,
                       help='Model ID (e.g., us.meta.llama3-3-70b-instruct-v1:0)')
    parser.add_argument('--folder', default='environments',
                       help='Base folder containing environments (default: environments)')
    parser.add_argument('--iteration', type=int, default=0,
                       help='Iteration number (default: 0 for baseline)')
    parser.add_argument('--evolution-name', default='default',
                       help='Evolution experiment name (default: default)')
    parser.add_argument('--attacks', default='',
                       help='Comma-separated list of attack names to filter (optional)')
    parser.add_argument('--defense-id', type=str, default=None,
                       help='Defense variant ID (int for evolved e.g. 0, 1, 2 or str for hardcoded e.g. spotlighting)')
    parser.add_argument('--env', type=str, default=None,
                       help='Specific environment name to evaluate (optional, default: all environments)')
    parser.add_argument('--no-rw-attacks', action='store_true',
                       help='Exclude real-world evolved attacks (those starting with "evolved_rw")')
    parser.add_argument('--output', help='Output JSON file (optional)')
    
    args = parser.parse_args()
    
    # Find all attack files
    from purpleaudit.evolution.utils import get_evolution_path_prefix
    evolution_path = get_evolution_path_prefix(args.iteration, args.evolution_name)
    
    # Build path description
    defense_desc = f", defense_{args.defense_id}" if args.defense_id is not None else ""
    evolution_desc = f" (evolution: {args.evolution_name}, path: {evolution_path}{defense_desc})" if args.evolution_name != "default" or args.defense_id is not None else ""
    
    # Show the path we're searching
    if args.defense_id is not None:
        search_path = f"{args.folder}/{args.model}/*/{evolution_path}iterations/iter_{args.iteration}/defenses/defense_{args.defense_id}/discussions/attack"
    else:
        search_path = f"{args.folder}/{args.model}/*/{evolution_path}iterations/iter_{args.iteration}/discussions/attack"
    
    # Add environment filter info to search description
    env_desc = f" for environment '{args.env}'" if args.env else ""
    print(f"Searching for attacks in {search_path}{evolution_desc}{env_desc}...")
    if args.no_rw_attacks:
        print("Excluding real-world attacks (evolved_rw*)")
    env_attacks = find_attack_files(args.folder, args.model, args.iteration, args.evolution_name, args.attacks, args.defense_id, args.env, args.no_rw_attacks)
    
    if not env_attacks:
        if args.env:
            print(f"\nNo attack files found for model {args.model} and environment {args.env}")
        else:
            print(f"\nNo attack files found for model {args.model}")
        if args.defense_id is not None:
            env_path = f"/{args.env}" if args.env else "/<env>"
            print(f"Check if the path exists: {args.folder}/{args.model}{env_path}/{evolution_path}iterations/iter_{args.iteration}/defenses/defense_{args.defense_id}/discussions/attack/")
        else:
            env_path = f"/{args.env}" if args.env else "/<env>"
            print(f"Check if the path exists: {args.folder}/{args.model}{env_path}/{evolution_path}iterations/iter_{args.iteration}/discussions/attack/")
        return
    
    print(f"\nFound {len(env_attacks)} environments with attacks")
    for env_name, files in env_attacks.items():
        print(f"  - {env_name}: {len(files)} attack files")
    
    # Run requested analyses
    results = {}
    
    if args.analysis in ['per-attack', 'all']:
        per_attack_results = analyze_per_attack(env_attacks, args.model)
        results['per_attack'] = per_attack_results
    
    if args.analysis in ['per-env', 'all']:
        per_env_results = analyze_per_environment(env_attacks, args.model)
        results['per_environment'] = per_env_results
    
    # Save to JSON if requested
    if args.output and results:
        save_results_json(results, args.model, args.output)


if __name__ == '__main__':
    main()
