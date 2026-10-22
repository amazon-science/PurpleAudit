# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import json
from datetime import datetime

from purpleaudit.utils.logger import log_separator, log_error, log_info, Colors
from purpleaudit.evolution.utils import get_evolution_path_prefix


def load_attack_results_from_files(lab):
    """Load all attack results from the file system"""
    results = []
    
    # Check if attack path exists
    if not os.path.exists(lab.attack_path):
        log_error(f"Attack path does not exist: {lab.attack_path}")
        return results
    
    # Iterate through task directories
    for task_name in os.listdir(lab.attack_path):
        task_path = os.path.join(lab.attack_path, task_name)
        if not os.path.isdir(task_path):
            continue
        
        # Iterate through vector directories
        for vector in os.listdir(task_path):
            vector_path = os.path.join(task_path, vector)
            if not os.path.isdir(vector_path):
                continue
            
            # Iterate through attack JSON files
            for attack_file in os.listdir(vector_path):
                if not attack_file.endswith('.json'):
                    continue
                
                attack_name = attack_file.replace('.json', '')
                filepath = os.path.join(vector_path, attack_file)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract result from the JSON structure
                    if 'results' in data and len(data['results']) > 0:
                        result = data['results'][0]
                    else:
                        result = data
                    
                    # Add metadata
                    result['vector'] = vector
                    result['attack_name'] = attack_name
                    result['task_name'] = task_name
                    result['filepath'] = filepath
                    
                    results.append(result)
                except Exception as e:
                    log_error(f"Error loading {filepath}: {e}")
    
    return results

def compute_attack_metrics(results):
    """Compute all metrics from attack results"""
    metrics = {
        'by_attack': {},
        'by_vector': {},
        'overall': {}
    }
    
    # Sort results by attack name first, then by vector
    results = sorted(results, key=lambda d: (d.get("attack_name", "unknown"), d.get("vector", "unknown")))
    
    # Group results by attack name
    attacks_found = set(r.get('attack_name') for r in results if r.get('attack_name'))
    
    for attack_name in sorted(attacks_found):
        attack_results = [r for r in results if r.get('attack_name') == attack_name]
        if not attack_results:
            continue
        
        metrics['by_attack'][attack_name] = {
            'vectors': {},
            'overall': {}
        }
        
        # Group by vector within this attack
        vectors_in_attack = set(r.get('vector') for r in attack_results if r.get('vector'))
        
        for vector in sorted(vectors_in_attack):
            vector_results = [r for r in attack_results if r.get('vector') == vector]
            if not vector_results:
                continue
            
            # Calculate metrics for this vector within this attack
            vector_attack_success = sum(1 for r in vector_results if r.get('attack_success', False))
            vector_task_success = sum(1 for r in vector_results if r.get('task_success', False))
            vector_asr = (vector_attack_success / len(vector_results) * 100) if vector_results else 0
            vector_task_rate = (vector_task_success / len(vector_results) * 100) if vector_results else 0
            
            metrics['by_attack'][attack_name]['vectors'][vector] = {
                'results': vector_results,
                'attack_success_count': vector_attack_success,
                'task_success_count': vector_task_success,
                'total_count': len(vector_results),
                'asr': vector_asr,
                'task_rate': vector_task_rate
            }
        
        # Calculate overall metrics for this attack across all vectors
        attack_attack_success = sum(1 for r in attack_results if r.get('attack_success', False))
        attack_task_success = sum(1 for r in attack_results if r.get('task_success', False))
        attack_asr = (attack_attack_success / len(attack_results) * 100) if attack_results else 0
        attack_task_rate = (attack_task_success / len(attack_results) * 100) if attack_results else 0
        
        metrics['by_attack'][attack_name]['overall'] = {
            'attack_success_count': attack_attack_success,
            'task_success_count': attack_task_success,
            'total_count': len(attack_results),
            'asr': attack_asr,
            'task_rate': attack_task_rate
        }
    
    # Calculate overall statistics across all attacks and vectors
    total_count = len(results)
    task_success_count = sum(1 for r in results if r.get('task_success', False))
    attack_success_count = sum(1 for r in results if r.get('attack_success', False))
    
    task_accuracy = (task_success_count / total_count * 100) if total_count > 0 else 0
    attack_success_rate = (attack_success_count / total_count * 100) if total_count > 0 else 0
    
    metrics['overall'] = {
        'attack_success_count': attack_success_count,
        'task_success_count': task_success_count,
        'total_count': total_count,
        'asr': attack_success_rate,
        'task_rate': task_accuracy
    }
    
    # Calculate vector-level summary across all attacks
    vectors_found = set(r.get('vector') for r in results if r.get('vector'))
    
    for vector in sorted(vectors_found):
        vector_results = [r for r in results if r.get('vector') == vector]
        if vector_results:
            vector_attack_success = sum(1 for r in vector_results if r.get('attack_success', False))
            vector_task_success = sum(1 for r in vector_results if r.get('task_success', False))
            vector_asr = (vector_attack_success / len(vector_results) * 100) if vector_results else 0
            vector_task_rate = (vector_task_success / len(vector_results) * 100) if vector_results else 0
            
            metrics['by_vector'][vector] = {
                'attack_success_count': vector_attack_success,
                'task_success_count': vector_task_success,
                'total_count': len(vector_results),
                'asr': vector_asr,
                'task_rate': vector_task_rate
            }
    
    return metrics


def print_metrics(metrics):
    # Display results
    if not metrics:
        return
    log_separator("ATTACK EVALUATION SUMMARY")
    
    # Calculate total cost across all results
    total_cost = 0
    for attack_name in metrics['by_attack'].keys():
        for vector in metrics['by_attack'][attack_name]['vectors'].keys():
            vector_results = metrics['by_attack'][attack_name]['vectors'][vector]['results']
            for result in vector_results:
                total_cost += result.get('total_cost', 0)
    
    truncate = 60
    
    # Display by attack
    for attack_name in sorted(metrics['by_attack'].keys()):
        attack_metrics = metrics['by_attack'][attack_name]
        
        log_separator(f"ATTACK: {attack_name.upper()}")
        
        # Display by vector within this attack
        for vector in sorted(attack_metrics['vectors'].keys()):
            vector_metrics = attack_metrics['vectors'][vector]
            
            log_info(f"\n{vector.upper()} Vector:")
            log_info("-" * 80)
            
            # Display individual results
            for result in vector_metrics['results']:
                prompt = result["prompt"]
                if isinstance(prompt, list) and len(prompt) > 0:
                    prompt_text = prompt[0].get("content", str(prompt)) if isinstance(prompt[0], dict) else str(prompt[0])
                else:
                    prompt_text = str(prompt)
                
                prompt_preview = prompt_text[:truncate].replace('\n', ' ')
                if len(prompt_text) > truncate:
                    prompt_preview += "..."
                
                task_color = Colors.SUCCESS if result.get('task_success', False) else Colors.ERROR
                attack_color = Colors.SUCCESS if result.get('attack_success', False) else Colors.ERROR
                ground_truth = result.get('ground_truth', 'N/A')
                
                task_status = "✓" if result.get('task_success', False) else "✗"
                attack_status = "✓" if result.get('attack_success', False) else "✗"
                
                log_info(f"{prompt_preview:<63} | {ground_truth[:15]:<15} | Task:{task_color}{task_status}{Colors.RESET} | Attack:{attack_color}{attack_status}{Colors.RESET}")
            
            # Display vector metrics
            log_info(f"  {vector.title()} ASR: {vector_metrics['attack_success_count']}/{vector_metrics['total_count']} ({vector_metrics['asr']:.1f}%) | " \
                f"Task Success: {vector_metrics['task_success_count']}/{vector_metrics['total_count']} ({vector_metrics['task_rate']:.1f}%)")
        
        # Display overall attack metrics
        overall = attack_metrics['overall']
        log_info(f"{attack_name} Overall ASR: {overall['attack_success_count']}/{overall['total_count']} ({overall['asr']:.1f}%)")
        log_info(f"{attack_name} Overall Task Success: {overall['task_success_count']}/{overall['total_count']} ({overall['task_rate']:.1f}%)")
    
    log_separator(log_file=None)
    
    # Display overall statistics
    overall = metrics['overall']
    log_info(f"Overall ASR (All Attacks): {overall['attack_success_count']}/{overall['total_count']} ({overall['asr']:.1f}%)")
    log_info(f"Overall Task Success (Under Attack): {overall['task_success_count']}/{overall['total_count']} ({overall['task_rate']:.1f}%)")
    
    # Display cost summary
    log_separator("COST SUMMARY")
    log_info(f"Total Execution Cost (All Attacks): ${total_cost:.4f}")
    
    # Display cost per attack
    log_info(f"\n{'Attack':<40} {'Total Cost':<20}")
    log_info("-" * 60)
    for attack_name in sorted(metrics['by_attack'].keys()):
        attack_cost = 0
        for vector in metrics['by_attack'][attack_name]['vectors'].keys():
            vector_results = metrics['by_attack'][attack_name]['vectors'][vector]['results']
            for result in vector_results:
                attack_cost += result.get('total_cost', 0)
        log_info(f"{attack_name:<40} ${attack_cost:.4f}")
    log_info("")
    
    # Display vector-level summary
    log_separator("METRICS PER VECTOR")
    log_info(f"{'Vector':<40} {'ASR':<25} {'Task Success':<25}")
    log_info("-" * 70)
    for vector in sorted(metrics['by_vector'].keys()):
        vector_metrics = metrics['by_vector'][vector]
        asr_str = f"{vector_metrics['attack_success_count']}/{vector_metrics['total_count']} ({vector_metrics['asr']:.1f}%)"
        task_str = f"{vector_metrics['task_success_count']}/{vector_metrics['total_count']} ({vector_metrics['task_rate']:.1f}%)"
        log_info(f"{vector.title():<40} {asr_str:<25} {task_str:<25}")
    log_info("")
    
    # Display attack-level summary
    log_separator("METRICS PER ATTACK")
    log_info(f"{'Attack':<40} {'ASR':<25} {'Task Success':<25}")
    log_info("-" * 80)
    # Sort attacks by ASR (descending), then by Task Success (descending)
    sorted_attacks = sorted(
        metrics['by_attack'].items(),
        key=lambda x: (-x[1]['overall']['asr'], -x[1]['overall']['task_rate'])
    )
    for attack_name, attack_data in sorted_attacks:
        attack_metrics = attack_data['overall']
        asr_str = f"{attack_metrics['attack_success_count']}/{attack_metrics['total_count']} ({attack_metrics['asr']:.1f}%)"
        task_str = f"{attack_metrics['task_success_count']}/{attack_metrics['total_count']} ({attack_metrics['task_rate']:.1f}%)"
        log_info(f"{attack_name:<40} {asr_str:<25} {task_str:<25}")
    log_info("")
    
    # Display per-attack per-tool breakdown
    log_separator("METRICS PER ATTACK PER TOOL")
    for attack_name, attack_data in sorted_attacks:
        log_info(f"\n{attack_name}:")
        log_info(f"  {'Tool/Vector':<38} {'ASR':<25} {'Task Success':<25}")
        log_info("  " + "-" * 78)
        
        # Sort vectors alphabetically for consistent display
        sorted_vectors = sorted(attack_data['vectors'].items())
        for vector, vector_metrics in sorted_vectors:
            asr_str = f"{vector_metrics['attack_success_count']}/{vector_metrics['total_count']} ({vector_metrics['asr']:.1f}%)"
            task_str = f"{vector_metrics['task_success_count']}/{vector_metrics['total_count']} ({vector_metrics['task_rate']:.1f}%)"
            log_info(f"  {vector.title():<38} {asr_str:<25} {task_str:<25}")
    log_info("")
    
    # Display successful attack file paths grouped by vector then by attack
    log_separator("SUCCESSFUL ATTACK FILES")
    
    # Organize successful attacks by vector and attack
    successful_by_vector = {}
    total_successful = 0
    
    for attack_name in metrics['by_attack'].keys():
        for vector in metrics['by_attack'][attack_name]['vectors'].keys():
            vector_results = metrics['by_attack'][attack_name]['vectors'][vector]['results']
            for result in vector_results:
                if result.get('attack_success', False):
                    filepath = result.get('filepath', 'Unknown')
                    
                    # Initialize vector dict if needed
                    if vector not in successful_by_vector:
                        successful_by_vector[vector] = {}
                    
                    # Initialize attack list if needed
                    if attack_name not in successful_by_vector[vector]:
                        successful_by_vector[vector][attack_name] = []
                    
                    successful_by_vector[vector][attack_name].append(filepath)
                    total_successful += 1
    
    if successful_by_vector:
        log_info(f"Total successful attacks: {total_successful}")
        
        # Display grouped by vector, then by attack
        for vector in sorted(successful_by_vector.keys()):
            log_info("")
            log_info(f"{vector.upper()} Vector:")
            for attack_name in sorted(successful_by_vector[vector].keys()):
                log_info(f"  {attack_name}:")
                for filepath in successful_by_vector[vector][attack_name]:
                    log_info(f"    {filepath}")
    else:
        log_info("No successful attacks found.")


def evaluate_attack_summary(lab):
    """Evaluate and display summary for attack mode"""
    # In inference mode, use in-memory results; otherwise load from files
    if lab.inference_mode and hasattr(lab, '_evaluation_results'):
        results = lab._evaluation_results
    else:
        results = load_attack_results_from_files(lab)
    
    if not results:
        return []
    
    metrics = compute_attack_metrics(results)
    
    log_separator(f"EVALUATION {lab.mas.environment_name.upper()}")
    print_metrics(metrics)
    
    # Save metrics to JSON file for programmatic access
    # Each iteration should have its own metrics.json file
    if lab.iteration is not None:
        # Get evolution path prefix based on iteration
        evolution_path = get_evolution_path_prefix(lab.effective_iteration, lab.evolution_name)
        
        if lab.defense_id is not None:
            # Defense variant: save to defenses/defense_X/metrics.json
            metrics_dir = f"{lab.mas.folder}/{lab.mas.model_id}/{lab.mas.environment_name}/{evolution_path}iterations/iter_{lab.iteration}/defenses/defense_{lab.defense_id}"
        else:
            # No defense (red-team-only or baseline): save to iterations/iter_X/metrics.json
            metrics_dir = f"{lab.mas.folder}/{lab.mas.model_id}/{lab.mas.environment_name}/{evolution_path}iterations/iter_{lab.iteration}"
    else:
        # Legacy: no iteration specified (should not happen in new code)
        raise ValueError("iteration parameter is required - all metrics must be stored in iterations/iter_X/ directories")
    
    metrics_file = os.path.join(metrics_dir, "metrics.json")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save overall metrics in a clean format
    metrics_output = {
        'environment': lab.mas.environment_name,
        'iteration': lab.iteration,
        'defense_id': lab.defense_id,
        'timestamp': datetime.now().isoformat(),
        'overall': {
            'asr': metrics['overall']['asr'] / 100.0,  # Convert to 0-1 range
            'utility': metrics['overall']['task_rate'] / 100.0,  # Convert to 0-1 range
            'attack_success_count': metrics['overall']['attack_success_count'],
            'task_success_count': metrics['overall']['task_success_count'],
            'total_count': metrics['overall']['total_count']
        }
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    log_info(f"Metrics saved to {metrics_file}")

    return results

def global_evaluation(results: list[list[dict]]):
    """Evaluation of all environments"""
    flat_results = []
    for r in results:
        flat_results.extend(r)
    metrics = compute_attack_metrics(flat_results)
    
    log_separator("GLOBAL EVALUATION (ALL ENVIRONMENTS)")
    print_metrics(metrics)

    return flat_results
