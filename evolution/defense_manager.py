# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Defense Manager for Blue Team evolution.

Manages the evolution of defense safeguards through iterative selection and generation.
"""

import os
import json
from typing import List, Dict, Tuple, Optional

from .agents import BlueTeamAgent
from .utils import (
    extract_system_prompts,
    save_evolution_file,
    load_evolution_file,
    get_evolution_path_prefix,
    compute_metrics_from_attack_files
)
from purpleaudit.utils.logger import log_info, log_success, log_separator, log_error



class DefenseManager:
    """
    Manages the evolution of defense safeguards.
    
    This class implements the blue team phase of the genetic framework:
    1. Generate M new safeguard patterns using attack examples
    2. Create M environment variants with safeguarded system prompts
    3. Save variants as YAML configs (not full environment copies)
    4. Track defense effectiveness for future iterations
    """
    
    def __init__(self,
                 folder: str = "environments",
                 region_name: str = "us-west-2",
                 model_id: str = "default",
                 fixed_model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                 evolution_name: str = "default"):
        """
        Initialize Defense Manager.
        
        Args:
            folder: Base folder containing environments
            region_name: AWS region for Bedrock
            model_id: Model ID subdirectory within folder (for generated conversations)
            fixed_model_id: Model ID for Blue Team agent
            evolution_name: Name of the evolution experiment (default: "default")
        """
        self.folder = folder
        self.model_id = model_id
        self.evolution_name = evolution_name
        self.blue_team = BlueTeamAgent(
            region_name=region_name, 
            model_id=fixed_model_id,
            temperature=0.0  # Deterministic for reproducibility
        )
        
        self.defense_pool = []  # List of defense safeguard dicts
        self.defense_performance = {}  # Maps defense_name to effectiveness metrics
    
    def generate_defenses(self,
                         num_defenses: int,
                         attack_examples: List[Dict],
                         environments: List[str],
                         effective_defenses: List[Dict] = None,
                         baseline_metrics: Dict[str, float] = None,
                         attack_performance: Dict = None,
                         evolution_guidance: str = None) -> List[Dict]:
        """
        Generate M new defense safeguard patterns.
        
        Args:
            num_defenses: Number of defense patterns to generate
            attack_examples: Attack examples to defend against
            environments: List of environment names to get system prompts from
            effective_defenses: Previously effective defenses for learning
            baseline_metrics: Dict with 'avg_asr' and 'avg_utility' for baseline performance
            attack_performance: Pre-computed attack performance metrics
            evolution_guidance: Custom instructions to guide defense evolution strategy
            
        Returns:
            List of defense safeguard dictionaries
        """
        log_info(f"Generating {num_defenses} defense safeguard patterns...")
        
        # Get system prompts from one representative environment
        system_prompts = None
        if environments:
            env_name = environments[0]
            system_prompts = extract_system_prompts(self.folder, env_name, self.model_id)
            log_info(f"Loaded system prompts from {env_name} ({len(system_prompts)} agents)")
        
        # Log baseline metrics if provided
        if baseline_metrics:
            log_info(f"Baseline (no defense): ASR={baseline_metrics.get('avg_asr', 0):.1%}, Utility={baseline_metrics.get('avg_utility', 0):.1%}")
        
        # Generate defenses
        new_defenses = self.blue_team.generate_defenses(
            num_defenses=num_defenses,
            attack_examples=attack_examples,
            system_prompts=system_prompts,
            effective_defenses=effective_defenses,
            baseline_metrics=baseline_metrics,
            attack_performance=attack_performance,
            evolution_guidance=evolution_guidance
        )
        
        if new_defenses:
            log_success(f"Generated {len(new_defenses)} defense safeguards")
            for i, defense in enumerate(new_defenses, 1):
                log_info(f"  {i}. {defense['name']}: {defense.get('description', 'N/A')}")
        else:
            log_error("Failed to generate defenses")
        
        return new_defenses
    
    # Defense variant creation is no longer needed with simplified defenses
    # Defenses are now stored directly in defenses.json with defense_prompt field
    
    def load_defenses(self, iteration: int) -> List[Dict]:
        """
        Load defenses for an iteration from defenses.json.
        
        Args:
            iteration: Iteration number
            
        Returns:
            List of defense dictionaries
        """
        # Load defenses from the shared evolution folder
        defenses = load_evolution_file(
            self.folder,
            iteration,
            "defenses.json",
            model_id=self.model_id,
            fallback_to_legacy=True,
            evolution_name=self.evolution_name
        )
        
        if defenses:
            log_info(f"Loaded {len(defenses)} defenses from iteration {iteration}")
        else:
            log_info(f"No defenses found for iteration {iteration}")
            defenses = []
        
        return defenses
    
    def evaluate_defenses(self,
                         iteration: int,
                         environments: List[str],
                         results_by_defense: Dict[str, Dict]) -> Dict[str, float]:
        """
        Evaluate defense effectiveness based on attack results.
        
        Args:
            iteration: Current iteration number
            environments: List of environment names
            results_by_defense: Attack results organized by defense
            
        Returns:
            Dictionary mapping defense names to effectiveness scores
        """
        log_separator("EVALUATING DEFENSE EFFECTIVENESS")
        
        # Load defenses for this iteration
        defenses = self.load_defenses(iteration)
        defense_scores = {}
        
        for env_name in environments:
            for defense_id, defense in enumerate(defenses, start=1):
                defense_name = defense.get('name', f'Defense_{defense_id}')
                
                # Get results for this defense
                defense_key = f"{env_name}_defense_{defense_id}"
                if defense_key in results_by_defense:
                    results = results_by_defense[defense_key]
                    
                    # Calculate ASR for this defense
                    total = len(results)
                    successful = sum(1 for r in results if r.get('attack_success', False))
                    asr = successful / total if total > 0 else 0.0
                    
                    # Effectiveness is inverse of ASR (lower ASR = better defense)
                    effectiveness = 1.0 - asr
                    
                    if defense_name not in defense_scores:
                        defense_scores[defense_name] = []
                    defense_scores[defense_name].append(effectiveness)
                    
                    log_info(f"{defense_name} on {env_name}: ASR={asr:.2%}, Effectiveness={effectiveness:.2%}")
        
        # Average effectiveness across environments
        avg_defense_scores = {}
        for defense_name, scores in defense_scores.items():
            avg_defense_scores[defense_name] = sum(scores) / len(scores) if scores else 0.0
        
        # Sort by effectiveness
        sorted_defenses = sorted(avg_defense_scores.items(), key=lambda x: x[1], reverse=True)
        
        log_separator("TOP DEFENSES BY EFFECTIVENESS")
        for defense_name, score in sorted_defenses[:5]:
            log_info(f"✓ {defense_name}: {score:.2%}")
        log_separator()
        
        self.defense_performance = avg_defense_scores
        return avg_defense_scores
    
    def select_effective_defenses(self, top_k: int = 3, current_iteration: int = None) -> List[Dict]:
        """
        Select top-k most effective defenses for learning.
        
        Args:
            top_k: Number of top defenses to select
            current_iteration: Current iteration number (for loading previous defenses if needed)
            
        Returns:
            List of effective defense dictionaries with avg_asr and avg_utility metrics added
        """
        # If defense performance hasn't been evaluated yet (e.g., in blue_team_only mode),
        # load metrics from disk and select best defenses across ALL previous iterations
        if not self.defense_performance:
            log_info("No defense performance data in memory, loading from disk across all iterations")
            
            if current_iteration and current_iteration > 1:
                # Load defenses and their metrics from ALL previous iterations
                all_defenses_with_metrics = self._load_all_defense_metrics(current_iteration)
                
                if all_defenses_with_metrics:
                    # Sort by ASR (lower is better)
                    sorted_defenses = sorted(
                        all_defenses_with_metrics,
                        key=lambda x: x.get('avg_asr', 1.0)
                    )
                    
                    # Select top_k best defenses
                    best_defenses = sorted_defenses[:top_k]
                    log_info(f"Selected {len(best_defenses)} best defenses across all previous iterations")
                    for i, defense in enumerate(best_defenses, 1):
                        log_info(f"  {i}. {defense['name']}: ASR={defense.get('avg_asr', 0):.1%}, Utility={defense.get('avg_utility', 0):.1%}")
                    return best_defenses
            
            # Fallback: if no metrics available, use most recent defenses
            log_info("No metrics found on disk, using most recent defenses from pool")
            if not self.defense_pool and current_iteration and current_iteration > 1:
                # Load defenses from previous iteration
                prev_iteration = current_iteration - 1
                log_info(f"Defense pool empty, loading from iteration {prev_iteration}")
                self.load_defense_pool(prev_iteration)
            
            recent_defenses = self.defense_pool[-top_k:] if self.defense_pool else []
            log_info(f"Selected {len(recent_defenses)} recent defenses (without metrics)")
            return recent_defenses
        
        # Sort by effectiveness (effectiveness = 1.0 - ASR, so higher is better)
        sorted_defenses = sorted(
            self.defense_performance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top-k defense names
        top_defense_names = [name for name, effectiveness in sorted_defenses[:top_k]]
        
        # Find full defense objects from pool and add metrics
        effective_defenses = []
        for defense in self.defense_pool:
            defense_name = defense.get('name')
            if defense_name in top_defense_names:
                # Create a copy to avoid modifying the original
                defense_with_metrics = defense.copy()
                
                # Add aggregate metrics (mean across all environments & attacks)
                effectiveness = self.defense_performance.get(defense_name, 0.5)
                # effectiveness = 1.0 - ASR, so ASR = 1.0 - effectiveness
                avg_asr = 1.0 - effectiveness
                
                # For utility, we need to compute it from evaluation results if available
                # For now, we'll estimate: if defense is effective (low ASR), assume reasonable utility
                # This is a simplification - ideally we'd track utility separately
                # Higher effectiveness usually means some utility cost, so estimate inverse relationship
                avg_utility = 0.5 + (effectiveness * 0.3)  # Range: 0.5-0.8
                
                defense_with_metrics['avg_asr'] = avg_asr
                defense_with_metrics['avg_utility'] = avg_utility
                
                effective_defenses.append(defense_with_metrics)
        
        log_info(f"Selected {len(effective_defenses)} effective defenses with metrics for learning")
        return effective_defenses
    
    def compute_baseline_metrics(self, environments: List[str], iteration: int = 0) -> Dict[str, float]:
        """
        Compute baseline performance metrics (ASR and Utility without any defense).
        
        This computes metrics directly from attack result files, falling back to metrics.json if available.
        
        Args:
            environments: List of environment names
            iteration: Iteration parameter (default 0 for baseline)
            
        Returns:
            Dict with 'avg_asr' and 'avg_utility' averaged across environments
        """
        asr_values = []
        utility_values = []
        
        for env in environments:
            env_asr = None
            env_utility = None
            
            # First, try to load from metrics.json
            metrics_file = os.path.join(self.folder, self.model_id, env, "iterations", f"iter_{iteration}", "metrics.json")
            
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        metrics_data = json.load(f)
                        env_asr = metrics_data.get('overall', {}).get('asr')
                        env_utility = metrics_data.get('overall', {}).get('utility')
                        if env_asr is not None and env_utility is not None:
                            asr_values.append(env_asr)
                            utility_values.append(env_utility)
                            log_info(f"Loaded baseline metrics for {env} from metrics.json")
                            continue
                except Exception as e:
                    log_error(f"Error loading baseline metrics from {metrics_file}: {e}")
            
            # Fallback: compute from attack result files directly
            env_metrics = self._compute_metrics_from_attack_files(env, iteration)
            if env_metrics:
                asr_values.append(env_metrics['asr'])
                utility_values.append(env_metrics['utility'])
                log_info(f"Computed baseline metrics for {env} from attack files: ASR={env_metrics['asr']:.1%}, Utility={env_metrics['utility']:.1%}")
        
        if asr_values and utility_values:
            baseline_metrics = {
                'avg_asr': sum(asr_values) / len(asr_values),
                'avg_utility': sum(utility_values) / len(utility_values)
            }
            log_info(f"Computed baseline metrics from {len(asr_values)} environments")
            return baseline_metrics
        
        log_error("No baseline metrics found")
        return None
    
    def _compute_metrics_from_attack_files(self, env: str, iteration: int = 0, defense_id: Optional[int] = None) -> Optional[Dict[str, float]]:
        """
        Compute ASR and Utility directly from attack result JSON files.
        
        This is a wrapper around the shared utility function.
        
        Args:
            env: Environment name
            iteration: Iteration number
            defense_id: Optional defense ID for defense-specific metrics
            
        Returns:
            Dict with 'asr' and 'utility', or None if no results found
        """
        return compute_metrics_from_attack_files(
            folder=self.folder,
            model_id=self.model_id,
            env=env,
            iteration=iteration,
            defense_id=defense_id,
            evolution_name=self.evolution_name
        )
    
    def evolve_defenses(self,
                       iteration: int,
                       num_new_defenses: int,
                       attack_examples: List[Dict],
                       environments: List[str],
                       effective_defenses: List[Dict] = None,
                       attack_performance: Dict = None,
                       evolution_guidance: str = None) -> List[Dict]:
        """
        Complete blue team evolution cycle for one iteration.
        
        This is the main entry point for defense evolution:
        1. Compute baseline metrics (from defense_0)
        2. Generate M new defense safeguards
        3. Create M variants for each environment
        4. Save variants as YAML configs
        
        Args:
            iteration: Current iteration number
            num_new_defenses: Number of new defenses to generate
            attack_examples: Attack examples to defend against
            environments: List of environment names
            effective_defenses: Previously effective defenses
            attack_performance: Pre-computed attack performance metrics
            evolution_guidance: Custom instructions to guide defense evolution strategy
            
        Returns:
            List of newly generated defenses
        """
        log_separator(f"BLUE TEAM EVOLUTION - ITERATION {iteration}")
        
        # Step 0: Compute baseline metrics if available
        baseline_metrics = self.compute_baseline_metrics(environments)
        
        # Step 1: Generate new defenses
        new_defenses = self.generate_defenses(
            num_defenses=num_new_defenses,
            attack_examples=attack_examples,
            environments=environments,
            effective_defenses=effective_defenses,
            baseline_metrics=baseline_metrics,
            attack_performance=attack_performance,
            evolution_guidance=evolution_guidance
        )
        
        if not new_defenses:
            log_error("No defenses generated. Skipping blue team evolution.")
            return []
        
        # Update defense pool
        self.defense_pool.extend(new_defenses)
        
        # Step 2: Save defense metadata to shared evolution folder
        # This allows the same defenses to be used across different models
        defenses_file = save_evolution_file(self.folder, iteration, "defenses.json", new_defenses, self.evolution_name)
        
        log_success(f"Defenses saved to {defenses_file}")
        
        log_separator()
        
        return new_defenses
    
    def get_defense_pool(self) -> List[Dict]:
        """Get the current defense pool"""
        return self.defense_pool
    
    def load_defense_pool(self, iteration: int) -> List[Dict]:
        """
        Load defense pool from a previous iteration from the shared evolution folder.
        
        This allows loading defenses evolved on one model and testing them on another.
        
        Args:
            iteration: Iteration number to load
            
        Returns:
            List of defense dictionaries
        """
        # Load from shared evolution folder with backward compatibility fallback
        defenses = load_evolution_file(
            self.folder,
            iteration,
            "defenses.json",
            model_id=self.model_id,
            fallback_to_legacy=True,
            evolution_name=self.evolution_name
        )
        
        if defenses is None:
            log_error(f"Defense pool not found for iteration {iteration}")
            log_info(f"Checked: {self.folder}/_evolution/iterations/iter_{iteration}/defenses.json")
            return []
        
        # Update internal pool
        self.defense_pool.extend(defenses)
        
        log_info(f"Loaded {len(defenses)} defenses from iteration {iteration}")
        return defenses
    
    def _load_all_defense_metrics(self, current_iteration: int) -> List[Dict]:
        """
        Load defense metrics from ALL previous iterations by scanning metrics.json files.
        
        This method scans all environments and iterations to collect actual performance data
        from evaluation runs, then computes aggregate metrics across all environments.
        
        Args:
            current_iteration: Current iteration number (we load from 1 to current-1)
            
        Returns:
            List of defense dictionaries with avg_asr and avg_utility metrics
        """
        all_defenses = {}  # {defense_name: {'def': defense_dict, 'asr_values': [], 'utility_values': []}}
        
        # Get list of environments
        model_folder = os.path.join(self.folder, self.model_id)
        if not os.path.exists(model_folder):
            return []
        environments = [d for d in os.listdir(model_folder) 
                       if os.path.isdir(os.path.join(model_folder, d)) and not d.startswith('_')]
        
        # Scan all previous iterations
        for iteration in range(1, current_iteration):
            # Load defenses for this iteration from shared evolution folder
            iter_defenses = load_evolution_file(
                self.folder,
                iteration,
                "defenses.json",
                model_id=self.model_id,
                fallback_to_legacy=True
            )
            
            if iter_defenses is None:
                continue
            
            # For each defense in this iteration, collect metrics from all environments
            for defense_id, defense in enumerate(iter_defenses, start=1):
                defense_name = defense.get('name')
                
                if defense_name not in all_defenses:
                    all_defenses[defense_name] = {
                        'def': defense,
                        'asr_values': [],
                        'utility_values': []
                    }
                
                # Load metrics from each environment
                for env in environments:
                    # In blue_team_only mode, metrics are in iter_0
                    metrics_file = os.path.join(
                        self.folder, self.model_id, env, "iterations", "iter_0", "defenses",
                        f"defense_{defense_id}", "metrics.json"
                    )
                    
                    # Also check the normal location
                    if not os.path.exists(metrics_file):
                        metrics_file = os.path.join(
                            self.folder, self.model_id, env, "iterations", f"iter_{iteration}", "defenses",
                            f"defense_{defense_id}", "metrics.json"
                        )
                    
                    if os.path.exists(metrics_file):
                        try:
                            with open(metrics_file, 'r') as f:
                                metrics_data = json.load(f)
                                asr = metrics_data.get('overall', {}).get('asr', 0)
                                utility = metrics_data.get('overall', {}).get('utility', 0)
                                
                                all_defenses[defense_name]['asr_values'].append(asr)
                                all_defenses[defense_name]['utility_values'].append(utility)
                        except Exception as e:
                            log_error(f"Error loading metrics from {metrics_file}: {e}")
                            continue
        
        # Compute aggregate metrics (mean across all environments and iterations)
        defenses_with_metrics = []
        for defense_name, data in all_defenses.items():
            if data['asr_values'] and data['utility_values']:
                defense_copy = data['def'].copy()
                defense_copy['avg_asr'] = sum(data['asr_values']) / len(data['asr_values'])
                defense_copy['avg_utility'] = sum(data['utility_values']) / len(data['utility_values'])
                defenses_with_metrics.append(defense_copy)
        
        log_info(f"Loaded metrics for {len(defenses_with_metrics)} defenses across {current_iteration-1} iterations")
        return defenses_with_metrics
