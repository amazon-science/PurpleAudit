# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Attack Manager for Red Team evolution.

Manages the evolution of attack prompts through iterative selection and generation.
"""

import os
import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from .agents import RedTeamAgent
from .utils import (
    load_attack_results,
    compute_asr_by_attack,
    get_attack_examples,
    save_evolution_file,
    load_evolution_file
)
from purpleaudit.utils.logger import log_info, log_success, log_separator, log_error



class AttackManager:
    """
    Manages the evolution of attack prompts.
    
    This class implements the red team phase of the genetic framework:
    1. Load and analyze attack results from all environments
    2. Select top-k successful attacks based on ASR
    3. Generate N new attack variants using LLM
    4. Prune failed attacks from the pool
    """
    
    def __init__(self,
                 folder: str = "environments",
                 region_name: str = "us-west-2",
                 model_id: str = "default",
                 fixed_model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                 evolution_name: str = "default",
                 evolution_guidance: str = "Based on the provided information, you are free to select the most relevant attack generation approach to maximize attack success rate."):
        """
        Initialize Attack Manager.
        
        Args:
            folder: Base folder containing environments
            region_name: AWS region for Bedrock
            model_id: Model ID subdirectory within folder (for generated conversations)
            fixed_model_id: Model ID for Red Team agent
            evolution_name: Name of the evolution experiment
            evolution_guidance: Instructions to guide attack evolution strategy
        """
        self.folder = folder
        self.model_id = model_id
        self.evolution_name = evolution_name
        self.evolution_guidance = evolution_guidance
        self.red_team = RedTeamAgent(region_name=region_name, model_id=fixed_model_id)
        
        self.attack_pool = {}  # Maps attack_name to attack dict
        self.attack_performance = {}  # Maps attack_name to (ASR, num_success, total)
    
    def load_and_analyze_results(self, environments: List[str] = None, iteration: int = None) -> Dict[str, List[Dict]]:
        """
        Load attack results from all environments and analyze performance.
        
        Args:
            environments: List of environment names to load, or None for all
            iteration: Iteration number to load results from (None for baseline)
            
        Returns:
            Dictionary mapping environment names to attack results
        """
        log_info(f"Loading attack results from iteration {iteration if iteration is not None else 0}...")
        
        # Load results from disk
        results_by_env = load_attack_results(self.folder, environments, self.model_id, iteration=iteration, evolution_name=self.evolution_name)
        
        if not results_by_env:
            log_error("No attack results found!")
            return {}
        
        # Aggregate all results
        all_results = []
        for env_name, results in results_by_env.items():
            all_results.extend(results)
        
        log_info(f"Loaded {len(all_results)} attack results from {len(results_by_env)} environments")
        
        # Compute ASR by attack
        self.attack_performance = compute_asr_by_attack(all_results)
        
        # Display summary
        log_separator("ATTACK PERFORMANCE SUMMARY")
        for attack_name, (asr, utility, num_success, total) in sorted(
            self.attack_performance.items(), 
            key=lambda x: x[1][0],  # Still sort by ASR
            reverse=True
        ):
            status = "✓" if asr > 0.5 else "✗"
            log_info(f"{status} {attack_name}: ASR={asr:.2%}, Utility={utility:.2%} ({num_success}/{total})")
        log_separator()
        
        return results_by_env
    
    def select_attacks(self,
                      results_by_env: Dict[str, List[Dict]],
                      top_k: int = 5,
                      top_k_per_attack: int = 3,
                      num_failed: int = 2,
                      ignore_evolved: bool = False) -> Tuple[List[Dict], List[Dict]]:
        """
        Select top-k successful attacks and some failed attacks for learning.
        
        Args:
            results_by_env: Dictionary of attack results by environment
            top_k: Number of top successful attacks to select
            num_failed: Number of failed attacks to include for contrast
            ignore_evolved: If True, filter out attacks starting with "evolved"
            
        Returns:
            Tuple of (successful_attacks, failed_attacks)
        """
        # Filter attack performance if requested
        attack_performance = self.attack_performance
        if ignore_evolved:
            attack_performance = {
                name: metrics for name, metrics in self.attack_performance.items()
                if not name.startswith('evolved')
            }
            log_info(f"Filtering attack examples (starting with 'evolved')")
            log_info(f"  Before filtering: {len(self.attack_performance)} attacks")
            log_info(f"  After filtering: {len(attack_performance)} attacks")
            filtered_out = set(self.attack_performance.keys()) - set(attack_performance.keys())
            if filtered_out:
                log_info(f"  Filtered from examples: {', '.join(sorted(filtered_out))}")
        
        # Sort attacks by ASR
        sorted_attacks = sorted(
            attack_performance.items(),
            key=lambda x: x[1][0],  # Sort by ASR
            reverse=True
        )
        
        # Select top-k successful attacks
        top_attack_names = [name for name, (asr, _, _, _) in sorted_attacks[:top_k] if asr > 0]
        
        # Select some failed attacks (low ASR)
        failed_attack_names = [name for name, (asr, _, _, _) in sorted_attacks[-num_failed:] if asr == 0.0]
        
        # Collect examples
        all_results = []
        for results in results_by_env.values():
            all_results.extend(results)
        
        successful_attacks = []
        for attack_name in top_attack_names:
            examples = get_attack_examples(all_results, attack_name, success_only=True)
            if examples:
                # Take a few examples of each attack type
                successful_attacks.extend(examples[:top_k_per_attack])
        
        failed_attacks = []
        for attack_name in failed_attack_names:
            examples = get_attack_examples(all_results, attack_name, success_only=False)
            if examples:
                failed_attacks.extend(examples[:2])
        
        log_info(f"Selected {len(successful_attacks)} successful attack examples")
        log_info(f"Selected {len(failed_attacks)} failed attack examples")
        
        return successful_attacks, failed_attacks
    
    def generate_new_attacks(self,
                            num_new_attacks: int,
                            successful_attacks: List[Dict],
                            failed_attacks: List[Dict],
                            safeguard_context: str = None) -> List[Dict]:
        """
        Generate new attack variants using the Red Team agent.
        
        Args:
            num_new_attacks: Number of new attacks to generate
            successful_attacks: Examples of successful attacks
            failed_attacks: Examples of failed attacks
            safeguard_context: Optional context about current defenses
            
        Returns:
            List of newly generated attack dictionaries
        """
        log_info(f"Generating {num_new_attacks} new attack variants...")
        log_separator()
        
        new_attacks = self.red_team.generate_attacks(
            num_attacks=num_new_attacks,
            successful_attacks=successful_attacks,
            failed_attacks=failed_attacks,
            safeguard_context=safeguard_context,
            attack_performance=self.attack_performance,
            evolution_guidance=self.evolution_guidance
        )
        
        if new_attacks:
            log_success(f"Generated {len(new_attacks)} new attacks")
            for i, attack in enumerate(new_attacks, 1):
                log_info(f"  {i}. {attack['name']}: {attack.get('description', 'N/A')}")
        else:
            raise RuntimeError("Failed to generate new attacks")
        
        return new_attacks
    
    def update_attack_pool(self,
                          new_attacks: List[Dict],
                          top_k_names: List[str],
                          keep_new: bool = True,
                          prune_failed: bool = True) -> List[Dict]:
        """
        Update the attack pool with new attacks and prune failed ones.
        
        Args:
            new_attacks: Newly generated attacks
            top_k_names: Names of top-k attacks to keep
            keep_new: Whether to keep all new attacks
            prune_failed: Whether to prune attacks with low ASR
            
        Returns:
            Updated attack pool as a list
        """
        # Keep top-k successful attacks
        updated_pool = []
        for attack_name in top_k_names:
            if attack_name in self.attack_pool:
                updated_pool.append(self.attack_pool[attack_name])
            else:
                # Create entry from attack name (for hard-coded attacks)
                asr, utility, num_success, total = self.attack_performance.get(attack_name, (0, 0, 0, 0))
                updated_pool.append({
                    'name': attack_name,
                    'type': 'hardcoded',
                    'asr': asr,
                    'utility': utility,
                    'num_success': num_success,
                    'total': total
                })
        
        # Add new attacks
        if keep_new:
            for attack in new_attacks:
                attack['type'] = 'generated'
                attack['asr'] = 0.0  # Will be evaluated in next iteration
                updated_pool.append(attack)
        
        log_info(f"Attack pool updated: {len(updated_pool)} attacks total")
        log_info(f"  - Top successful attacks: {len(top_k_names)}")
        log_info(f"  - New generated attacks: {len(new_attacks) if keep_new else 0}")
        
        return updated_pool
    
    def evolve_attacks(self,
                      iteration: int,
                      num_new_attacks: int = 5,
                      top_k: int = 5,
                      environments: List[str] = None,
                      safeguard_context: str = None,
                      ignore_evolved_from_context: bool = False) -> List[Dict]:
        """
        Complete red team evolution cycle for one iteration.
        
        This is the main entry point for attack evolution:
        1. Load and analyze attack results
        2. Select top-k successful attacks
        3. Generate N new attack variants
        4. Update and save attack pool
        
        Args:
            iteration: Current iteration number
            num_new_attacks: Number of new attacks to generate
            top_k: Number of top attacks to keep
            environments: List of environment names to process
            safeguard_context: Optional context about defenses
            
        Returns:
            Updated attack pool
        """
        log_separator(f"RED TEAM EVOLUTION - ITERATION {iteration}")
        
        # Step 1: Load and analyze results from the previous iteration
        # For iteration N, we load results from iteration N-1
        previous_iteration = iteration - 1 if iteration > 0 else 0
        results_by_env = self.load_and_analyze_results(environments, iteration=previous_iteration)
        
        if not results_by_env:
            log_error("No results to analyze. Skipping red team evolution.")
            return []
        
        # Step 2: Select attacks for learning
        successful_attacks, failed_attacks = self.select_attacks(
            results_by_env, 
            top_k=top_k,
            ignore_evolved=ignore_evolved_from_context
        )

        if len(successful_attacks) == 0:
            log_success("No successful attack found (ASR=0%). No need to evolve further.")
            exit(0)
        
        # Step 3: Generate new attacks
        new_attacks = self.generate_new_attacks(
            num_new_attacks=num_new_attacks,
            successful_attacks=successful_attacks,
            failed_attacks=failed_attacks,
            safeguard_context=safeguard_context
        )
        
        # Step 4: Update attack pool
        # Filter out evolved attacks if requested (attacks starting with "evolved")
        if ignore_evolved_from_context:
            filtered_performance = {}
            for name, metrics in self.attack_performance.items():
                # Keep only attacks that don't start with "evolved"
                if not name.startswith('evolved'):
                    filtered_performance[name] = metrics
            
            log_info(f"Filtering out evolved attacks (starting with 'evolved')")
            log_info(f"  Before filtering: {len(self.attack_performance)} attacks")
            log_info(f"  After filtering: {len(filtered_performance)} attacks")
            
            # Log which attacks were filtered out
            filtered_out = set(self.attack_performance.keys()) - set(filtered_performance.keys())
            if filtered_out:
                log_info(f"  Filtered out: {', '.join(sorted(filtered_out))}")
        else:
            filtered_performance = self.attack_performance
        
        top_attack_names = [name for name, _ in sorted(
            filtered_performance.items(),
            key=lambda x: x[1][0],
            reverse=True
        )[:top_k]]
        
        updated_pool = self.update_attack_pool(
            new_attacks=new_attacks,
            top_k_names=top_attack_names,
            keep_new=True,
            prune_failed=True
        )
        
        # Step 5: Save iteration state to shared evolution folder
        # Save ONLY the new generated attacks (not the top-k from previous iteration)
        # This allows the same attacks to be used across different models
        attacks_file = save_evolution_file(self.folder, iteration, "attacks.json", new_attacks, 
                                          evolution_name=self.evolution_name)
        
        log_success(f"Saved {len(new_attacks)} new generated attacks to {attacks_file}")
        log_separator()
        
        return updated_pool
    
    def get_attack_pool(self) -> List[Dict]:
        """Get the current attack pool"""
        return list(self.attack_pool.values())
    
    def load_attack_pool(self, iteration: int) -> List[Dict]:
        """
        Load attack pool from a previous iteration from the shared evolution folder.
        
        This allows loading attacks evolved on one model and testing them on another.
        
        Args:
            iteration: Iteration number to load
            
        Returns:
            List of attack dictionaries
        """
        # Load from shared evolution folder with backward compatibility fallback
        attacks = load_evolution_file(
            self.folder, 
            iteration, 
            "attacks.json",
            model_id=self.model_id,
            fallback_to_legacy=True,
            evolution_name=self.evolution_name
        )
        
        if attacks is None:
            log_error(f"Attack pool not found for iteration {iteration}")
            log_info(f"Checked: {self.folder}/_evolution/iterations/iter_{iteration}/attacks.json")
            return []
        
        # Update internal pool
        for attack in attacks:
            name = attack.get('name', 'unknown')
            self.attack_pool[name] = attack
        
        log_info(f"Loaded {len(attacks)} attacks from iteration {iteration}")
        return attacks
