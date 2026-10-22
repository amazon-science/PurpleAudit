# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Clean implementation of the red team-blue team co-evolution pipeline.
"""
import os
import sys
import shlex
import subprocess
import argparse
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from purpleaudit.evolution import EvolutionOrchestrator
from purpleaudit.utils.logger import log_info, log_success, log_error, log_separator
from purpleaudit.utils.pipeline_utils import (
    get_iteration_dir,
    load_attacks, load_defenses,
    load_attack_results, compute_metrics, save_metrics,
    save_json_file
)


class EvolutionPipeline:
    """Main evolution pipeline for red team-blue team co-evolution."""
    
    def __init__(self, 
                 folder: str = "environments",
                 model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                 fixed_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                 evolution_name: str = "default",
                 max_workers: int = 5,
                 attacks_filter: str = "",
                 env_filter: str = "",
                 region: str = "us-west-2",
                 evolution_guidance: str = "",
                 ignore_evolved_from_context: bool = False,
                 blue_team_only: bool = False,
                 red_team_only: bool = False):
        self.folder = folder
        self.model = model
        self.fixed_model = fixed_model
        self.evolution_name = evolution_name
        self.max_workers = max_workers
        self.attacks_filter = attacks_filter  # Comma-separated list of baseline attacks
        self.env_filter = env_filter  # Specific environment or comma-separated list
        self.region = region
        self.evolution_guidance = evolution_guidance
        self.ignore_evolved_from_context = ignore_evolved_from_context
        self.blue_team_only = blue_team_only
        self.red_team_only = red_team_only
        self.environments = self._get_environments()
        
        # Initialize a single orchestrator instance
        self.orchestrator = EvolutionOrchestrator(
            folder=self.folder,
            region_name=self.region,
            model_id=self.model,
            fixed_model_id=self.fixed_model,
            evolution_name=self.evolution_name
        )
        
    def _get_environments(self) -> List[str]:
        """Get list of available environments."""
        model_folder = os.path.join(self.folder, self.model)
        if not os.path.exists(model_folder):
            return []
        
        # Get all available environments
        all_envs = [d for d in os.listdir(model_folder) 
                    if os.path.isdir(os.path.join(model_folder, d)) and not d.startswith('_')]
        
        # Filter by env_filter if specified
        if self.env_filter:
            specified_envs = [e.strip() for e in self.env_filter.split(',')]
            envs = [e for e in all_envs if e in specified_envs]
            if not envs:
                raise RuntimeError(f"No matching environments found for filter: {self.env_filter}, available: {envs}")
        else:
            envs = all_envs
        
        return sorted(envs)
    
    def _run_command(self, cmd: str, log_file: str = None, timeout: int = 3600) -> Tuple[bool, str]:
        """Run a shell command and capture output."""
        log_info(f"Running: {cmd}")
        
        if log_file:
            log_info(f"📝 Log file: {log_file}")
        
        try:
            # Create log file and directory if specified
            if log_file:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                with open(log_file, 'w') as f:
                    f.write(f"Command: {cmd}\n")
                    f.write(f"Started: {datetime.now()}\n")
                    f.write("-" * 80 + "\n")
                
                # Redirect command output to log file using shell redirection
                # This ensures all output from the subprocess goes to the log
                cmd_with_redirect = f"{cmd} >> {log_file} 2>&1"
                
                # Run command with output redirection
                result = subprocess.run(
                    cmd_with_redirect,
                    shell=True,
                    text=True,
                    timeout=timeout
                )
                
                # Append completion info to log file
                with open(log_file, 'a') as f:
                    f.write("\n" + "-" * 80 + "\n")
                    f.write(f"Exit code: {result.returncode}\n")
                    f.write(f"Ended: {datetime.now()}\n")
                
                # Read the log file to get the output for return value
                with open(log_file, 'r') as f:
                    output = f.read()
                
                if result.returncode == 0:
                    return True, output
                else:
                    return False, output
            else:
                # If no log file, run normally with capture
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                if result.returncode == 0:
                    return True, result.stdout
                else:
                    return False, result.stderr or result.stdout
                
        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after {timeout} seconds"
            log_error(error_msg)
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"\nERROR: {error_msg}\n")
            return False, error_msg
        except Exception as e:
            error_msg = f"Command failed: {e}"
            log_error(error_msg)
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"\nERROR: {error_msg}\n")
            return False, str(e)
    
    def check_baseline(self) -> bool:
        # Check if baseline already exists
        all_exist = True
        for env in self.environments:
            benign_dir = os.path.join(self.folder, self.model, env, "iterations", "iter_0", "discussions", "benign")
            attack_dir = os.path.join(self.folder, self.model, env, "iterations", "iter_0", "discussions", "attack")
            if not os.path.exists(benign_dir) or not os.path.exists(attack_dir):
                all_exist = False
                break
        
        if all_exist:
            log_success("Some baseline attacks exists for all environments")
        else:
            raise RuntimeError("Benign and baseline attacks should be run first (--benign --attack).")
    
    def evolve_attacks(self, iteration: int, num_attacks: int = 5, top_k: int = 5) -> List[Dict]:
        """Evolve attacks based on previous iteration results."""
        # Check if attacks.json already exists
        attacks_file = os.path.join(get_iteration_dir(self.folder, iteration, self.evolution_name), "attacks.json")
        if os.path.exists(attacks_file):
            log_info(f"✓ Attacks already exist for iteration {iteration}: {attacks_file}")
            log_info(f"  Skipping attack evolution to avoid overwriting")
            attacks = load_attacks(self.folder, iteration, self.evolution_name)
            log_success(f"Loaded {len(attacks)} existing evolved attacks ({[e['name'] for e in attacks]})")
            return attacks
        
        log_info(f"Evolving {num_attacks} new attacks for iteration {iteration}")
        
        # Call the orchestrator with red_team_only mode
        self.orchestrator.run_iteration(
            iteration=iteration,
            environments=self.environments,
            num_new_attacks=num_attacks,
            num_new_defenses=0,  # Not used in red team only mode
            top_k_attacks=top_k,
            evolution_name=self.evolution_name,
            evolution_guidance=self.evolution_guidance,
            ignore_evolved_from_context=self.ignore_evolved_from_context,
            red_team_only=True,
            blue_team_only=False
        )
        
        # Load the evolved attacks
        attacks = load_attacks(self.folder, iteration, self.evolution_name)
        log_success(f"ATTACK EVOLUTION DONE")
        return attacks
    
    def evolve_defenses(self, iteration: int, num_defenses: int = 1) -> List[Dict]:
        """Evolve defenses based on attack results."""
        # Check if defenses.json already exists
        defenses_file = os.path.join(get_iteration_dir(self.folder, iteration, self.evolution_name), "defenses.json")
        if os.path.exists(defenses_file):
            log_info(f"✓ Defenses already exist for iteration {iteration}: {defenses_file}")
            log_info(f"  Skipping defense evolution to avoid overwriting")
            defenses = load_defenses(self.folder, iteration, self.evolution_name)
            log_success(f"Loaded {len(defenses)} existing defenses")
            return defenses
        
        log_info(f"Evolving {num_defenses} defenses for iteration {iteration}")
        
        # Call the orchestrator with blue_team_only mode
        self.orchestrator.run_iteration(
            iteration=iteration,
            environments=self.environments,
            num_new_attacks=0,  # Not used in blue team only mode
            num_new_defenses=num_defenses,
            top_k_attacks=0,  # Not used in blue team only mode
            evolution_name=self.evolution_name,
            evolution_guidance=self.evolution_guidance,
            ignore_evolved_from_context=self.ignore_evolved_from_context,
            blue_team_only=True,
            red_team_only=False
        )
        
        # Load the evolved defenses
        defenses = load_defenses(self.folder, iteration, self.evolution_name)
        log_success(f"DEFENSE EVOLUTION DONE")
        return defenses
    
    def run_attacks(self, iteration: int, defense_id: Optional[int] = None) -> Dict[str, Dict]:
        """Run attacks for an iteration, optionally with a defense.
        
        For iteration 0: runs baseline attacks only
        For iteration > 0: runs baseline + current iteration + all previous evolved attacks
        """
        if defense_id is not None:
            log_info(f"Running attacks for iteration {iteration} with defense {defense_id}")
        else:
            log_info(f"Running attacks for iteration {iteration} (no defense)")
        
        # Log which attacks will be run
        if iteration == 0:
            log_info("Running baseline attacks only")
        
        results = {}
        commands = []
        
        for env in self.environments:
            cmd = (f"{shlex.quote(sys.executable)} -m purpleaudit.framework {env} --attack "
                   f"--folder {self.folder} --model {self.model} "
                   f"--iteration {iteration} --evolution-name {self.evolution_name} "
                   f"--fixed-model {self.fixed_model}")
            
            # For iterations > 1, include all previous evolved attacks
            if iteration > 1:
                # Use --all-attacks to load baseline + all evolved attacks from all iterations
                cmd += " --all-attacks"
            
            if defense_id is not None:
                cmd += f" --defense-id {defense_id}"
            
            # Pass attacks filter for all iterations when specified
            if self.attacks_filter:
                cmd += f" --attacks {self.attacks_filter}"
            
            commands.append((env, cmd))
        
        # Run in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for env, cmd in commands:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                if defense_id is not None:
                    log_file = os.path.join(self.folder, self.model, env, "iterations", f"iter_{iteration}",
                                          "logs", f"{timestamp}_attack_defense{defense_id}.log")
                else:
                    log_file = os.path.join(self.folder, self.model, env, "iterations", f"iter_{iteration}",
                                          "logs", f"{timestamp}_attack.log")
                
                future = executor.submit(self._run_command, cmd, log_file)
                futures[future] = env
            
            # Collect results
            for future in as_completed(futures):
                env = futures[future]
                success, output = future.result()
                
                if success:
                    # Load and compute metrics
                    attack_results = load_attack_results(self.folder, self.model, env, iteration, defense_id, 
                                                        evolution_name=self.evolution_name)
                    metrics = compute_metrics(attack_results)
                    save_metrics(self.folder, self.model, env, iteration, defense_id, metrics, 
                               evolution_name=self.evolution_name)
                    results[env] = metrics
                    log_success(f"{env}: ASR={metrics['asr']:.1%}, Utility={metrics['utility']:.1%}")
                else:
                    raise RuntimeError(f"{env} failed: {output}")
        
        return results
    
    def run_iteration(self, iteration: int, num_attacks: int = 5, num_defenses: int = 1, top_k: int = 5) -> bool:
        """Run a complete iteration of the evolution pipeline."""
        log_separator(f"ITERATION {iteration}")
        
        # Step 1: Evolve attacks
        attacks = self.evolve_attacks(iteration, num_attacks, top_k)
        if not attacks:
            log_error("Failed to evolve attacks")
            return False
        
        # Step 2: Run evolved attacks (no defense)
        baseline_results = self.run_attacks(iteration, defense_id=None)
        
        # Step 3: Evolve defenses
        defenses = self.evolve_defenses(iteration, num_defenses)
        if not defenses:
            log_error("Failed to evolve defenses")
            return False
        
        # Step 4: Run attacks against each defense
        all_defense_results = {}
        for defense_idx in range(0, len(defenses)):
            log_info(f"Testing defense {defense_idx+1}/{len(defenses)} (defense_id {defense_idx}: {defenses[defense_idx].get('name', '')})")
            defense_results = self.run_attacks(iteration, defense_id=defense_idx)
            all_defense_results[defense_idx] = defense_results
        
        # Step 5: Save iteration summary
        summary = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'num_attacks': len(attacks),
            'num_defenses': len(defenses),
            'baseline_results': baseline_results,
            'defense_results': all_defense_results
        }
        
        summary_file = os.path.join(get_iteration_dir(self.folder, iteration, self.evolution_name), "summary.json")
        save_json_file(summary_file, summary)
        
        # Display summary
        log_separator("ITERATION SUMMARY")
        avg_baseline_asr = sum(m['asr'] for m in baseline_results.values()) / len(baseline_results)
        avg_baseline_utility = sum(m['utility'] for m in baseline_results.values()) / len(baseline_results)
        log_info(f"Baseline: ASR={avg_baseline_asr:.1%}, Utility={avg_baseline_utility:.1%}")
        
        for def_id, def_results in all_defense_results.items():
            avg_def_asr = sum(m['asr'] for m in def_results.values()) / len(def_results)
            avg_def_utility = sum(m['utility'] for m in def_results.values()) / len(def_results)
            log_info(f"Defense {def_id}: ASR={avg_def_asr:.1%}, Utility={avg_def_utility:.1%}")
        
        return True
    
    def run(self, num_iterations: int, num_attacks: int = 5, num_defenses: int = 1, top_k: int = 5) -> bool:
        """Run the complete evolution pipeline. Returns False if an iteration failed."""
        log_separator("RED TEAM-BLUE TEAM CO-EVOLUTION PIPELINE")
        log_info(f"Model: {self.model}")
        log_info(f"Environments: {', '.join(self.environments)}")
        log_info(f"Iterations: {num_iterations}")
        log_info(f"Evolution name: {self.evolution_name}")
        if self.attacks_filter:
            log_info(f"Baseline attacks: {self.attacks_filter}")
        
        self.check_baseline()
        
        # Run evolution iterations
        for iteration in range(1, num_iterations + 1):
            if not self.run_iteration(iteration, num_attacks, num_defenses, top_k):
                log_error(f"Failed at iteration {iteration}")
                return False
        
        log_separator("PIPELINE COMPLETE")
        log_success(f"Successfully completed {num_iterations} iterations")
        return True


def main():
    parser = argparse.ArgumentParser(description="Clean Red Team-Blue Team Co-Evolution Pipeline")
    
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations')
    parser.add_argument('--folder', default='environments', help='Base folder')
    parser.add_argument('--model', default='us.anthropic.claude-sonnet-4-5-20250929-v1:0', help='Model ID')
    parser.add_argument('--fixed-model', default='us.anthropic.claude-sonnet-4-5-20250929-v1:0', help='Fixed model ID')
    parser.add_argument('--region', default='us-west-2', help='AWS region for Bedrock')
    parser.add_argument('--evolution-name', default='default', help='Evolution name')
    parser.add_argument('--num-attacks', type=int, default=3, help='Number of attacks per iteration')
    parser.add_argument('--num-defenses', type=int, default=1, help='Number of defenses per iteration')
    parser.add_argument('--top-k', type=int, default=5, help='Top-k attacks to keep')
    parser.add_argument('--max-workers', type=int, default=5, help='Max parallel workers')
    parser.add_argument('--attacks', default='', help='Comma-separated list of baseline attacks')
    parser.add_argument('--env', default='', help='Specific environment(s) to run (comma-separated) or empty for all')
    parser.add_argument('--evolution-guidance', default='', 
                       help='Instructions to guide Red or Blue Team agent evolution strategy')
    parser.add_argument('--ignore-evolved-from-context', action='store_true',
                       help='When selecting top-k attacks, ignore attacks that start with "evolved_"')
    parser.add_argument('--blue-team-only', action='store_true',
                       help='Blue team only mode: evolve defenses against hardcoded attacks')
    parser.add_argument('--red-team-only', action='store_true',
                       help='Red team only mode: evolve attacks with no defenses')
    
    args = parser.parse_args()
    
    pipeline = EvolutionPipeline(
        folder=args.folder,
        model=args.model,
        fixed_model=args.fixed_model,
        evolution_name=args.evolution_name,
        max_workers=args.max_workers,
        attacks_filter=args.attacks,
        env_filter=args.env,
        region=args.region,
        evolution_guidance=args.evolution_guidance,
        ignore_evolved_from_context=args.ignore_evolved_from_context,
        blue_team_only=args.blue_team_only,
        red_team_only=args.red_team_only
    )
    
    ok = pipeline.run(
        num_iterations=args.iterations,
        num_attacks=args.num_attacks,
        num_defenses=args.num_defenses,
        top_k=args.top_k
    )
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    # Add parent directory to path to import utils
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
