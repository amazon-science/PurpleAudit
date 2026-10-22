# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import sys
import traceback
import json
import argparse
from dotenv import load_dotenv
load_dotenv()

import purpleaudit.attacks as attacks_module
from purpleaudit.utils.logger import log_success, log_info, log_separator, log_warning, log_error
from purpleaudit.utils.bedrock_config import BedrockLLMConfig
from purpleaudit.core import MultiAgentSystemLab, MultiAgentSystemBuilder, create_mas
from purpleaudit.core.mas_magentic_one import MagenticOneMultiAgentSystem
from purpleaudit.core.mas_lab_magentic_one import MagenticOneSystemLab
from purpleaudit.utils.attack_utils import get_specific_attacks
from purpleaudit.utils.utils import extract_project_name

MAGENTIC_ONE_ENVS = ("assistant_bench", "assistantbench", "gaia")


def load_hardcoded_attacks_with_filter(attacks_filter: str = ""):
    """Load hardcoded attacks with optional filtering
    
    Args:
        attacks_filter: Comma-separated list of attack names to include (empty = all)
        
    Returns:
        Dictionary mapping attack names to attack functions
    """
    # Parse attacks filter - if empty, load all hardcoded attacks
    attacks_list = [a.strip() for a in attacks_filter.split(",") if a.strip()] if attacks_filter.strip() else []
    
    if not attacks_list:
        # Load all registered attacks from the ATTACKS registry (populated by @register_attack decorator)
        # This ensures we only get actual attack functions, not other callables like imported types
        attacks = dict(attacks_module.ATTACKS)
    else:
        attacks = get_specific_attacks(attacks_module, attacks=attacks_list)
    
    return attacks


def get_successful_baseline_attacks(folder: str, env_name: str, model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0") -> set:
    """Get set of attack names that succeeded at least once in baseline (iteration 0)
    
    Args:
        folder: Base folder for environments
        env_name: Environment name
        model_id: Model ID subdirectory within folder
        
    Returns:
        Set of attack names that have attack_success=True at least once
    """
    successful_attacks = set()
    
    # Use the updated load_attack_results function to handle both blue-team and red-team modes
    from purpleaudit.evolution.utils import load_attack_results
    
    try:
        results_by_env = load_attack_results(
            folder=folder,
            environments=[env_name],
            model_id=model,
            iteration=0,
            evolution_name="default"
        )
        
        env_results = results_by_env.get(env_name, [])
        
        for result in env_results:
            if result.get('attack_success', False):
                attack_name = result.get('attack_name', '')
                if attack_name:
                    successful_attacks.add(attack_name)
    except Exception as e:
        # Fallback to original logic if load_attack_results fails
        attack_dir = os.path.join(folder, model, env_name, "iterations", "iter_0", "discussions", "attack")
        if os.path.exists(attack_dir):
            # Iterate through all attack subdirectories
            for attack_name in os.listdir(attack_dir):
                attack_path = os.path.join(attack_dir, attack_name)
                if not os.path.isdir(attack_path):
                    continue
                
                # Check all vector subdirectories (tool, prompt, etc.)
                for vector in os.listdir(attack_path):
                    vector_path = os.path.join(attack_path, vector)
                    if not os.path.isdir(vector_path):
                        continue
                    
                    # Check all attack files in this vector
                    for attack_file in os.listdir(vector_path):
                        if not attack_file.endswith('.json'):
                            continue
                        
                        attack_file_path = os.path.join(vector_path, attack_file)
                        try:
                            with open(attack_file_path, 'r') as f:
                                attack_data = json.load(f)
                                
                                # Check if this attack succeeded
                                if attack_data.get('attack_success', False):
                                    # Extract attack name from the attack_data
                                    attack_name_from_data = attack_data.get('attack_name', '')
                                    if attack_name_from_data:
                                        successful_attacks.add(attack_name_from_data)
                        except Exception as inner_e:
                            continue
    
    return successful_attacks


def load_evolved_attacks(folder: str, iteration: int, model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                        evolution_name: str = "default"):
    """Load evolved attacks from an iteration
    
    Simply loads all attacks from attacks.json at the specified iteration.
    No filtering is applied - evolved attacks are always loaded in full.
    
    Args:
        folder: Base folder
        iteration: Iteration number
        model: Model ID subdirectory within folder
        evolution_name: Name of the evolution experiment
    """
    # Load from shared evolution folder (model-independent)
    from purpleaudit.evolution.utils import load_evolution_file
    
    evolved_attacks = load_evolution_file(
        folder=folder,
        iteration=iteration,
        filename="attacks.json",
        model_id=model,
        fallback_to_legacy=True,
        evolution_name=evolution_name
    )
    
    if evolved_attacks is None:
        return None
    
    attack_dict = {}
    
    for attack in evolved_attacks:
        attack_name = attack.get('name')
        
        if attack.get('type') == 'hardcoded':
            # Reference to a hardcoded attack
            if hasattr(attacks_module, attack_name):
                attack_dict[attack_name] = getattr(attacks_module, attack_name)
        
        elif attack.get('type') == 'generated':
            # Keep generated attacks as dictionaries - mas_lab.py expects them this way
            # and will use InjectionAgent to apply them dynamically
            attack_dict[attack_name] = attack
    
    return attack_dict if attack_dict else None


def load_all_evolved_attacks(folder: str, model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0", include_hardcoded: bool = True, attacks_filter: str = "", evolution_name: str = "default"):
    """Load all evolved attacks from all iterations
    
    Args:
        folder: Base folder
        model: Model ID subdirectory within folder
        include_hardcoded: Whether to include hardcoded attacks from attacks.py
        attacks_filter: Comma-separated list of attack names to include (empty = all)
        evolution_name: Name of the evolution experiment
        
    Returns:
        Dictionary mapping attack names to attack definitions (functions or dicts)
    """
    attack_dict = {}
    filter_names = set(attacks_filter.split(",")) if attacks_filter else None
    
    # Add hardcoded attacks first if requested
    if include_hardcoded:
        for name in dir(attacks_module):
            if not name.startswith('_') and callable(getattr(attacks_module, name)):
                # Apply filter if specified
                if filter_names and name not in filter_names:
                    continue
                attack_dict[name] = getattr(attacks_module, name)
    
    # Find all iteration directories in shared evolution folder
    iterations_dir = os.path.join(folder, "_evolution", evolution_name, "iterations")
    
    # Check if iterations directory exists
    if not os.path.exists(iterations_dir):
        return attack_dict if attack_dict else None
    
    # Load attacks from each iteration
    for iter_dir in sorted(os.listdir(iterations_dir)):
        if iter_dir.startswith("iter_"):
            try:
                iteration = int(iter_dir.split("_")[1])
                iteration_attacks = load_evolved_attacks(folder, iteration, model=model, evolution_name=evolution_name)
                if iteration_attacks:
                    attack_dict.update(iteration_attacks)
            except (ValueError, IndexError):
                continue
    
    return attack_dict if attack_dict else None


def run_factory(args, bedrock_config):
    """Generate MAS environments using factory"""
    log_separator("FACTORY MODE - MAS GENERATION")
    
    # Set environment variable for tools.py to use
    os.environ['PURPLEAUDIT_MODEL_ID'] = args.model
    os.environ['PURPLEAUDIT_OUTPUT_FOLDER'] = args.folder
    
    builder = MultiAgentSystemBuilder(
        args.factory_env,
        bedrock_config,
        folder="factories",
        model_id=args.model,  # Use the same model as specified in args
        system_prompt_variables={
            "{OUT_FOLDER}": f"{args.folder}/{args.model}",  # Include model in output path
            "{NUM_GENERATED_PROMPTS}": args.num_prompts,
        },
    )
    builder.load()
    
    results = builder.run(prompt=args.prompt or None)
    project_names = [extract_project_name(r) for r in results]
    
    log_success(f"Generated {len(project_names)} MAS environments")
    for name in project_names:
        log_info(f"  - {name}")
    
    return project_names


def run_lab(args, bedrock_config, environments):
    """Run benign/attack on existing environments. Returns the environments that failed."""
    log_separator(f"{args.mode.upper()} MODE")
    
    failed = []
    for env in environments:
        # One broken environment must not stop a campaign over the others
        try:
            _run_env(args, bedrock_config, env)
        except Exception as e:
            log_error(f"{env} ({args.mode}) failed: {e}\n{traceback.format_exc()}")
            failed.append(env)
    return failed


def _run_env(args, bedrock_config, env):
    """Run the current mode (benign/attack/eval) on one environment"""
    # Check if this is a MagenticOne environment
    use_magentic_one = env.lower() in MAGENTIC_ONE_ENVS

    # Load defense if specified
    defense_config = None
    defense_id = None

    if args.defense_id is not None:
        # Try to parse as integer (evolved defense)
        try:
            defense_id = int(args.defense_id)
            # Load evolved defense from defenses.json
            from purpleaudit.evolution.utils import load_evolution_file
            defenses = load_evolution_file(
                folder=args.folder,
                iteration=args.iteration,
                filename="defenses.json",
                evolution_name=args.evolution_name
            )

            if defenses and 0 <= defense_id < len(defenses):
                defense = defenses[defense_id]
                # Create a simple config with the defense prompt to append
                defense_config = {
                    'defense_name': defense.get('name', f'defense_{defense_id}'),
                    'defense_prompt': defense.get('defense_prompt', ''),
                    'description': defense.get('description', '')
                }
                log_info(f"Using evolved defense {defense_id}: {defense.get('name', 'unknown')}")
                if use_magentic_one:
                    log_warning("MagenticOne has limited defense support - only system prompt additions are applied")
            else:
                raise RuntimeError(f"Defense {defense_id} not found in iteration {args.iteration}")

        except ValueError:
            # It's a string, treat as hardcoded defense name
            defense_id = args.defense_id  # Keep as string
            from purpleaudit.defenses import create_defense_config
            defense_config = create_defense_config([args.defense_id])
            log_info(f"Using hardcoded defense: {args.defense_id}")
            if use_magentic_one:
                # Check if this defense is supported in MagenticOne
                supported_defenses = ["spotlighting", "naive_safeguard", "tool_output_extraction"]
                if args.defense_id in supported_defenses:
                    log_warning("MagenticOne has limited defense support - only system prompt additions are applied to orchestrator")
                else:
                    log_warning(f"Defense '{args.defense_id}' is not supported in MagenticOne - will attempt but may fail")

    # Create the appropriate MAS type
    if use_magentic_one:
        log_info(f"Using MagenticOne for benchmark environment: {env}")
        mas = MagenticOneMultiAgentSystem(env, bedrock_config, folder=args.folder,
                                           defense_config=defense_config,
                                           model_id=args.model,
                                           max_turns=8) # setting higher risks too-many-tokens error
    else:
        # Adapter is selected by the "framework" field in environment.json
        # (defaults to "ag2" for backwards compatibility).
        mas = create_mas(
            environment_name=env,
            bedrock_config=bedrock_config,
            folder=args.folder,
            defense_config=defense_config,
            model_id=args.model,
            max_rounds=15,
        )

    mas.load()

    # Load attacks based on mode
    if args.all_attacks:
        # Load all attacks (hardcoded + all iterations)
        attacks = load_all_evolved_attacks(args.folder, model=args.model, include_hardcoded=True, attacks_filter=args.attacks, evolution_name=args.evolution_name)
        if attacks:
            log_info(f"Loaded {len(attacks)} attacks (hardcoded + all iterations)")
        else:
            log_info("No attacks found, using default hardcoded attacks")
            attacks = load_hardcoded_attacks_with_filter(args.attacks)
    elif args.evolved_only:
        # Load only evolved attacks (no hardcoded)
        attacks = load_all_evolved_attacks(args.folder, model=args.model, include_hardcoded=False, attacks_filter=args.attacks, evolution_name=args.evolution_name)
        if attacks:
            log_info(f"Loaded {len(attacks)} evolved attacks (no hardcoded)")
        else:
            log_info("No evolved attacks found")
            attacks = {}
    elif args.iteration >= 0:
        # Determine if we should filter by successful attacks only
        # This is useful in blue-team-only mode for iterations > 0
        successful_only = None
        if args.successful_only and args.iteration == 0:
            # For baseline (iteration 0), we need to evaluate ALL attacks to know which succeed
            successful_only = None
        elif args.successful_only and args.iteration > 0:
            # For iterations > 0 in blue-team-only mode, collect successful attacks from both:
            # 1. Baseline (hardcoded attacks from iteration 0)
            # 2. Current iteration (evolved attacks)
            successful_only = set()

            # Get successful baseline attacks
            baseline_successful = get_successful_baseline_attacks(args.folder, env, args.model)
            if baseline_successful:
                successful_only.update(baseline_successful)
                log_info(f"Found {len(baseline_successful)} successful baseline attacks")

            # Get successful evolved attacks from current iteration
            try:
                from purpleaudit.evolution.utils import load_attack_results
                evolved_results = load_attack_results(
                    folder=args.folder,
                    environments=[env],
                    model_id=args.model,
                    iteration=args.iteration,
                    evolution_name=args.evolution_name
                )

                evolved_successful = set()
                env_results = evolved_results.get(env, [])
                for result in env_results:
                    if result.get('attack_success', False):
                        attack_name = result.get('attack_name', '')
                        if attack_name:
                            evolved_successful.add(attack_name)

                if evolved_successful:
                    successful_only.update(evolved_successful)
                    log_info(f"Found {len(evolved_successful)} successful evolved attacks from iteration {args.iteration}")

            except Exception as e:
                log_info(f"Could not load evolved attack results: {e}")

            if successful_only:
                log_info(f"Filtering to {len(successful_only)} total successful attacks (baseline + evolved)")
            else:
                log_info("Warning: No successful attacks found, will use all attacks")
                successful_only = None

        # Load attacks from both baseline and evolved iterations for blue-team-only mode
        attacks = {}

        if args.iteration > 0:
            log_info(f"Loading attacks from both baseline (iteration 0) and evolved (iteration {args.iteration}) for comprehensive evaluation")

            # First, load baseline attacks (hardcoded attacks from attacks.py)
            # Baseline attacks are stored as discussion files in model directories, not as JSON in _evolution
            # So we directly use hardcoded attacks as the baseline
            log_info("Loading hardcoded attacks as baseline attacks")

            hardcoded_attacks = load_hardcoded_attacks_with_filter(args.attacks)

            # Filter hardcoded attacks if successful_only is specified
            if successful_only is not None:
                filtered_hardcoded = {}
                for attack_name, attack_func in hardcoded_attacks.items():
                    if attack_name in successful_only:
                        filtered_hardcoded[attack_name] = attack_func
                hardcoded_attacks = filtered_hardcoded
                log_info(f"Filtered to {len(hardcoded_attacks)} successful hardcoded baseline attacks")

            attacks.update(hardcoded_attacks)
            log_info(f"Loaded {len(hardcoded_attacks)} baseline (hardcoded) attacks")

            # Second, load evolved attacks from current iteration (no filtering - load all evolved attacks)
            evolved_attacks = load_evolved_attacks(args.folder, args.iteration, model=args.model, evolution_name=args.evolution_name)
            if evolved_attacks:
                # Check for name conflicts and handle them
                evolved_count = 0
                for attack_name, attack_def in evolved_attacks.items():
                    if attack_name in attacks:
                        # If there's a name conflict, prioritize the evolved attack
                        log_info(f"Attack name conflict: '{attack_name}' exists in both baseline and evolved. Using evolved version.")
                    attacks[attack_name] = attack_def
                    evolved_count += 1
                log_info(f"Loaded {evolved_count} evolved attacks from iteration {args.iteration}")
            else:
                log_info(f"No evolved attacks found for iteration {args.iteration}")

            total_count = len(attacks)
            log_info(f"Total attacks available for evaluation: {total_count} (baseline + evolved)")

            # If we have a filter but no baseline attacks matched, raise an error
            if args.attacks and len(hardcoded_attacks) == 0:
                error_msg = f"No baseline attacks matched filter '{args.attacks}'"
                raise ValueError(error_msg)
        else:
            # For iteration 0, just load baseline attacks
            log_info(f"Loading baseline attacks from iteration 0")
            attacks = load_evolved_attacks(args.folder, 0, model=args.model, evolution_name=args.evolution_name)
            if attacks:
                log_info(f"Loaded {len(attacks)} baseline attacks from iteration 0")
            else:
                # For iteration 0, fall back to hardcoded attacks
                log_info(f"No evolved attacks found for iteration 0, using hardcoded attacks")
                attacks = load_hardcoded_attacks_with_filter(args.attacks)
                log_info(f"Loaded {len(attacks)} hardcoded attacks")

        # Verify we have attacks to run
        if not attacks:
            error_msg = f"No attacks available for iteration {args.iteration}. Expected attacks from baseline and/or evolved sources."
            log_info(f"ERROR: {error_msg}")
            raise FileNotFoundError(error_msg)
    else:
        attacks = load_hardcoded_attacks_with_filter(args.attacks)

    # Print loaded attacks in inference mode
    if args.inference and attacks:
        log_separator("LOADED ATTACKS (INFERENCE MODE)")
        log_info(f"Total attacks loaded: {len(attacks)}")
        for i, attack_name in enumerate(sorted(attacks.keys()), 1):
            attack = attacks[attack_name]
            attack_type = "generated" if isinstance(attack, dict) else "hardcoded"
            log_info(f"  {i}. {attack_name} ({attack_type})")
        log_separator()

    # Create the appropriate lab type
    if use_magentic_one:
        # MagenticOne only supports tool vector
        lab = MagenticOneSystemLab(mas, args.mode, vectors=args.vectors.split(","),
                                    attacks=attacks, injection=args.injection, flag=args.flag,
                                    iteration=args.iteration, defense_id=args.defense_id,
                                    fixed_model_id=args.fixed_model,
                                    inference_mode=args.inference,
                                    evolution_name=args.evolution_name,
                                    model=args.model)
    else:
        lab = MultiAgentSystemLab(mas, args.mode, vectors=args.vectors.split(","), 
                                  attacks=attacks, injection=args.injection, flag=args.flag,
                                  iteration=args.iteration, defense_id=args.defense_id,
                                  fixed_model_id=args.fixed_model, 
                                  inference_mode=args.inference,
                                  evolution_name=args.evolution_name,
                                  model=args.model)
    if args.mode in ["benign", "attack"]:
        lab.run()
    
    return lab.evaluate()


def main(args):
    """Main function - unified factory and lab"""
    
    # Step 1: Generate MAS if requested
    generated_envs = []
    if args.generate:
        # For factory, use fixed_model
        factory_bedrock_config = BedrockLLMConfig(
            region_name=args.region, 
            model_id=args.fixed_model,
        )
        generated_envs = run_factory(args, factory_bedrock_config)
    
    # Determine which environments to use
    if generated_envs:
        environments = generated_envs
    elif args.environments:
        if args.environments == "all":
            # Environments defined for this model or in default/, plus MagenticOne benchmarks already run
            environments = set()
            for base in (os.path.join(args.folder, args.model), os.path.join(args.folder, "default")):
                if os.path.isdir(base):
                    environments.update(
                        d for d in os.listdir(base)
                        if not d.startswith('_') and (os.path.exists(os.path.join(base, d, "environment.json"))
                                                      or d.lower() in MAGENTIC_ONE_ENVS)
                    )
            environments = sorted(environments)
            if not environments:
                print(f"No environments found in {args.folder}/{args.model} or {args.folder}/default")
        else:
            environments = args.environments.split(",")
    else:
        environments = []

    # For MAS conversations, use args.model
    mas_bedrock_config = BedrockLLMConfig(
        region_name=args.region, 
        model_id=args.model,
    )
    
    failed = []
    # Step 2: Run benign if requested
    if args.benign and environments:
        args.mode = "benign"
        failed += run_lab(args, mas_bedrock_config, environments)
    
    # Step 3: Run attack if requested
    if args.attack and environments:
        args.mode = "attack"
        failed += run_lab(args, mas_bedrock_config, environments)
    
    # Step 4: Run eval if requested (standalone, not combined with benign/attack)
    if args.eval and environments:
        args.mode = "eval"
        failed += run_lab(args, mas_bedrock_config, environments)
    
    log_separator()
    if failed:
        log_error(f"Failed: {', '.join(sorted(set(failed)))} (see the errors above)")
        sys.exit(1)
    log_success("Execution completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Agent System Factory and Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Core args
    parser.add_argument("environments", nargs='?', default=None,
                       help="Environment name(s) comma-separated, or 'all' (not needed if only --generate)")
    
    # Mode selection
    parser.add_argument("--generate", action="store_true",
                       help="Generate new MAS environments")
    parser.add_argument("--benign", action="store_true",
                       help="Run benign discussions")
    parser.add_argument("--attack", action="store_true",
                       help="Run attack campaigns")
    parser.add_argument("--eval", action="store_true",
                       help="Evaluate only (no generation)")
    
    # Factory args (for --generate)
    parser.add_argument("--factory-env", default="factory",
                       help="Factory environment to use for generation")
    parser.add_argument("--num-prompts", type=int, default=30,
                       help="Number of benign prompts to generate per MAS")
    parser.add_argument("--prompt", default="",
                       help="Specific prompt name to generate MAS from")
    
    # Lab args
    parser.add_argument("--injection", default="", 
                       help="Injection text for evaluation")
    parser.add_argument("--flag", default="",
                       help="Flag for attack success evaluation")
    parser.add_argument("--vectors", default="tool",
                       help="Attack vectors: prompt,tool")
    parser.add_argument("--attacks", default="",
                       help="Specific attacks to run (comma-separated)")
    parser.add_argument("--all-attacks", action="store_true",
                       help="Load and run ALL attacks (hardcoded + all evolved iterations)")
    parser.add_argument("--evolved-only", action="store_true",
                       help="Load and run ONLY evolved attacks from all iterations (no hardcoded)")
    parser.add_argument("--inference", action="store_true",
                       help="Inference mode: run attacks without writing any discussion files (read-only evaluation)")
    
    # Evolution args
    parser.add_argument("--iteration", type=int, default=0,
                       help="Iteration number for evolved attacks/defenses (default: 0 for baseline)")
    parser.add_argument("--defense-id", type=str, default=None,
                       help="Defense variant ID (int for evolved, e.g. 0, 1, 2) or name (str for hardcoded, e.g. 'spotlighting')")
    parser.add_argument("--successful-only", action="store_true",
                       help="For iterations > 0, only test attacks that succeeded in baseline (useful for blue-team-only mode)")
    parser.add_argument("--evolution-name", default="default",
                       help="Name of the evolution experiment (default: 'default')")
    
    # System args
    parser.add_argument("--folder", default="environments",
                       help="Output/input folder for environments")
    parser.add_argument("--region", default="us-west-2",
                       help="AWS region for Bedrock")
    parser.add_argument("--model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                       help="Bedrock model ID for generated conversations (also used for organizing experiment folders)")
    parser.add_argument("--fixed-model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                       help="Bedrock model ID for pipeline management (factory, judge, agents)")
    
    # Wandb
    parser.add_argument("--wandb", action="store_true",
                       help="Enable wandb tracking (LOCAL only)")
    
    args = parser.parse_args()
    
    # Validation
    if not args.generate and not args.benign and not args.attack and not args.eval:
        parser.error("At least one mode flag is required: --generate, --benign, --attack, or --eval")
    
    if (args.benign or args.attack or args.eval) and not args.generate and not args.environments:
        parser.error("environments argument is required when running benign/attack/eval (unless using --generate)")
    
    main(args)
