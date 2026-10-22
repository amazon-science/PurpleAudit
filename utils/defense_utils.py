# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Defense utilities for wrapping tool functions with defense mechanisms.

This module provides utilities to apply defense wrappers to tool functions,
similar to how attack_utils.py handles attack injection.
"""

from typing import Callable, Any, List
import functools


def defense_tool_wrapper(function: Callable, defense_wrappers: List[Callable]) -> Callable:
    """
    Wrapper that applies multiple defense wrappers to a tool function's output.
    
    Defense wrappers are applied in the order they are provided in the list.
    For example, if defense_wrappers = [wrapper1, wrapper2], the output will be:
        wrapper2(wrapper1(original_output))
    
    Args:
        function: The tool function to wrap
        defense_wrappers: List of defense wrapper functions to apply to the output
    
    Returns:
        Callable: The wrapped function
    """
    if not defense_wrappers:
        return function
    
    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Call the original function
        output = function(*args, **kwargs)
        
        # Apply each defense wrapper in sequence
        result = str(output)
        for defense_wrapper in defense_wrappers:
            result = defense_wrapper(result)
        
        return result
    
    return wrapper


def create_defense_wrapper_from_config(defense_config: dict, original_prompt: str = None) -> Callable:
    """
    Create a combined defense wrapper function from a defense configuration.
    
    This combines multiple defense wrappers into a single wrapper function that
    can be applied to tool outputs.
    
    Args:
        defense_config: Defense configuration dict with 'defenses' key containing
                       list of Defense objects
        original_prompt: The original user prompt (needed for prompt_sandwiching defense)
    
    Returns:
        Callable: Combined wrapper function, or None if no wrappers needed
    """
    if not defense_config or 'defenses' not in defense_config:
        return None
    
    # Set original prompt for any PromptSandwichingDefense instances
    if original_prompt:
        for defense in defense_config['defenses']:
            if hasattr(defense, 'set_original_prompt'):
                defense.set_original_prompt(original_prompt)
    
    # Collect all tool output wrappers from defenses
    wrappers = []
    for defense in defense_config['defenses']:
        wrapper = defense.get_tool_output_wrapper()
        if wrapper:
            wrappers.append(wrapper)
    
    if not wrappers:
        return None
    
    # Return a combined wrapper function
    def combined_wrapper(output: str) -> str:
        result = output
        for wrapper in wrappers:
            result = wrapper(result)
        return result
    
    return combined_wrapper
