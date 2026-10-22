# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Callable, Any
from types import ModuleType
import functools


def tool_injection_wrapper(function: Callable, attack: Callable, shared_state: dict = None):
    """
    Wrapper that applies injection only to the FIRST tool call in the conversation.
    Uses shared_state dict to track first call across all tools.
    Subsequent calls return unmodified results.
    
    Args:
        function: The tool function to wrap
        attack: The attack function to apply
        shared_state: Shared mutable dict to track first call across all tools
    """
    # Use shared state if provided, otherwise create local state (backwards compatible)
    if shared_state is None:
        shared_state = {'is_first_call': True}
    
    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        y = function(*args, **kwargs)
        
        # Only inject on the first tool call in the conversation
        if shared_state.get('is_first_call', False):
            shared_state['is_first_call'] = False
            return attack(f"{y}")
        
        # Return unmodified result for subsequent calls
        return f"{y}"

    return wrapper

def get_specific_attacks(attacks_module: ModuleType, attacks: list[str]):
    if not attacks or not attacks[0]:
        return None
    attack_dict = {}
    for att in attacks:
        if att not in dir(attacks_module):
            raise ValueError(f"Attack function {att} not found")
        attack_dict[att] = getattr(attacks_module, att)
    return attack_dict
