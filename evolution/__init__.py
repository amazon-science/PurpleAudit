# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Evolution module for red team-blue team genetic framework.

This module implements a genetic/evolutionary approach for adversarial 
attack-defense optimization in multi-agent systems.
"""

from .agents import RedTeamAgent, BlueTeamAgent
from .injection_agent import InjectionAgent
from .defense_agent import DefenseAgent
from .attack_manager import AttackManager
from .defense_manager import DefenseManager
from .orchestrator import EvolutionOrchestrator

__all__ = [
    'RedTeamAgent',
    'BlueTeamAgent',
    'InjectionAgent',
    'DefenseAgent',
    'AttackManager',
    'DefenseManager',
    'EvolutionOrchestrator',
]
