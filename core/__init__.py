# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Core module for PurpleAudit multi-agent systems."""

from .mas import MultiAgentSystem
from .mas_builder import MultiAgentSystemBuilder
from .mas_lab import MultiAgentSystemLab
from .mas_eval import global_evaluation, evaluate_attack_summary
from .mas_magentic_one import MagenticOneMultiAgentSystem
from .mas_lab_magentic_one import MagenticOneSystemLab
from .adapters import BaseMASAdapter, create_mas

__all__ = [
    "MultiAgentSystem",
    "MultiAgentSystemBuilder",
    "MultiAgentSystemLab",
    "MagenticOneMultiAgentSystem",
    "MagenticOneSystemLab",
    "global_evaluation",
    "evaluate_attack_summary",
    "BaseMASAdapter",
    "create_mas",
]
