# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""PurpleAudit: co-evolutionary auditing framework for task hijacking in multi-agent systems."""

import os

from dotenv import find_dotenv, load_dotenv

# API keys: a .env in the directory you run from, else the one at the repo root.
# Explicit paths, because python-dotenv's default lookup depends on how Python was
# started (with ``python -m``, the package loads before its caller is known).
load_dotenv(find_dotenv(usecwd=True))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from .core import MultiAgentSystem, MultiAgentSystemLab, MultiAgentSystemBuilder
from .attacks import ATTACKS

__version__ = "1.0.0"
__all__ = [
    "MultiAgentSystem", 
    "MultiAgentSystemLab", 
    "MultiAgentSystemBuilder",
    "ATTACKS",
]
