# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
AG2 (a.k.a. autogen) adapter.

The existing ``MultiAgentSystem`` in ``purpleaudit.core.mas`` already implements
the contract the rest of the codebase relies on, so the AG2 adapter is just an
alias that registers it under the new factory.

Keeping it separate lets us change the AG2 implementation later without
touching the public adapter API, and makes the framework registry symmetric
with the other adapters.
"""

from purpleaudit.core.mas import MultiAgentSystem

from .base import BaseMASAdapter

# Make plain MultiAgentSystem instances also satisfy isinstance(x, BaseMASAdapter)
# without changing the original class's bases.
BaseMASAdapter.register(MultiAgentSystem)


class AG2MASAdapter(MultiAgentSystem):
    """AG2 MAS adapter — direct passthrough to the original implementation."""

    framework_name = "ag2"


__all__ = ["AG2MASAdapter"]
