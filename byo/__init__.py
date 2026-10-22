# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Bring your own MAS: audit an existing multi-agent system with PurpleAudit.

Point ``environment.json`` at an entrypoint that runs your MAS on one task::

    {"entrypoint": "audit:run"}

    # audit.py
    from my_app import answer

    def run(task: str, ctx: AuditContext):
        return answer(task, ctx.chat_model())

While it runs, PurpleAudit instruments LangChain / LangGraph, CrewAI and AG2 (see
``instrument``); other code uses ``ctx.wrap_tools`` / ``ctx.system_prompt``.
See the README and ``examples/``.
"""

from .context import AuditContext

__all__ = ["AuditContext"]
