# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""PurpleAudit entrypoint: run the support MAS on one task, on the model under audit.

PurpleAudit instruments LangGraph while this runs (tool outputs, system prompts,
transcript), so the app itself is unchanged.
"""

from support_app import answer


def run(task, ctx):
    return answer(task, ctx.chat_model())
