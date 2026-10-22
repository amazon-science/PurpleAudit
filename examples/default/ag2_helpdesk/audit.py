# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""PurpleAudit entrypoint: run the helpdesk on one task, on the model under audit.

PurpleAudit instruments AG2 while this runs (tool outputs, system prompts,
transcript), so the app itself is unchanged.
"""

from helpdesk_app import run_helpdesk


def run(task, ctx):
    return run_helpdesk(task, ctx.ag2_llm_config())
