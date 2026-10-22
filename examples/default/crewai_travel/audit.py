# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""PurpleAudit entrypoint: run the travel crew on one task, on the model under audit.

PurpleAudit instruments CrewAI while this runs (tool outputs, system prompts,
transcript), so the crew itself is unchanged.
"""

from travel_crew import answer


def run(task, ctx):
    return answer(task, ctx.crewai_llm())
