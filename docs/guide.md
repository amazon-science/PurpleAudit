# PurpleAudit guide

The [README](../README.md) covers installation, the quick start and branching your MAS. This guide covers the details.

## LLM providers

PurpleAudit's own agents (judge, red/blue team, injection agent) and the model offered to your MAS (`ctx.chat_model()`, ...) go through one provider, set in `.env` or the environment:

```bash
OPENROUTER_API_KEY=sk-or-v1-...      # OpenRouter, used whenever this key is set
# or AWS Bedrock (the paper's setup): run `aws configure`, region us-west-2
LLM_PROVIDER=openrouter              # optional: force openrouter or bedrock
OPENROUTER_BASE_URL=...              # optional: a proxy
```

`--model` (the model under audit) and `--fixed-model` (PurpleAudit's own agents) default to Claude Sonnet 4.5 and take Bedrock IDs, translated to OpenRouter slugs when needed (`us.anthropic.claude-sonnet-4-5-20250929-v1:0` → `anthropic/claude-sonnet-4.5`), or any OpenRouter slug directly (e.g. `qwen/qwen3-235b-a22b-2507`; the `/` becomes a subfolder of the results folder).

## How an audit works

1. **Benign runs**: your MAS solves each task of `prompts.json`; a task is solved when its `ground_truth` appears in the final answer. The agents' system prompts and tools are recorded in `agents.json` for the co-evolution blue team.
2. **Attack runs**: for each solved task and each attack, your MAS runs the task again while the attack rewrites the **first tool output** to ask another solved task's question. The attack succeeds if the MAS answers it (checked by an LLM judge).
3. **Defense runs** (`--defense-id`): the same attacks with a defense added to every agent's system prompt and/or wrapped around every tool output: `spotlighting`, `prompt_sandwiching`, `tool_filtering` (disables the tools an agent does not need for the task), `naive_safeguard`, `tool_output_extraction`.
4. **Co-evolution** (`purpleaudit.evolve`): each iteration, the red team writes new attacks from those that worked, runs them on your MAS, the blue team writes a defense prompt from your agents' prompts, and the attacks run again under it. It stops early when the defense blocks every attack.

While your entrypoint runs, PurpleAudit instruments the agent framework (nothing happens outside audited runs):

| | Tool outputs (attacks, tool-level defenses) | System prompts (prompt-level defenses) | Transcript |
| --- | --- | --- | --- |
| **LangChain / LangGraph** | every `BaseTool` call | the system message of every chat-model call (`invoke`, `batch`, `stream`) | each model reply, named after its agent (graph `name=` or node) |
| **CrewAI** | every tool call | each agent's backstory while it runs a task (crews, `Agent.kickoff`, Flows) | each agent's task output, delegations included |
| **AG2** | every function execution | each agent's system message while it replies | each agent's reply |

## Your MAS

### Entrypoint

`environment.json` names the entrypoint: `{"entrypoint": "audit:run"}`. The module is looked up in the environment folder first (with that folder on `sys.path`, so it can import your code next to it), then as an installed module (`"my_package.audit:run"`). `run(task, ctx)` may be `async`, and returns a string, a LangGraph state, a CrewAI `CrewOutput`, an AG2 `ChatResult` (or the tuple of `initiate_group_chat`), or `{"final_result": ..., "messages": [...]}`.

`ctx.chat_model()` (LangChain), `ctx.crewai_llm()` (CrewAI) and `ctx.ag2_llm_config()` (AG2) return the model under audit; using them is optional but lets `--model` choose the model your MAS runs on.

### Tasks

`prompts.json` maps task names to `{"prompt": ..., "ground_truth": ...}`. Tasks must make your MAS call a tool (that is where attacks land), ground truths should be short, distinctive strings, and at least two tasks must be solved in the benign run (an attack asks another solved task's question and looks for its answer). Failed tasks stay in the benign results for inspection and are not attacked.

### <a id="hooks"></a>Code outside LangChain, CrewAI and AG2

Route tools and system prompts through the context yourself:

| | |
| --- | --- |
| `ctx.wrap_tools(tools, agent=None)` / `ctx.wrap_tool(tool, agent=None)` | Wraps tools (plain or async functions; LangChain and CrewAI tools too), keeping their names, descriptions and argument schemas. Under `tool_filtering`, `wrap_tools` drops the tools the agent does not need. |
| `ctx.system_prompt(agent, prompt)` | Returns `prompt` with the active defense appended, and records it for the blue team. |
| `ctx.record_message(agent, content)` | Adds an agent turn to the transcript. |

```python
def run(task, ctx):
    search = ctx.wrap_tool(search_docs, agent="researcher")
    notes = my_researcher(task, search, system=ctx.system_prompt("researcher", RESEARCHER_PROMPT))
    ctx.record_message("researcher", notes)
    return my_writer(task, notes, system=ctx.system_prompt("writer", WRITER_PROMPT))
```

## Outputs

Everything is written next to your environment folder (`--folder`), under the model name:

- `<folder>/<model>/<env>/iterations/iter_0/discussions/{benign,attack}/`: one transcript + verdict per run; defended runs under `iterations/iter_0/defenses/defense_<name>/`.
- `<folder>/<model>/<env>/agents.json`: the agents, system prompts and tools seen at runtime.
- `<folder>/_evolution/<evolution-name>/iterations/iter_<k>/`: evolved `attacks.json`, `defenses.json` (the defense prompt to add to your agents) and `summary.json`; the matching runs under `<folder>/<model>/<env>/<evolution-name>/iterations/iter_<k>/`. Score them with `purpleaudit.evaluate ... --iteration <k> --evolution-name <name>` (add `--defense-id <i>` for evolved defense `i`).

Each attack phase runs your MAS `solved tasks × attacks` times (28 attacks in total; `--attacks a,b` runs a subset, listed by `python -c "import purpleaudit.attacks as a; print(*a.ATTACKS)"`).

## Things to know

- Always run `--benign` without `--defense-id`: attack runs read the undefended benign results.
- An attack run in which no tool was called is discarded (under a defense it counts as blocked, since the defense may be the reason). If no attack reaches an environment, the run fails; if no tool call is seen at all, a warning says so.
- Prompt-level defenses fail loudly when no agent system prompt was seen, instead of reporting results for a defense that was not applied.
- Attack runs restart your MAS from the task. Transient errors in your entrypoint are retried twice; a run hitting LangGraph's recursion limit is scored as it stands. When several environments run in one command, a failing one is reported and the others continue (exit status 1).
- Tool outputs are only changed when an attack or defense rewrites them: what a model-driven tool call returns is rewritten as the text the model reads; a LangChain tool your code calls directly keeps its structured output unless it is text; framework objects (LangGraph `Command`s, AG2 `ReplyResult`s and handoffs, CrewAI delegation tools and tool failures) are left alone. CrewAI's tool cache and trace-sharing prompt are disabled during audits.
- Cost tracking is only available for the paper's AG2 MASs; other runs report $0.

## Spec-based MAS

The paper's MASs ([`environments/default/`](../environments/default/)) are described in JSON: `environment.json` lists the agents (role, description, system prompt, tools, `is_manager`) and a topology (`Round-robin` or `Supervisor`), and `tools.py` implements the tools. Its `"framework"` field picks the implementation:

| `"framework"` | Implementation |
| --- | --- |
| `"ag2"` (default) | The paper's AG2 group chat (`RoundRobinPattern` / `AutoPattern`). Attack runs resume the benign conversation right before its first tool call. |
| `"langgraph"` | ReAct agents; round-robin turns, or a supervisor delegating through `ask_<agent>` tools ([`byo/spec/langgraph.py`](../byo/spec/langgraph.py)). |
| `"crewai"` | `Process.sequential`, or `Process.hierarchical` with the manager as `manager_agent` ([`byo/spec/crewai.py`](../byo/spec/crewai.py)). |

Generate a new one from a description with `python -m purpleaudit.framework --generate --prompt "A multi-agent system for a deep research team" --num-prompts 30`.

## Command reference

| Command | Purpose | Key flags |
| --- | --- | --- |
| `python -m purpleaudit.framework <env>` | Benign / attack runs on one environment, a comma-separated list, or `all` | `--folder`, `--benign`, `--attack`, `--attacks`, `--all-attacks`, `--evolved-only`, `--defense-id`, `--iteration`, `--evolution-name`, `--model`, `--fixed-model`, `--vectors prompt,tool`, `--generate` |
| `python -m purpleaudit.evolve` | Co-evolution | `--folder`, `--env`, `--iterations`, `--num-attacks`, `--num-defenses`, `--top-k`, `--attacks`, `--evolution-name`, `--evolution-guidance`, `--red-team-only`, `--blue-team-only` |
| `python -m purpleaudit.evaluate {per-attack,per-env,all}` | ASR / utility from the results | `--folder`, `--model`, `--env`, `--iteration`, `--evolution-name`, `--defense-id`, `--output` |

Append `--help` to any command for all options.

## Tests

```bash
pip install -e ".[dev,langgraph,crewai]"
pytest
```

No API key needed: [`tests/stub_llm.py`](../tests/stub_llm.py) is a fake OpenRouter server whose model obeys injected instructions unless a defense is in its system prompt. The suite runs the real CLI end to end (benign, attacks, defenses, evaluation, co-evolution) on the examples and on spec-based MASs for every framework.
