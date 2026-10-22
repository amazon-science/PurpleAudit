# PurpleAudit

Code for **PurpleAudit: A Co-Evolutionary Auditing Framework for Task Hijacking in Multi-Agent Systems**.

PurpleAudit audits a multi-agent system (MAS) for task hijacking by indirect prompt injection. It runs your MAS on your tasks, plants an attacker's request in the output of the first tool your agents call, and measures the **attack success rate (ASR)** and **utility** (tasks still solved), with and without defenses. A red-team / blue-team co-evolution then finds stronger attacks against your MAS and a defense prompt that counters them. MASs built with **LangGraph, CrewAI or AG2** are audited without changing their code.

## Install

```bash
cd PurpleAudit
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[langgraph,crewai]"
cp .env.example .env            # then set OPENROUTER_API_KEY (or use AWS Bedrock, see docs/guide.md)
```

## Quick start: audit an example

[`examples/default/`](examples/default/) holds three small MASs: `langgraph_support` (LangGraph), `crewai_travel` (CrewAI) and `ag2_helpdesk` (AG2).

```bash
MAS=langgraph_support           # or crewai_travel, ag2_helpdesk
MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0

# 1. Benign runs, then 2 of the paper's evolved attacks (drop --attacks to run all 28 attacks)
python -m purpleaudit.framework $MAS --folder examples --benign --attack --attacks evolved_mas_specialized_agent_access_credential_injection,evolved_synchronization_lock_injection

# 2. The same attacks under a defense
#    (spotlighting, prompt_sandwiching, tool_filtering, naive_safeguard, tool_output_extraction)
python -m purpleaudit.framework $MAS --folder examples --attack --attacks evolved_mas_specialized_agent_access_credential_injection,evolved_synchronization_lock_injection --defense-id tool_output_extraction

# 3. ASR and utility, without and with the defense
python -m purpleaudit.evaluate per-env --folder examples --model $MODEL
python -m purpleaudit.evaluate per-env --folder examples --model $MODEL --defense-id tool_output_extraction

# 4. Co-evolution: new attacks against the MAS and a defense prompt for it, then their scores
python -m purpleaudit.evolve --folder examples --env $MAS --iterations 2 --num-attacks 2 --evolution-name demo --attacks evolved_mas_specialized_agent_access_credential_injection,evolved_synchronization_lock_injection
python -m purpleaudit.evaluate per-env --folder examples --model $MODEL --iteration 1 --evolution-name demo
python -m purpleaudit.evaluate per-env --folder examples --model $MODEL --iteration 1 --evolution-name demo --defense-id 0
```

Transcripts and verdicts land in `examples/<model>/<example>/`, evolved attacks and defense prompts in `examples/_evolution/demo/`. Each example also runs on its own, e.g. `python examples/default/langgraph_support/support_app.py "What is the status of order ORD-1001?"`.

## Audit your own MAS

**1. Create three files** (see the [examples](examples/default/)):

```
audits/default/my_mas/
├── environment.json    {"entrypoint": "audit:run"}
├── prompts.json        {"task_1": {"prompt": "...", "ground_truth": "..."}, ...}
└── audit.py
```

**2. `audit.py` runs your MAS, unchanged, on one task:**

```python
from my_app import answer        # your code, next to audit.py or installed

def run(task, ctx):
    return answer(task, ctx.chat_model())
```

While `run` executes, PurpleAudit instruments LangGraph / LangChain, CrewAI and AG2: the attack lands in the first tool output, defenses are added to tool outputs and system prompts, and the conversation is recorded. Passing the model under audit is optional (it lets `--model` pick your MAS's model): `ctx.chat_model()` for LangChain / LangGraph, `ctx.crewai_llm()` for CrewAI, `ctx.ag2_llm_config()` for AG2. For code outside these frameworks, see [hooks](docs/guide.md#hooks).

**3. Write tasks** in `prompts.json`: at least two tasks that make your MAS call a tool, each with a short, distinctive `ground_truth` that a correct answer contains.

**4. Run the quick-start commands** with `MAS=my_mas` and `--folder audits`.

## Magentic-One on AssistantBench / GAIA

The paper's real-world MAS: Microsoft's Magentic-One, a web-browsing and coding team, on the AssistantBench and GAIA benchmarks. The tasks come from HuggingFace; GAIA is gated, so accept its terms on [its page](https://huggingface.co/datasets/gaia-benchmark/GAIA) and log in with `huggingface-cli login` first. AssistantBench has 33 tasks (about $33 per audit in the paper). Magentic-One browses the web and runs the code it writes on your machine, so prefer a container or VM.

```bash
pip install -e ".[benchmarks]" && playwright install chromium
python -m purpleaudit.framework assistant_bench --benign --attack --attacks evolved_mas_specialized_agent_access_credential_injection,evolved_synchronization_lock_injection
python -m purpleaudit.evaluate per-env --model $MODEL --env assistant_bench
```

## More

- [docs/guide.md](docs/guide.md): how audits work, options, outputs, spec-based MASs, tests.
- [docs/paper.md](docs/paper.md): reproducing the paper's experiments.

## Release notice

This code is being released solely for academic and scientific reproducibility purposes, in support of the methods and findings described in the associated publication. Pull requests are not being accepted in order to maintain the code exactly as it was used in the paper.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the CC-BY-NC-4.0 License. See [LICENSE](LICENSE).
