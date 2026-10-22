# Reproducing the paper

Commands run from the repo root, on the 10 generated MASs of [`environments/default/`](../environments/default/). All paper experiments used AWS Bedrock (`aws configure`, region `us-west-2`); OpenRouter works too (see the [guide](guide.md#llm-providers)).

```bash
pip install -e ".[benchmarks]" && playwright install chromium    # Magentic-One only
export MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

**Models (Table 8)**, as Bedrock IDs (prefix with `us.` for cross-region inference): Llama 3 70B `meta.llama3-70b-instruct-v1:0`, Qwen3 235B A22B 2507 `qwen.qwen3-235b-a22b-2507-v1:0`, Claude Sonnet 3.7 `anthropic.claude-3-7-sonnet-20250219-v1:0`, Claude Sonnet 4 `anthropic.claude-sonnet-4-20250514-v1:0`, Claude Sonnet 4.5 `anthropic.claude-sonnet-4-5-20250929-v1:0`, Claude Opus 4.5 `anthropic.claude-opus-4-5-20251101-v1:0`, Claude Opus 4.7 `anthropic.claude-opus-4-7`.

## Baseline attack campaign (§5.2, Table 1)

```bash
python -m purpleaudit.framework all --benign --attack --model $MODEL
python -m purpleaudit.evaluate all --model $MODEL --output baseline_${MODEL}.json
```

Average cost: ~$7.37 × 10 on Sonnet 4.5 (§B.5). Repeat for each model of Table 8.

## Co-evolution (§5.3, Figures 2 & 3)

K=3 iterations, 3 evolved attacks per iteration, Sonnet 4.5 driver:

```bash
python -m purpleaudit.evolve --iterations 3 --num-attacks 3 --num-defenses 1 --model $MODEL \
    --fixed-model us.anthropic.claude-sonnet-4-5-20250929-v1:0 --evolution-name paper_repro
```

Outputs: `environments/_evolution/paper_repro/iterations/iter_{1,2,3}/{attacks.json,defenses.json,summary.json}`.

## Defense evaluation (§5.4, Figures 4 & 5)

The headline defense, `tool_output_extraction`, ships as a hardcoded defense:

```bash
python -m purpleaudit.framework all --attack --all-attacks --evolution-name paper_repro \
    --defense-id tool_output_extraction --model $MODEL
python -m purpleaudit.evaluate all --model $MODEL --defense-id tool_output_extraction --output safeguarded_${MODEL}.json
```

The other defenses of Figure 5: `spotlighting`, `prompt_sandwiching`, `tool_filtering`, `naive_safeguard`.

## Magentic-One on AssistantBench / GAIA (§5.6, Table 2)

Auto-detected for the environment names `assistant_bench` and `gaia`, whose tasks are downloaded from HuggingFace (GAIA is gated: accept its terms on [its page](https://huggingface.co/datasets/gaia-benchmark/GAIA), then `huggingface-cli login`). Magentic-One browses the web and runs the code it writes locally, so prefer a container or VM.

```bash
python -m purpleaudit.framework assistant_bench --benign --attack --model $MODEL
python -m purpleaudit.framework gaia --benign --attack --model $MODEL
python -m purpleaudit.evolve --iterations 3 --num-attacks 3 --env assistant_bench,gaia \
    --evolution-name paper_repro_mone --model $MODEL
python -m purpleaudit.framework assistant_bench,gaia --attack --all-attacks --evolution-name paper_repro_mone \
    --defense-id tool_output_extraction --model $MODEL
```

Average cost: ~$33.39 per audit (§B.5).
