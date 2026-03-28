# Final test set evaluation — run plan

## Setup

- **Input:** `data/splits/test.csv` (2,200 rows, includes 117 NaN-vitals rows)
- **Sharding:** 10 shards of 220 rows each (`--skip-rows 0/220/440/... --n-rows 220`)
- **Checkpoints:** `--checkpoint-every 50` within each shard
- **Rate limiting:** set `--requests-per-minute` per shard so that
  `RPM_PER_SHARD × 10` stays under the provider limit
  (e.g., Gemini paid tier 1000 RPM → `--requests-per-minute 80` per shard)

---

## Per-config procedure

### Step 1 — Mechanical canary (once per new config family)

Run once for each materially distinct config, meaning each unique combination
of prompt template, model, and gating/tool flags. Skip if the exact same config
already passed on the same git commit.

Purpose: catch auth, rate-limit, schema, provider, or output-format failures
before launching the full test run. This is a plumbing check, not a model-
quality check.

```bash
.venv/bin/python experiments/query_strategy_sweep.py \
    --input data/splits/test.csv --n-rows 20 \
    <CONFIG FLAGS> \
    --output-prefix canary_<NAME>
```

Confirm only that the output CSV exists, predictions parse to valid ESI integers,
and there are no crashes or provider/mechanical failures.

Do not inspect canary predictions for quality and do not change prompts,
retrieval settings, or decision logic based on the canary outputs. If the
canary fails, fix execution issues only. Delete the canary output directory
afterward.

### Step 2 — Launch 10 shards, wait for all, check exit codes

```bash
pids=()
for i in $(seq 0 9); do
  .venv/bin/python experiments/query_strategy_sweep.py \
      --input data/splits/test.csv \
      --skip-rows $((i * 220)) --n-rows 220 \
      --requests-per-minute <RPM_PER_SHARD> \
      --checkpoint-every 50 \
      --output-prefix test_<NAME>_shard${i} \
      <CONFIG FLAGS> &
  pids+=($!)
done

# Wait for all shards and check exit codes
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || ((failed++))
done
echo "${failed} shard(s) failed"
```

If any shards failed, investigate logs before proceeding.

### Step 3 — Merge

The sweep writes final CSVs as
`data/runs/<output-prefix>/<output-prefix>_<label>_<timestamp>.csv`.
Match the actual output pattern:

```bash
# For LLM-only / tool-use configs (label = "llm"):
.venv/bin/python scripts/merge_run_shards.py \
    --inputs data/runs/test_<NAME>_shard*/test_<NAME>_shard*_llm_*.csv \
    --output data/runs/test_<NAME>/test_<NAME>_merged_2200.csv \
    --expect-rows 2200 --strict

# For RAG configs (label = strategy name, e.g. "concat"):
.venv/bin/python scripts/merge_run_shards.py \
    --inputs data/runs/test_<NAME>_shard*/test_<NAME>_shard*_concat_*.csv \
    --output data/runs/test_<NAME>/test_<NAME>_merged_2200.csv \
    --expect-rows 2200 --strict
```

Merge validates: schema match, no duplicate `stay_id`s, shard metadata
consistency (model, input hash, prompt template), row count == 2,200.

### Step 4 — Pre-flight sanity check

```bash
.venv/bin/python experiments/eval_triage.py \
    --retrocompute data/runs/test_<NAME>/test_<NAME>_merged_2200.csv \
    --pred triage_LLM  # or triage_RAG for RAG configs
```

Hard gates (must **all** pass before recording):

| Check | Threshold |
|-------|-----------|
| `n_evaluated` + `n_skipped` | == 2,200 |
| `parse_fail_rate` | < 0.01 |
| `n_unique_predicted` | >= 3 |

If any fail → investigate the merged CSV, do not re-run blindly.

### Step 5 — Record results

```bash
.venv/bin/python experiments/eval_triage.py \
    --input data/runs/test_<NAME>/test_<NAME>_merged_2200.csv \
    --pred triage_LLM \
    --model <MODEL> --split test \
    --notes "final test eval: <CONFIG DESCRIPTION>"
```

Appends to `experiments/results/experiment_log.csv`.

---

## Config flags reference

| Mode | Example flags |
|------|--------------|
| LLM-only fewshot | `--mode llm --model gemini-3-flash --prompt-template fewshot` |
| Tool-use (case bank) | `--mode llm --model gemini-3-flash --prompt-template tool_use` |
| Tool-use (PMC only) | `--mode llm --model gemini-3-flash --prompt-template tool_use_pmc` |
| Tool-use (PMC, custom budget) | `--mode llm --model gemini-3-flash --prompt-template tool_use_pmc --context-chars 1000` |
| Two-role | `--mode llm --model gemini-3-flash --prompt-template two_role_case_bank` |
| RAG fewshot | `--mode rag --top-k 5 --strategies concat --model gemini-2.5-flash --prompt-template fewshot` |
| RAG gated | `--mode rag --top-k 5 --strategies concat --model gemini-2.5-flash --prompt-template fewshot --distance-gate 0.25` |

`--context-chars` controls the per-article excerpt budget (default: `RETRIEVAL_CONTEXT_CHARS = 8000`).
After PR #64, all excerpt paths use `prepare_article_excerpt()` which strips front matter before truncation.

Add `--boundary-review`, `--vitals-guardrail`, `--thinking-level LOW`, etc. as
needed to match the specific config being promoted from the dev_tune leaderboard.

---

## Experiments

### Completed (historical, pre-fix)

These runs were completed before the retrieval-text fix (PR #64). They used
the pre-fix code path where active PMC tool calls returned raw uncleaned
1000-char excerpts. Non-PMC runs (W1_ref, G3_W1ref, G3_tool_use) are
unaffected by the fix.

| Internal ID | Reader-facing name | Exact mechanism | Role in final story | Status |
|---|---|---|---|---|
| W1_ref | Safety comparator | Fewshot on G2.5 Flash | Model-generation ablation | **done** |
| G3_W1ref | Primary baseline | Fewshot on G3 Flash | Does the tool add value? | **done** |
| G3_tool_use | Main proposed system | Single-agent, case-bank tool | Our best framework config | **done** |

### Pending (post-fix)

These runs use code after PR #64. All PMC-sensitive paths now use
`prepare_article_excerpt()` which strips front matter via section markers
before truncation. The default excerpt budget is `RETRIEVAL_CONTEXT_CHARS = 8000`.

| Internal ID | Reader-facing name | Exact mechanism | Role in final story |
|---|---|---|---|
| G3_tool_use_pmc_1k_clean | Evidence-source ablation (1k) | Active PMC tool, cleaned 1k excerpts | Matched-budget comparison to pre-fix |
| G3_tool_use_pmc_8k_clean | Evidence-source ablation (8k) | Active PMC tool, cleaned 8k excerpts (default budget) | Is more PMC context better? |
| G3_W1gated_8k_clean | Retrieval-delivery ablation | Gated passive PMC RAG, cleaned 8k snippets | Does distance-gating help vs no retrieval? |
| G4_two_role_pro_critic | Orchestration ablation | Nurse + critic (case-bank only, not PMC-sensitive) | Does multi-agent help? |

### Per-experiment config flags

#### Completed runs

**1. W1_ref** — Safety comparator (G2.5 Flash)

```
--mode llm --prompt-template fewshot --model gemini-2.5-flash
```

Generation model: `gemini-2.5-flash`
Merge label: `llm` | Pred column: `triage_LLM`

**2. G3_W1ref** — Primary baseline (fewshot, no tools)

```
--mode llm --prompt-template fewshot --model gemini-3-flash-preview
```

Generation model: `gemini-3-flash-preview`
Merge label: `llm` | Pred column: `triage_LLM`

**3. G3_tool_use** — Main proposed system

```
--mode llm --prompt-template tool_use --model gemini-3-flash-preview
```

Generation model: `gemini-3-flash-preview`
Merge label: `llm` | Pred column: `triage_LLM`

#### Pending runs

**4. G3_tool_use_pmc_1k_clean** — Evidence-source ablation (cleaned 1k excerpts)

```
--mode llm --prompt-template tool_use_pmc --model gemini-3-flash-preview --context-chars 1000
```

Generation model: `gemini-3-flash-preview`
PMC tool only (no case bank). Cleaned body text, capped at 1,000 chars per article.
Merge label: `llm` | Pred column: `triage_LLM`

**5. G3_tool_use_pmc_8k_clean** — Evidence-source ablation (cleaned 8k excerpts, default budget)

```
--mode llm --prompt-template tool_use_pmc --model gemini-3-flash-preview
```

Generation model: `gemini-3-flash-preview`
PMC tool only (no case bank). Cleaned body text, default `RETRIEVAL_CONTEXT_CHARS = 8000`.
Merge label: `llm` | Pred column: `triage_LLM`

**6. G3_W1gated_8k_clean** — Retrieval-delivery ablation (gated RAG, cleaned 8k)

```
--mode rag --prompt-template fewshot --model gemini-3-flash-preview --distance-gate 0.25
```

Generation model: `gemini-3-flash-preview` | Embedding model: `text-embedding-005` (fixed in config)
Passive RAG with cleaned 8k excerpts (default budget). Distance gate at 0.25.
Merge label: `concat` | Pred column: `triage_RAG`

**7. G4_two_role_pro_critic** — Orchestration ablation (nurse + critic)

```
--mode llm --prompt-template two_role_case_bank --model gemini-3.1-pro-preview --fast-model gemini-3-flash-preview
```

Nurse (fast) model: `gemini-3-flash-preview` | Critic (main) model: `gemini-3.1-pro-preview`
Case-bank only — not PMC-sensitive, unaffected by the retrieval-text fix.
Merge label: `llm` | Pred column: `triage_LLM`

> **Warning:** Both `--model` and `--fast-model` MUST be specified. If
> `--fast-model` is omitted it defaults to `--model`, which would run Pro for
> both roles — a different (and much more expensive) experiment.

> **Note:** Pro has significantly lower rate limits than Flash. Use 5 shards of
> 440 rows (`--skip-rows 0/440/880/... --n-rows 440`) with
> `--requests-per-minute 3` per shard to stay under Pro's aggregate RPM ceiling.

---

## Repeat for each config

Run Steps 1–5 for each config promoted to final eval. Configs are independent —
no shared state between them.
