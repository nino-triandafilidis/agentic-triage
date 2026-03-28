# TODO

## V2 PMC-RAG Experiment Flow

Goal: test additive PMC value on top of few-shot baseline.

**Setup:** dev_tune.csv, pilots n=150, decisive n=1,000. Log path + count + SHA256.

### Wave 1 — Baseline + additive PMC

- [x] **W1-ref:** Few-shot only, n=1,000 *(done: κ=0.382, `data/runs/W1_ref/W1_ref_combined_1000.csv`)*
- [x] **W1-rag:** Few-shot + PMC (top_k=5, no gate), n=1,000 *(done: κ=0.219, gate FAIL — Δκ=-0.163, `data/runs/W1_rag/W1_rag_combined_1000.csv`)*
- [x] **W1-gated:** Few-shot + PMC (gate=0.25), n=1,000 *(done: κ=0.288, gate FAIL — Δκ=-0.094, `data/runs/W1_gated/W1_gated_combined_1000.csv`)*
- [x] Evaluate Wave 1 gate *(FAIL: W1-rag κ=0.219, W1-gated κ=0.288, both below ref κ=0.382. → Wave 2A skipped, proceed to Wave 2B)*

**Promotion ladder** (applied to each candidate vs paired non-PMC scaffold):

1. **Reliable output?** parse_fail_rate acceptable → else prune/fix
2. **Safety gate?** under_triage_rate and expected_cost non-inferior → else prune
3. **Primary benchmark gain?** exact_accuracy and/or range_accuracy up → else check 4
4. **Primary flat but safety/cost clearly better?** → else prune
5. **Collapse guardrails?** macro_recall, per-class recall, MA-MAE, over_triage_rate okay → else diagnose
6. **Diagnostics supportive?** AC2 direction consistent, QWK not showing collapse alarm → else manual review
7. **Promote:** pilot (150) → decisive (1,000); decisive → next wave / holdout candidate

**Wave 1 verdict: FAIL (pruned at step 3).** Both W1-rag and W1-gated pass reliability and safety but show primary benchmark regression (exact_acc and range_acc both down). Neither offers safety/cost improvement to compensate. → Wave 2A skipped, proceed to Wave 2B.

### Wave 2A — PMC optimization (SKIPPED — gate failed)

- [x] ~~Threshold sweep: gates 0.20–0.30, n=1,000~~ *(skipped)*
- [x] ~~Reranker pilot: top 20 → top 5, n=150~~ *(skipped)*
- [x] ~~If reranker promising → promote to n=1,000; else stop reranker~~ *(skipped)*

### Wave 2B — Orchestration (if gate fails)

- [ ] Pilot 1: Single-agent, PMC-as-tool, n=150
- [ ] Pilot 2: Single-agent, PMC + case bank, n=150
- [x] Pilot 3: Multi-role (Nurse → Reader → Adjudicator), n=150 *(completed: κ=0.227, exact_acc=52.0%, range_acc=93.3%, under-triage=6.7%, reported cost=$1.87 generation-only; see experiment_results.md)*
- [ ] If best orchestration branch found → Wave 2C; else stop orchestration
- [x] Add dedicated evidence reranker for orchestration evidence selection *(implemented: `three_role_rerank_critic` profile with explicit reranker stage)*
- [x] Refactor Pilot 3 nurse tool-use token/cost tracking to avoid before/after counter deltas *(implemented: shared `_run_tool_structured_stage`)*
- [ ] Remove or document unused `multi_role` prompt-template registry keys (`rag`/`llm`/`handbook_*`)
- [ ] Add uncertainty-routed PMC availability for the new orchestration profiles instead of relying only on unconditional access or nurse confidence
- [ ] Add a separate case-bank-first single-agent dual-tool profile; keep current `tool_use_dual` as the unrestricted dual-tool ablation
- [ ] Document legacy `tool_use_rag` as a conflated branch (passive PMC preload + optional extra PMC tool calls), not a clean PMC-on-demand baseline
- [ ] Optional expensive follow-up: rerun best PMC-capable orchestration branch with helper roles on `gemini-3.1-pro` *(requires explicit cost approval)*

#### G3 orchestration search screen (500 rows, same-row paired vs G3 tool_use anchor)

- [x] `G3_tool_use_pmc_500` *(completed: exact=60.0%, range=81.4%, under-triage=18.4%, AC2=0.763, κ=0.422; net lift -2/500 vs anchor, not promoted)*
- [x] `G3_tool_use_dual_500` *(completed: exact=60.0%, range=77.2%, under-triage=22.6%, AC2=0.760, κ=0.449; net lift -3/500 vs anchor, not promoted; mean `pmc_call_count=0.448`, mostly case-bank-dominant behavior)*
- [x] `G3_two_role_case_bank_500` *(completed: exact=50.4%, range=89.8%, under-triage=9.6%, over-triage=40.0%, AC2=0.640, κ=0.278; net lift -50/500 vs anchor, not promoted; ESI-3 recall collapsed to 0.138)*
- [ ] `G3_two_role_case_bank_pmc_conditional_500` *(held out in `w2b-conditional-pmc` worktree; needs smoke after PMC-first policy change in commit 3905006; first-pass results suggest diminishing returns)*
- [ ] `G3_three_role_rerank_critic_500` *(not yet run; first-pass orchestration results all negative — optional)*

#### G4 guardrail experiments (500 rows, same-row paired vs G3 tool_use anchor)

- [x] `G4_decision_tree_500` *(completed: exact=61.2%, range=77.6%, under-triage=22.2%, AC2=0.766, κ=0.438; net lift +3/500, near-neutral — ESI-3 recall preserved at 0.630)*
- [x] `G4_dt_boundary_review_500` *(completed: exact=59.2%, range=77.6%, under-triage=22.2%, AC2=0.753, κ=0.411; net lift -7/500, not promoted — boundary review amplified 3→2 over-triage)*
- [x] `G4_vitals_guardrail_500` *(completed: exact=58.4%, range=79.6%, under-triage=20.2%, AC2=0.752, κ=0.398; net lift -12/500, not promoted — ESI-3 recall collapsed 0.626→0.514)*

### Wave 2C — Multi-model follow-up (if W2B succeeds)

- [ ] Best orchestration variant, n=150

### Wave 3 — Refinement

- [ ] Re-query or uncertainty gating (from W2A/W2C winners)
- [ ] Final evaluation: compare to paired non-PMC scaffold
- [ ] Row-level PMC analysis: changed / helped / hurt / neutral by ESI and distance

### Pre-wave infrastructure fixes (from 2026-03-11 postmortem)

Must be fixed before running experiments on the corresponding wave:

**Before any run:**
- [x] Fix timeout recovery crash: `pipeline._provider._client` → registry-based provider reset *(done: integrate-wave1)*
- [x] Add `error`/`generation_status` columns to CSV output *(done: integrate-wave1)*
- [x] Add fail-fast on first-row null prediction *(done: sweep-infra-fixes)*

**Before Anthropic runs:**
- [x] Auth preflight: call `provider.is_available()` before row loop *(done: sweep-infra-fixes)*
- [x] Add `--requests-per-minute` throttle flag *(done: sweep-infra-fixes)*

**Before Wave 2B:**
- [x] Fix PMC truncation fairness *(done: replace hardcoded 1000-char rerank excerpts with dedicated `rerank_chars`, default 1500)*
- [x] Add tool-use controls (`--max-pmc-calls-per-row`)
- [x] Fix prompt/tool mode alignment *(done: hide PMC tool when retrieval is disabled / `top_k=0`, including `mode=llm`)*

**Before next eval run:**
- [ ] Harden `eval_triage.py` `--split` default: infer from sidecar `input_file` field or fail when `--split scratch` is used on a non-scratch input *(bug: W1-gated was logged as `split=scratch` instead of `dev_tune`)*

**Before any run >200 rows:**
- [x] Incremental checkpointing (`_partial.csv` every N rows, `--checkpoint-every`) *(done: sweep-infra-fixes)*

### Worktree cleanup

Wave 1 was audited, P0 bugs fixed, and merged into main via `integrate/wave1-into-main`
(11 commits, +1326/-64, fast-forwarded 2026-03-11).

- [x] Diff wave1 vs main, audit each changed file for correctness
- [x] Verify `.env` loading fix works in both main and worktree contexts
- [x] Verify Haiku cost tracking produces non-zero costs (unit test) *(root cause: missing fallback entry, already fixed in PR #43; regression test added)*
- [x] Run 2-row smoke test on Gemini to confirm no regressions *(LLM-only + RAG pass; artifacts cleaned up — session-only verification, not persisted)*
- [x] Merge to main (fast-forward, 11 commits)
- [x] Remove `agent-abfd9ede` worktree + branch (subsumed by wave1)
- [x] Remove `agent-af466124` worktree + branch (subsumed by wave1)
- [x] Remove `agent-a5628526` worktree + branch (subsumed by wave2b)
- [x] Remove `integrate-wave1` worktree + branch (merge complete)
- [x] Remove `wave1` worktree + branch (merge complete)
- [x] Keep `wave2b` as reference only — do NOT merge (has known bugs)

## Experiment: BM25 with 8k indexed chars

E15 built the BM25 index on 2k cleaned characters per article (for ranking efficiency), while
FAISS embeddings encode ~8k characters. This mismatch means BM25 ranks documents using a
narrower text window than FAISS, which may explain part of the κ gap (0.274 vs 0.313).
Rebuild the BM25 index with `--max-chars 8000` and re-run the E15 setup to isolate whether
the indexing window size accounts for the sparse-vs-dense performance difference.

- Modify `build_bm25_index.py` to accept `--max-chars` (currently hardcoded at 2000).
- Rebuild index with 8k chars (expect ~2× build time and index size).
- Run E17 with `RETRIEVAL_BACKEND=bm25` on dev_tune (same config as E15).
- Compare κ against E15 (2k BM25), E07 (FAISS 8k), and E16 (Hybrid).

## Integrate eval logging into query_strategy_sweep.py

`query_strategy_sweep.py` computes metrics via `evaluate()` but does not call `log_result()` — the experiment log (`experiments/results/experiment_log.csv`) is only populated by running `eval_triage.py` as a separate manual step. Add automatic `log_result()` after each strategy's `evaluate()` call so every sweep run is logged without a separate eval invocation.

## Make cloud storage paths portable

`src/config.py` hardcodes a machine-specific Google Drive path as the default for
`FAISS_STORE_DIR` (`/Users/ninotriandafilidis/Library/CloudStorage/...`). A collaborator
cloning the repo gets a path that doesn't exist on their machine.

- Replace the hardcoded default with a portable fallback (e.g. `~/medllm/faiss_store`),
  matching the pattern already used by `FAISS_LOCAL_DIR` and `BM25_LOCAL_DIR`.
- Add `FAISS_STORE_DIR`, `FAISS_LOCAL_DIR`, and `BM25_LOCAL_DIR` to `.env.example` so
  collaborators know to set them.

## Faster model testing while preserving comparability

Running 1,000-row experiments is slow at ~10s/row (~2.8 hrs per run). To speed up
iteration without breaking comparability:

1. **Parallel CLI invocations**: The sweep script is single-process. Two instances with
   non-overlapping `--skip-rows`/`--n-rows` ranges can run concurrently against the same
   API without conflicts (separate cost budgets, separate output files). Concatenate CSVs
   afterward. Rate-limit risk: Gemini Flash paid tier allows ~1000 RPM — two processes at
   ~6 RPM each is safe. Three+ processes need monitoring.

2. **Batch API (not yet available for Gemini)**: When Gemini adds batch prediction,
   switch to async batch submission. This would allow 1,000 rows in a single request with
   results returned asynchronously. No code changes needed beyond a batch-mode flag in
   `agentic_pipeline.py`.

3. **Faster model substitution for pilots**: Use a cheaper/faster model (e.g., Gemini
   Flash Lite or Haiku) for mechanism pilots (n=150) where the goal is "does this idea
   work at all?" NOT for decisive 1,000-row runs. Requires:
   - `--model` flag already exists in sweep script
   - Add `pilot_model` field to sidecar manifest so it's clear which model was used
   - Promotion from pilot to decisive MUST re-run on the standard model (Gemini 2.5 Flash)
   - Never compare metrics across different models — only compare within same model

4. **Shard and merge pattern**: Split dev_tune.csv into N shards, run N parallel
   processes, merge results. Script needed: `scripts/merge_run_shards.py` that
   concatenates CSVs, validates no row overlap, and produces a unified sidecar manifest.

## Deferred / Post-MVP

- Defer bootstrap CIs for experiment comparisons (post-MVP).
- Refactor `experiment_results.md` into a short landing page plus separate protocol / archive / synthesis docs once the current experiment cycle stabilizes, so findings stay readable without increasing update overhead right now.
- Implement targeted ESI excerpt mode (loader + prompt integration + manifest fields).
- Validate estimated token/cost metrics against historical billing export.
- Create dev_tune/dev_holdout fixed row-id manifests and enforce in experiment scripts.
- Consider rebuilding FAISS index on filtered corpus (retracted=no, commercial CC license,
pub_year >= 1990) — ~11.5% of current index would be excluded. See issue #4 and
paper_notes.md § "Corpus quality: unfiltered index as a limitation". Estimated rebuild
cost ~$0.90 (same as original export). Run scripts/count_filtered_pmc.py first to confirm
row counts before committing.

## NON-MVP
### Expand pricing extraction for all Gemini 2.5 Flash SKUs

Update `src/rag/triage_core.py` `fetch_vertex_pricing()` to properly track costs for all Gemini 2.5 Flash SKUs (Context Caching, Thinking mode) and Text Embeddings. The current implementation uses rigid string matching that explicitly excludes "caching" and "thinking" SKUs, and only returns a 2-key dict (`input`, `output`).

**Changes:**

1. **Expand `FALLBACK_PRICING` dict** to new dimensions:
  - `input_standard` (0.075/1M), `input_cached` (0.01/1M), `output_standard` (0.30/1M), `output_thinking` (0.30/1M)
  - Add `text-embedding-004`: `input_standard` (0.025/1M)
2. **Update `fetch_vertex_pricing` return structure** from `{"input": ..., "output": ...}` to `{"input_standard": ..., "input_cached": ..., "output_standard": ..., "output_thinking": ...}`.
3. **Update SKU matching logic** — remove explicit exclusions for "caching"/"thinking". Map SKUs by description keywords:
  - `input_cached`: "caching" or "cache" in description
  - `output_thinking`: "thinking" in description
  - `input_standard`: "input" but not "caching"/"cache"
  - `output_standard`: "output" but not "thinking"
  - Embeddings: "embeddings" in description when model_id implies embeddings
4. **Update all downstream callers** that read `pricing["input"]` / `pricing["output"]` (e.g. `agentic_pipeline.py:_compute_cost`, `run_rag_triage.py`).

**Verification:** Call `fetch_vertex_pricing('gemini-2.5-flash')` and confirm 4 keys; call with `'text-embedding-004'` and confirm embedding pricing. Run a sample RAG pipeline call and verify cost calculation doesn't crash.

### Remaining GCP config items

- Python-side embeddings: Call the embedding API from Python (`google-genai` SDK) instead of `CREATE MODEL ... REMOTE` in BigQuery. Removes the need for `bigqueryconnection.googleapis.com` and Connection Admin role.
- Auto-enable APIs in a setup script: The original notebook auto-enables APIs via `service_usage_v1`. Add a one-time setup script that does the same.
