# Agent Instructions

## Critical Rules

### Eval data — do not touch
- `third_part_code/medLLMbenchmark/MIMIC-IV-Ext-Triage-Specialty-Diagnosis-Decision-Support.csv`
  is the benchmark eval dataset. Never use it for any purpose other than a formal eval run.
- Prohibited uses: sampling for analytics, informing prompt engineering or hyperparameter
  decisions, passing rows to any LLM or embedding model outside a controlled eval pipeline.
- Any work requiring benchmark-like cases must use `MIMIC-IV-Ext-Dev.csv` (rows `df[2200:]`
  from the creation script). See README § [DRAFT] Dev / Eval Split Strategy.

### Retrieval backend — local only (FAISS, BM25, or hybrid)
- Valid backends: `faiss` (default), `bm25` (sparse), `hybrid` (FAISS + BM25 via RRF).
- NEVER set or use `RETRIEVAL_BACKEND=bq`. BigQuery vector search costs real money on every
  call and is unnecessary now that local indexes are provisioned.
- If code or a script defaults to or accepts a `bq` backend, do not invoke it with that option.

### Cost risk assessment — required before running scripts
- Before running any script that calls a pay-per-use API (BigQuery, Vertex AI, Gemini,
  Anthropic, embedding models), explicitly assess cost risk and flag concerns to the user.
  Do not proceed without user confirmation.
- BigQuery: $5/TB scanned. `article_text` full scan (2.3M rows) ≈ 150 GB ≈ $0.75.
  Use hash-based sampling: `WHERE MOD(ABS(FARM_FINGERPRINT(pmc_id)), 100) < N`.
  LIMIT without ORDER BY does NOT reliably reduce scan cost.
- LLM APIs: watch for 429 RESOURCE_EXHAUSTED. Use exponential backoff (tenacity) around
  all generation calls.

### Statements in documentation must be grounded
- Never write a finding, statistic, or behavioural claim in the README or any doc unless it
  is directly produced by running the relevant script or verified in code.
- If a statement is inferred rather than computed, mark it explicitly:
  "(inferred)", "(not verified by script output)", or "(estimated — re-run script to confirm)".

### Parallel agents — worktrees required
Before creating or switching branches, ALWAYS run:
1. `git worktree list` — see all worktrees and which branch each has checked out
2. `ps` / check terminals — determine if other agents are active in any worktree

If ANY other agent is active in the repo (even on `main`), you MUST use a separate worktree.
No exceptions. Do not assume the other agent is idle — it may commit, stage, or modify files
at any time. `git switch` in a shared worktree moves HEAD for everyone in that directory,
silently corrupting both agents' work.

```bash
# BAD — another agent is active in the same directory:
git switch -c feature/my-task    # WRONG: changes branch for BOTH agents

# GOOD — create an isolated worktree:
git worktree list
# shows: /path/to/medLLM  abc1234 [main]  ← another agent is here

git worktree add "../agent-sparse" -b feature/sparse-retrieval-bm25 main
cd "../agent-sparse"
# ... do all work here ...

# When done — merge from the original worktree:
cd "/path/to/medLLM"
git merge --ff-only feature/sparse-retrieval-bm25
git worktree remove "../agent-sparse"
git branch -d feature/sparse-retrieval-bm25
```

---

## Project Mission

The goal of this project is to develop and evaluate an **agentic RAG framework** for clinical
decision support. The framework's quality is measured by how well it performs on a concrete
task — ESI triage prediction using PMC literature as a retrieval corpus — but the primary
deliverable is the *framework itself*: its orchestration patterns, tool-use strategies,
retrieval gating, and evaluation methodology. Finding the best possible triage pipeline is a
means to that end, not the end itself.

The CS224N paper evaluates the framework against PMC-backed RAG on MIMIC-IV-ED triage cases,
so PMC retrieval experiments remain in scope as the paper's empirical contribution. However,
architectural decisions (prompt strategy, tool design, gating logic) should be evaluated for
their generalizability as framework components, not only for their ESI accuracy on this
specific corpus.

### Evaluation methodology

Framework configurations are evaluated using a **multilevel metric stack**, not a single
headline score. The tiers, in priority order:

1. **Benchmark** — `exact_accuracy`, `range_accuracy` (for comparability with Gaber et al.)
2. **Safety** — `under_triage_rate`, `over_triage_rate` (asymmetric clinical harm)
3. **Imbalance diagnostics** — `macro_recall`, `per_class_recall`, `MA-MAE` (collapse detection)
4. **Chance-adjusted diagnostics** — `linear_weighted_gwet_ac2`, `quadratic_weighted_kappa`
   (AC2 preferred; QWK retained as collapse alarm only)

See `docs/paper_notes.md § Evaluation metrics` for rationale and sources.

### Current state

- The framework (`src/rag/`, `src/llm/`) retrieves articles from a local FAISS index + SQLite
  store (embeddings via text-embedding-005) and generates predictions via a provider-agnostic
  LLM interface supporting Anthropic, Google, and open-source models.
- Experiments to date have tested retrieval strategies (dense, sparse, hybrid, gated),
  orchestration patterns (passive injection, tool-use, multi-role agents), prompt strategies
  (few-shot, CoT, handbook injection), and model generalizability (Gemini 2.5/3 Flash, Pro,
  Claude Haiku/Sonnet).

### Key Paths
- `src/config.py`               — GCP credentials, BQ client, constants
- `src/llm/`                    — Provider-agnostic LLM interface (Anthropic, Google, Kimi)
- `src/rag/retrieval.py`        — Article retrieval: FAISS (dense), BM25 (sparse), hybrid (RRF), BQ (legacy)
- `src/rag/generation.py`       — LLM response generation
- `src/rag/triage_core.py`      — Core triage prediction logic and prompt building
- `src/rag/agentic_pipeline.py` — Multi-step agentic RAG pipeline
- `src/rag/pipeline.py`         — End-to-end RAG orchestration
- `experiments/`                — Experiment scripts (eval, sweeps, tracking)
- `third_part_code/medLLMbenchmark/` — Upstream benchmark (triage, diagnosis, specialty)
- `third_part_code/pubmed-rag/`      — Upstream Google PubMed RAG example
- `analytics/`                  — PubMed content analytics scripts

---

## Workflow

### Branch conventions
- Always branch from `main`. Prefer short-lived branches; one logical change per branch.
- Naming: lowercase, hyphen-separated — `feature/<area>-<desc>`, `fix/<scope>`, `chore/<scope>`
- Commit message format: `type(scope): concise present-tense summary`
- Avoid direct commits to `main`. Use a branch, then fast-forward or squash-merge.

### Branch lifecycle
1. Start from updated main:
   - FIRST: run the worktree pre-flight check (see Critical Rules above).
   - If another agent is active → create a worktree:
     ```bash
     git worktree add "../<worktree-name>" -b feature/<area>-<short-desc> main
     ```
     Then cd into that worktree and work there exclusively.
   - If you are the ONLY agent active → you may branch in the main worktree:
     ```bash
     git switch main
     git pull --rebase   # if remote configured
     git switch -c feature/<area>-<short-desc>
     ```
2. Commit in small, meaningful steps.
3. Keep your branch current — periodically rebase onto `main`.
4. Merge back cleanly — prefer `git merge --ff-only`. Delete branch after merge.

### Subtree-specific guidance
- Don't mix vendor updates with feature work. Use `vendor/update-<name>-<yyyymmdd>` branches.
- `git subtree pull --prefix third_part_code/pubmed-rag https://github.com/google/pubmed-rag.git main --squash`

### Pre-commit checklist
- Review `TODO.md` for outstanding items related to the changes being committed.
- If a TODO is resolved by the current work, mark it done or remove it.
- If the work introduces a new concern, add it to the appropriate TODO section.
- Do not commit if there is an open TODO that directly conflicts with the changes.

### Writing experiment results
- **Decision doc** (`experiment_results.md`): Update the leaderboard tables, key findings,
  and open experiments sections after each evaluated run. Keep this file concise (~250 lines).
- **Archive** (`experiment_results_archive.md`): Write the full per-experiment writeup here
  (Hypothesis, Script command, Design, Results table, Interpretation, Output files).
  Follow the style of existing sections (e.g. E02a, E05, W2B).
- **CSV log** (`experiments/results/experiment_log.csv`): Machine-readable source of truth
  for all runs. Always append a row here.
- After any experiment finishes running, always summarize the result for the user immediately.
  Start that summary with the exact question the experiment was meant to answer, then give the
  grounded result, caveats, and next-step recommendation.

### GitHub / collaboration
- Never assign GitHub issues or PRs to nevingeorge.
- Design for reproducibility: a new collaborator should be able to clone and run with minimal
  manual steps. Avoid machine-specific workarounds; fix the root cause.
