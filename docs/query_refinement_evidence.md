# Query Refinement Evidence

This note is a citation-ready summary of the repo experiments that support the
claim:

> Early experiments suggested that iterative query refinement was not the most
> useful lever: changing how retrieval queries were rewritten or decomposed did
> not consistently improve prediction quality on our development split.

## Scope

- All experiments below use the `dev_tune` development split.
- The direct query-formulation experiments were run on the same 150-row pilot
  slice used throughout the early retrieval study.
- The table focuses only on experiments that changed *how the query was built*
  or *whether the system re-queried*, rather than changing the model family or
  the downstream prompting scaffold.
- Early pilots primarily reported exact accuracy, range accuracy, MAE, and
  quadratic kappa. Later work added the full multilevel metric stack.

## Curated evidence table

| Experiment | Split / n | Query strategy under test | Comparator | Exact acc | QWK | What changed | Result |
|---|---:|---|---|---:|---:|---|---|
| `E02a concat` | `dev_tune`, 150 | Single query built from concatenated HPI + patient info + vitals | `E00.5` LLM-only | 56.0% | 0.232 | Baseline RAG query formulation | Already below LLM-only (`κ=0.338`) |
| `E02a hpi_only` | `dev_tune`, 150 | Single query built from HPI only | `E02a concat` | 50.7% | 0.184 | Removes non-HPI fields | Worse than concat (`Δκ=-0.048`) |
| `E02b rewrite` | `dev_tune`, 150 | LLM rewrites concat into PubMed-style keyword query | `E02a concat` | 54.7% | 0.236 | Adds iterative query rewriting | Effectively flat on kappa (`Δκ=+0.004`) and lower exact accuracy (`-1.3pp`) |
| `E09 multifacet` | `dev_tune`, 150 | Three facet-specific queries pooled into one result set | `E07` single query (8k+strip) | 54.0% | 0.218 | Adds query decomposition / diversity | Worse than simpler single-query retrieval (`Δκ=-0.095`) |
| `E12 adaptive` | `dev_tune`, 150 | Re-query difficult rows using extracted key medical terms, then merge results | `E07` no gate / `E11` gate 0.25 | 58.0% | 0.292 | Adds targeted second-pass retrieval | Worse than ungated RAG (`Δκ=-0.021`) and much worse than simple distance gating (`Δκ=-0.074`) |

## What this shows

Three patterns are consistent across the direct query-formulation experiments:

1. **LLM query rewriting did not materially improve quality.** In `E02`, the
   rewritten query was nearly identical to plain concatenation on the same 150
   rows (`κ=0.236` vs `0.232`) and slightly worse on exact accuracy (`54.7%`
   vs `56.0%`).
2. **Query decomposition hurt rather than helped.** In `E09`, multi-facet
   retrieval expanded the article pool substantially, but prediction quality
   fell relative to the simpler single-query baseline (`κ=0.218` vs `0.313`).
3. **Adaptive re-querying also failed to beat simpler alternatives.** In
   `E12`, a second retrieval pass for hard cases still underperformed both the
   simpler ungated setup (`κ=0.292` vs `0.313`) and the best simple gating
   strategy (`κ=0.366`).

Taken together, these results support the narrower claim that **query
refinement was not the main lever on this corpus**. When performance improved,
it was not because the system learned to write better queries; it was more
consistent with suppressing low-quality retrieval altogether.

## Important qualifier: `E11` is not a query-refinement win

`E11` (distance-gated retrieval) is the only early PMC-RAG configuration that
exceeded the LLM-only pilot on the same 150-row split (`κ=0.366` vs `0.338`).
However, that result should not be read as evidence for iterative query
refinement:

- `E11` does **not** rewrite, decompose, or re-query.
- It keeps the original single query and instead discards retrieved articles
  when top-1 distance exceeds a threshold.
- At the best threshold (`0.25`), `68.0%` of rows fell back to the LLM-only
  path.

So the main positive early signal came from **using retrieval less often**, not
from making the retrieval query more sophisticated.

## Scale-up check

The one promising early retrieval control (`E11` gate = `0.25`) did not hold at
decisive scale. On the 1,000-row `W1` runs:

| Experiment | Split / n | Exact acc | Macro recall | QWK | AC2 | Notes |
|---|---:|---:|---:|---:|---:|---|
| `W1-ref` | `dev_tune`, 1000 | 53.4% | 0.378 | 0.382 | 0.727 | Few-shot baseline |
| `W1-rag` | `dev_tune`, 1000 | 50.1% | 0.316 | 0.219 | 0.636 | Ungated article RAG |
| `W1-gated` | `dev_tune`, 1000 | 51.9% | 0.341 | 0.288 | 0.719 | Distance gate = 0.25 |

`W1-gated` remained better than ungated RAG, but still failed to beat the
non-RAG baseline on exact accuracy, QWK, macro recall, or AC2. This reinforces
the interpretation that retrieval control was at best a partial mitigation, not
the main path to better triage prediction.

## Citation-ready wording

Short version:

> Early retrieval experiments suggested that iterative query refinement was not
> the main lever on this corpus: LLM query rewriting was effectively flat
> relative to simple concatenation (`κ=0.236` vs `0.232`), multi-facet
> decomposition was worse (`κ=0.218` vs `0.313`), and adaptive re-querying also
> underperformed both ungated RAG (`κ=0.292` vs `0.313`) and simple distance
> gating (`κ=0.366`) on the development split.

Safer version:

> On the early `dev_tune` retrieval pilots, changing how the query was
> rewritten, decomposed, or re-issued did not consistently improve prediction
> quality. Query rewriting was nearly flat relative to simple concatenation,
> multi-facet decomposition was worse, and adaptive re-querying underperformed
> simpler baselines.

## Sources

- [`experiment_results_archive.md § E02a / E02b`](../experiment_results_archive.md#e02a--e02b--core-query-ablation-concat-vs-hpi_only-vs-rewrite)
- [`experiment_results_archive.md § E09`](../experiment_results_archive.md#e09--multi-facet-query-decomposition)
- [`experiment_results_archive.md § E11`](../experiment_results_archive.md#e11--distance-gated-retrieval)
- [`experiment_results_archive.md § E12`](../experiment_results_archive.md#e12--adaptive-re-querying)
- [`experiment_results_archive.md § W1-rag & W1-gated`](../experiment_results_archive.md#w1-rag--w1-gated--gemini-fewshotrag-1000-rows)
