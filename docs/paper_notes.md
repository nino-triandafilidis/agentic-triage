# Paper Notes

> **Guide:** see `docs/paper_agent.md` for report structure, scope, and space budget.
> This file covers **only the RAG/agentic triage workstream** of the broader project.

---

## Abstract [done]

Emergency department triage is time-sensitive and high-stakes, and both
under-triage and over-triage can have serious clinical and operational
consequences [18, 21]. Recent work by Gaber et al. suggested that
retrieval-augmented agentic workflows can support triage decisions [1], but
the public artifacts are limited to benchmark-construction code rather than a
reproducible RAG pipeline or packaged retrieval corpus [3]. We present an
open-source clinical RAG pipeline designed for local execution and agentic
orchestration, and evaluate it on Emergency Severity Index (ESI) triage
prediction using a MIMIC-IV-ED-derived benchmark. We find that, although
Gaber et al. reported gains from a multi-role agentic workflow guided by
iterative query refinement, the same general retrieval strategy did not
generalize to our corpus. By contrast, performance improved when agents were
given tools for selective access to task-aligned triage evidence. This
suggests that, for ESI triage, the main benefit of agency lies less in broad
biomedical article-corpus retrieval than in on-demand access to task-aligned
triage cases. We also include exploratory work on alternative retrieval
improvements, including reinforcement-learning-based retrieval tuning and
reranking, though these remain preliminary in the current workstream.

---

## Introduction

*Target: problem, why hard, why current approaches fail, key idea, main result.*

**Draft prose:**

Emergency department triage is a high-stakes ordinal decision problem: models
must distinguish among a small number of ordered acuity levels under severe
class imbalance, while the clinical consequences of different errors are not
symmetrical. Benchmark metrics such as exact accuracy and one-level-tolerant
range accuracy are useful for comparability, but they do not reveal whether a
system is systematically under-triaging urgent patients, over-triaging lower
acuity patients, or collapsing onto the dominant ESI-2/3 boundary. That makes
triage a useful stress test for agentic retrieval workflows: it is not enough
to retrieve more text; the system must retrieve the right evidence, use it at
the right time, and improve clinically meaningful behavior.

Our workstream began from a broad evidence-in-the-loop hypothesis. We first
tested PubMed Central article retrieval, including query rewriting,
multi-facet decomposition, and adaptive re-querying, under the assumption that
better query control might make biomedical literature RAG useful for triage.
Those experiments did not support that hypothesis. Query reformulation changed
the retrieved articles but did not materially improve exact accuracy, safety,
or class balance, and article-RAG variants continued to degrade macro recall
and ordinal agreement relative to handbook-based baselines at 1,000-case scale.

A different pattern emerged when we switched from generic articles to
task-aligned ESI handbook cases. Fixed handbook few-shot prompting improved
over earlier direct-prompting pilots, and adaptive handbook retrieval improved
further, especially on macro recall and rare-class recovery. The strongest
result of the agentic workstream was therefore not a maximal multi-agent
PubMed pipeline, but a simpler design: a single agent that can selectively
retrieve relevant handbook cases at inference time. The remainder of the paper
is organized around that distilled claim: which evidence source helps, which
access mode helps, and which additional orchestration turns out not to help
enough to justify its complexity.

---

## Related Work [done]

Gaber et al. [1] define the MIMIC-IV-Ext triage benchmark and provide the
natural starting point for this work, but their evaluation centers on exact
and range accuracy rather than the design of reproducible agentic retrieval
systems. We move beyond that metric frame because triage evaluation must also
capture asymmetric safety errors, class imbalance, and prevalence-sensitive
chance correction, which motivates our emphasis on under-/over-triage, macro
recall, and AC2 alongside benchmark accuracy [6, 10, 11, 16, 17, 18]. More
broadly, prior LLM-triage studies establish the clinical relevance of
automated triage [7, 8, 9], while the RAG and tool-use literature motivates
explicit external evidence access [13, 14, 15]; taken together, these lines
of work support shifting attention away from iterative query refinement as the
main design question and toward selective tool-mediated evidence access, yet
they still do not provide a reproducible agentic retrieval pipeline or a
packaged article corpus for follow-on experimentation.

---

## Approach (paper § Approach)

*Target: ~1 page. Architecture, tools, strategy space.*

**Draft prose:**

We treat ESI triage as a prediction task: given a patient's triage note,
vitals, and demographics, the model must assign one ESI level from 1 to 5. The
main question in this work is how the model should access outside evidence. We
compare four simple setups: `Few-Shot Only`, `Handbook Lookup Tool`, `Article
RAG`, and `Two-Step Review`.

**Main method: Handbook Lookup Tool.** Our main system is a single triage model
that can look up relevant cases from the ESI handbook during inference. Instead
of seeing the same examples on every case, the model decides when to search the
handbook and which cases to retrieve before making its final prediction. This
is the main agentic idea in the paper: on-demand access to task-aligned triage
cases.

**Baseline and ablations.** The main baseline is `Few-Shot Only`: the model
gets a fixed set of handbook examples in the prompt and makes a standalone
prediction with no retrieval. We compare this against two article-based
variants. In `Article RAG`, retrieved biomedical articles are added to the
prompt as extra context. In `Article Lookup Tool`, the model chooses when to
search the article corpus during inference. We also test `Two-Step Review`,
where one model makes an initial triage decision and a second model reviews it.

**Prompt variants.** Across these setups, we vary two secondary design choices.
First, we test two ways of writing the ESI rules in the prompt: standard prose
definitions and an explicit Step A-D decision tree. Second, we compare
single-model prediction against the two-step reviewer setup. These are
supporting ablations, not the main contribution.

**What is ours.** The benchmark task and clinical framing follow Gaber et al.
[1], and the general idea of retrieval and tool use follows prior RAG and agent
literature [13, 14, 15]. Our contribution is the local clinical RAG pipeline,
the handbook lookup tool, and the comparison between fixed examples, article
retrieval, handbook lookup, and two-step review for triage.

---

## Experiments

### Data (paper § Experiments → Data)

*Target: ~0.25 pages. Reference Gaber et al. for shared setup; only note differences.*

**Draft prose:**

We evaluate on the MIMIC-IV-Ext-Triage benchmark introduced by Gaber et al.
[1], derived from MIMIC-IV-ED 2.2 [2]. We follow their dataset construction
pipeline [3] and adopt their "clinical user" input setting, which provides the
model with triage-time information including chief complaint, history of present
illness, initial vitals, and demographics.

In addition to the benchmark cases themselves, the experiments use two
auxiliary evidence sources. The first is a case bank of 60 expert-classified
ESI triage vignettes (30 practice and 30 competency cases) extracted from the
ESI v4 Implementation Handbook [5]. Each case contains a clinical
presentation, the gold-standard ESI level, and a clinician-authored rationale.
The second is a corpus of 2.31M PubMed Central Open Access articles used for
article-retrieval experiments. These two sources support the paper's core
comparison between task-aligned handbook evidence and broad biomedical article
evidence.

We depart from Gaber et al. in three ways:

1. **Split structure and vitals-missing rows.** Gaber et al. [1] predict on
   n=2,200 deterministically selected cases (the first 2,200 rows surviving
   their MIMIC-IV preprocessing pipeline), then postprocess to n≈2,000 by
   dropping rows with missing vitals, empty specialty ground truth, LLM parse
   failures, and a positional `[:2000]` slice. We evaluate on the full
   n=2,200. Our task is triage only (no specialty or diagnosis prediction),
   so the specialty- and diagnosis-driven filters do not apply. We also
   deliberately retain the 117 rows (5.3%) with missing `initial_vitals`
   (statistics verified by `experiments/analysis/missing_vitals_analysis.py`).
   Analysis on dev\_tune (n=1,000) shows these rows are heavily skewed toward
   ESI-1: 46 of 61 missing-vitals rows (75%) are ESI-1, representing 39% of
   all ESI-1 cases in that split. This is clinically expected — ESI-1 patients
   often bypass full triage recording for immediate resuscitation. Dropping
   them would disproportionately remove the most critical patients and
   artificially deflate ESI-1 representation. In the agentic pipeline used
   for all final evaluation runs, missing vitals are rendered in the prompt
   as "Missing/not recorded" rather than passed through as a literal NaN.
   The remaining ~21k cases form a fixed development
   partition: `dev_tune` (n=1,000) for iterative strategy selection and
   `dev_holdout` (n=500) for single-use confirmation, both seeded
   deterministically (seed=42).

2. **Model family.** Gaber et al. [1] evaluate Claude 3/3.5 Sonnet and Haiku.
   We evaluate Gemini 3 Flash and Gemini 2.5 Flash (via Vertex AI), both
   permitted under the PhysioNet Credentialed Data Use Agreement [4].

3. **Evaluation metrics.** Gaber et al. [1] report exact accuracy and range
   accuracy. We extend this with safety metrics (under/over-triage rates)
   and chance-corrected ordinal agreement (Gwet AC2). See § Evaluation method.

**Background notes (not for paper):**

- ESI distribution: ESI 1 ~12%, ESI 2 ~45%, ESI 3 ~43%, ESI 4 <1%.
  ESI 4 too rare for reliable per-class metrics (86 cases in dev, 12 in test).
- One row per patient enforced via `drop_duplicates(subset=['subject_id'])`.
- Temporal ordering check: no detectable distribution shift between first/last
  2200 rows (acuity within 1.3pp, chief complaints within 0.5pp).
- PhysioNet permitted providers: Azure OpenAI, Amazon Bedrock, Google Gemini
  (Vertex AI), Anthropic Claude. We use Gemini and Claude.

---

### Experimental details

### Vector search infrastructure

For article-backed RAG, we use a local FAISS IVF4096 index plus a local SQLite
article store rather than live BigQuery retrieval. The index is exported once
from the precomputed PMC embedding table [20], and all experiment-time queries
are served locally. The local article sidecar stores up to 8,000 characters per
article, matching the default retrieval context budget used in the final PMC
experiments.

### Prompting and decoding

All prompts follow the "clinical user" framing of Gaber et al. [1], presenting
the model with a triage nurse persona and the patient's chief complaint,
history of present illness, vitals, and demographics. Across configurations, we
test two encodings of the ESI rules: a flat-prose definition (e.g., "1:
Requires immediate lifesaving intervention ... 5: No
diagnostic/therapeutic interventions needed beyond physical exam") and a
structured decision-tree prompt (e.g., "Step A - Immediate lifesaving
intervention needed? ... Step D - Vital-sign danger zone check"). All
generation uses temperature 0.0 and structured JSON output, with automatic
retry on rate limits.

### ESI case-bank tool

The tool-use prompt strategy (`--prompt-template tool_use` in
`experiments/query_strategy_sweep.py`) gives the LLM access to a
function-calling tool called `search_esi_case_bank` instead of injecting
few-shot examples directly into the prompt. The model decides when and how to
query the case bank during its reasoning.

**Tool interface.** The tool schema (`src/rag/case_bank.py:170-207`,
`CASE_BANK_TOOL_SCHEMA`) exposes three optional filter parameters:

- `esi_level` (int, 1-5): return only cases assigned this ESI level
- `keywords` (string): space-separated terms; all must match in vignette or
  rationale (case-insensitive)
- `chapter` (`practice` | `competency`): restrict to one chapter

Results are capped at 10 (hard cap 15). The agentic pipeline
(`src/rag/agentic_pipeline.py`) manages the tool-use loop: the system prompt
advertises the tool, the model emits one or more `function_call` parts, the
pipeline executes each call against the local case bank, and the model
continues until it returns a final ESI prediction.

**Observed usage patterns.** On Gemini 3 Flash at n=1,000, the `tool_call_count`
column shows a median of 4 calls per row, with 64% of rows making exactly 4
calls, 1% making 0 calls, and a maximum of 11. This suggests the model usually
queries the case bank as a comparative evidence source rather than as a single
one-shot lookup.

### Compared configurations (decisive runs, n=1,000 leaderboard)

| # | Name | Strategy | Key architectural difference |
|---|---|---|---|
| 1 | **G3-tool** (anchor) | `tool_use` | Single-agent, case-bank tool available, model decides when/whether to call it |
| 2 | G3 decision_tree | `tool_use_dt` | Same as #1 but system prompt replaces flat ESI prose with structured Step A→D decision algorithm |
| 3 | G3 tool_use_pmc | `tool_use_pmc` | Single-agent, PMC literature search tool only — no case bank available |
| 4 | G3 two_role | `two_role` | Two-stage: nurse agent drafts initial assessment, critic agent reviews with case-bank tool. Both roles ran on the same model (Flash); architecture supports separate fast/main models |
| 5 | **G3-ref** | `fewshot` | No tools — few-shot examples injected directly into prompt, model predicts ESI from patient presentation |
| 6 | G3-gated | `fewshot+gate` | Same as #5 but PMC articles injected when embedding distance < 0.25 |
| 7 | **W1-ref** (safe best) | `fewshot` | Same as #5 but on G2.5 Flash — different model, same strategy |
| 8 | W2B tool_use | `tool_use` | Same as #1 but on G2.5 Flash |
| 9 | W1-gated | `fewshot+gate` | Same as #6 but on G2.5 Flash |
| 10 | W1-rag | `fewshot+RAG` | Few-shot + PMC articles always injected (no distance gate) |
| 11 | W2B tool_rag | `tool_use+RAG` | Tool-use case bank + passive PMC preload in context |

---

### Evaluation method

We evaluate the triage systems with a multilevel metric stack designed for
clinical decision support rather than generic classification alone. Exact
accuracy is retained for benchmark comparability, but it is not sufficient on
its own. Because triage errors are directionally asymmetric, we report both
under-triage and over-triage rates. To detect collapse into the dominant
ESI-2/3 classes, we track macro recall and per-class recall, which surface
failures on rare but clinically important classes even when headline accuracy
looks acceptable. Finally, we use Gwet AC2 as the primary chance-adjusted
ordinal summary because the label space is ordered and highly imbalanced.
Range accuracy and quadratic kappa are retained as diagnostics, but both can
look favorable when a system simply shifts predictions upward by one level.

### Why no single metric is sufficient

ESI triage is ordinal (1–5) with heavy class imbalance: ESI 2 ~45%, ESI 3 ~43%, ESI 1 ~12%,
ESI 4 <1%. This reflects real ED populations — Gaber et al. [1] report the same pattern.

- **Exact accuracy** rewards majority-class exploitation: a model that always predicts ESI 3
  scores ~43% for free with no indication it has learned nothing.
- **Range accuracy** has a high chance baseline under imbalance: because ESI 2–3 dominate
  (~88% of cases) and the acceptable window is two levels wide, a random predictor concentrated
  on ESI 2–3 achieves high range accuracy by luck.
- **QWK** is chance-corrected and ordinal, but is brittle under prevalence and marginal
  distortions (the "kappa paradox" [10][11]). By contrast, Gwet-style agreement
  coefficients were designed to be more stable under high agreement and skewed
  category distributions [16][17]. In constant-prediction bins QWK collapses to
  0 even when most predictions are correct and all errors are in the clinically
  safe direction.

No single metric captures all three dimensions (benchmark comparability, clinical safety,
behavioral health).

### Adopted metric stack

Results are evaluated using a **multilevel metric stack** rather than a single headline score:

1. **Benchmark** — `exact_accuracy`, `range_accuracy` (for comparability with Gaber et al. [1])
2. **Safety** — `under_triage_rate`, `over_triage_rate` (asymmetric clinical harm)
3. **Imbalance diagnostics** — `macro_recall`, `per_class_recall`, `MA-MAE` (collapse detection)
4. **Chance-adjusted diagnostics** — `linear_weighted_gwet_ac2` (preferred; less prevalence-
   sensitive than kappa [16][17]), `quadratic_weighted_kappa` (collapse alarm only)

Promotion decisions require passing gates at each level — see `experiment_results.md §
Promotion ladder`.

---

### Results

The experimental arc began with a broad evidence-in-the-loop hypothesis: ED
triage might improve if the model could retrieve and refine external evidence
before committing to an ESI decision. We first tested that idea in the most
direct way, using PubMed Central retrieval with increasingly elaborate
retrieval control. Those experiments did not support the idea that better
query manipulation would fix article-based RAG. On the 150-case query
ablation, simple concatenated retrieval reached 56.0% exact accuracy, LLM
query rewriting was effectively unchanged at 54.7%, and HPI-only retrieval
fell to 50.7%. Multi-facet decomposition performed worse still at 54.0%
exact accuracy. Later, adaptive re-querying reached 58.0% exact accuracy, but
still failed to outperform the simpler gated retrieval configuration on the
full metric stack. The lesson from this phase was that increasingly elaborate
query control changed the retrieved articles but did not make them reliably
more useful for triage.

That conclusion became clearer once we evaluated article retrieval at
1,000-case scale. On Gemini 2.5 Flash, the fixed handbook few-shot baseline
achieved 53.4% exact accuracy, 10.8% under-triage, 0.378 macro recall, and
0.727 AC2. Replacing that setup with unconditional article RAG reduced
performance on nearly every clinically relevant dimension: exact accuracy fell
to 50.1%, macro recall to 0.316, and AC2 to 0.636, while under-triage
remained essentially flat at 10.9%. Gated article RAG was less harmful but
still did not help enough to justify itself, reaching 51.9% exact accuracy,
11.8% under-triage, 0.341 macro recall, and 0.719 AC2. In other words,
article retrieval did not reliably improve safety, class balance, or ordinal
agreement; the problem was not simply how the literature was queried, but
whether a broad biomedical article corpus was the right evidence source for ESI
triage at all.

A different pattern emerged when we switched from broad biomedical article evidence to
task-aligned ESI handbook cases. Fixed handbook few-shot prompting was the
first evidence source that produced a consistent gain over earlier
direct-prompting pilots. We then made that evidence access adaptive. In the
150-case pilot, allowing the model to retrieve handbook cases on demand
produced 61.1% exact accuracy, compared with 58.0% for fixed handbook
few-shot prompting, while also reducing over-triage from 34.0% to 28.5%. At
1,000-case scale on Gemini 3 Flash, this adaptive handbook retrieval design
remained the strongest single-agent configuration. It reached 62.7% exact
accuracy, 21.6% under-triage, 0.467 macro recall, and 0.775 AC2, and was the
only decisive run to achieve any recovery of ESI-4 (recall_ESI4 = 0.333). The
same-model fixed handbook baseline was close on headline accuracy but clearly
weaker on the broader stack, reaching 61.6% exact accuracy, 26.4%
under-triage, 0.381 macro recall, 0.766 AC2, and complete ESI-4 collapse.
This is why the paper should center not on "tool use" as a generic mechanism,
but on adaptive handbook retrieval as the main agentic design: the model
benefits when it can selectively access task-aligned triage cases rather than
rely on fixed exemplars or broad article retrieval.

The remaining final configurations are then naturally motivated as explanatory
comparisons rather than as unrelated leaderboard entries. The fixed handbook
baseline isolates the value of adaptive evidence access when the model and
evidence family are held constant. The Gemini 2.5 handbook run anchors the
safer, more conservative error profile. The PubMed Central variants test
whether the gain comes from agentic retrieval itself or from the alignment of
the retrieved evidence source: an adaptive article-retrieval system asks
whether agency alone is sufficient, and gated article RAG asks whether passive
article injection can substitute for model-controlled retrieval. Finally, the
two-stage reviewer configuration retains the spirit of the milestone's
critique-and-refinement loop, but functions here as an orchestration ablation
rather than the main claim. The overall experimental story is therefore not
that a larger multi-agent PubMed pipeline won, but that development
progressively distilled the useful agentic contribution down to a narrower and
more defensible claim: for ED triage, the strongest agentic pattern is
adaptive access to handbook-aligned evidence.

---

## Analysis

This section scaffolds the paper's central analytical contribution: showing that
benchmark-only evaluation (exact accuracy, range accuracy) is insufficient for
clinical triage, and demonstrating what becomes visible when safety and
chance-corrected metrics are added.

### 1. Why we extend beyond Gaber's benchmark metrics

Gaber et al. [1] report two metrics: exact accuracy and range accuracy. We
argue this is insufficient for three reasons:

- **Range accuracy masks systematic over-triage.** A model that shifts every
  ESI-3 prediction to ESI-2 scores perfectly on range accuracy for those cases
  (over-triage by one level is forgiven). This hides a clinically harmful
  resource-allocation pattern — the metric cannot distinguish safe conservatism
  from indiscriminate escalation. [TODO: compute hypothetical range accuracy of
  an always-overTriage-by-1 predictor on our data to quantify this ceiling]

- **Neither metric captures directional safety asymmetry.** Under-triage (patient
  assigned lower acuity than needed) can delay evaluation and treatment for
  high-risk conditions, whereas over-triage (assigned higher acuity) primarily
  wastes resources and is generally clinically safer [18]. Exact
  accuracy treats both error directions identically. Range accuracy partially
  addresses this by forgiving over-triage, but only by 1 level and only for
  ESI 2–5 — it cannot quantify the rate or magnitude of under-triage.

- **Neither metric detects class collapse.** A model that predicts ESI-2 for
  every patient achieves ~45% exact accuracy and ~88% range accuracy on our
  distribution — appearing competent while providing zero discrimination for
  ESI-1 (life-threatening) and ESI-3+ (lower acuity) patients. This failure
  mode is invisible to benchmark metrics but immediately flagged by per-class
  recall and macro recall.

We therefore add two metric tiers beyond benchmark:

| Tier | Metrics | What it catches |
|---|---|---|
| **Safety** | under_triage_rate, over_triage_rate | Directional error asymmetry — is the model dangerous or just wasteful? |
| **Chance-corrected agreement** | Gwet AC2, per-class recall, macro recall | Class collapse, prevalence-inflated agreement, minority-class blindness |

### 2. What the extended metrics reveal

[TODO: fill with final numbers from best configs on test set. Structure below
uses dev_tune 1K leaderboard numbers as placeholders.]

**The benchmark-safety paradox.** Our two best-performing configs by benchmark
metrics exhibit opposite safety profiles:

- **G3-tool** (exact 62.7%, range 78.3%): highest benchmark scores but
  under-triage 21.6% — one in five patients assigned *lower* acuity than
  ground truth. Safety-blocked.
- **W1-ref** (exact 53.4%, range 88.3%): highest range accuracy but
  over-triage 35.8% — one in three patients unnecessarily escalated.
  Safe but wasteful.

Under Gaber's metrics alone, W1-ref looks superior (range 88.3% vs 78.3%).
The safety tier reveals this is an artifact of systematic over-triage, not
genuine accuracy. [TODO: compute what Gaber's reported models would score on
our safety metrics if confusion matrices were available]

**Class collapse hidden by accuracy.** [TODO: show that configs with similar
exact accuracy (e.g., G3-tool 62.7% vs G3-ref 61.6%) have dramatically
different per-class recall profiles. One recovers ESI-4 (0.333), the other
shows complete ESI-4 collapse (0.000). This difference is invisible to
aggregate accuracy.]

**Chance-corrected agreement reranks the leaderboard.** [TODO: show specific
cases where AC2 ranking differs from exact-accuracy ranking, demonstrating
that prevalence-adjusted agreement changes which configs appear best]

### 3. Confusion matrix comparison: G3-tool vs W1-ref

[TODO: generate and include 5×5 confusion matrices (heatmaps or tables) for
the two best-performing configs on the test set]

**G3-tool (best benchmark) confusion matrix:**

[TODO: 5×5 table. Expected pattern: strong ESI-2/3 diagonal, under-triage
visible as off-diagonal mass below the diagonal (predicted lower than true),
partial ESI-4 recovery, weak ESI-1 recall]

**W1-ref (best safety) confusion matrix:**

[TODO: 5×5 table. Expected pattern: mass shifted above the diagonal
(systematic over-triage), high ESI-2 recall at the expense of ESI-3, complete
ESI-4 collapse, high range accuracy from over-triage forgiveness]

**Side-by-side interpretation:**

- Where do the two models disagree? [TODO: compute the set of rows where G3-tool
  and W1-ref predict differently; characterize by true ESI level]
- What is the clinical consequence of each error pattern? [TODO: map confusion
  matrix cells to clinical outcomes — e.g., ESI-3 predicted as ESI-2 means
  unnecessary immediate bed assignment; ESI-2 predicted as ESI-3 means delayed
  evaluation of a genuinely urgent patient]
- Which error pattern is more acceptable in a real ED? [TODO: frame in terms
  of the safety-efficiency tradeoff that ED clinicians face daily]

### 4. Framing for the paper

The analytical contribution is not "our model is better than Gaber's" (different
models, different splits, not directly comparable). Instead:

- **Methodological:** benchmark-only evaluation is blind to safety-critical
  failure modes in triage. We propose and validate a multilevel metric stack.
- **Empirical:** when safety and collapse metrics are added, the leaderboard
  reranks — the "best" model depends on which tier you prioritize.
- **Clinical:** the confusion matrix comparison makes the safety-efficiency
  tradeoff concrete and interpretable for ED clinicians.

---

## Conclusion

**Draft prose:**

The main finding of this workstream is narrower than the milestone's original
vision, but stronger. Query-refined and multi-step PubMed RAG did not emerge
as the useful agentic lever for ESI triage. Instead, development experiments
progressively distilled the contribution down to adaptive access to
handbook-aligned evidence. The strongest single-agent design was a Gemini 3
system that could selectively retrieve ESI handbook cases during inference,
outperforming the same-model fixed-handbook baseline on exact accuracy, macro
recall, AC2, and rare-class recovery. Just as importantly, the evaluation
stack showed why benchmark-only conclusions would have been misleading:
architectures with attractive range accuracy could still be clinically
unbalanced, and retrieval variants that looked plausible in the abstract could
still fail on safety and collapse metrics.

The work also clarifies the limits of the current framework. The PMC corpus is
only weakly aligned with the triage task, abstract-level retrieval remains a
constraint, and the strongest benchmark configuration still carries too much
under-triage to be called deployment-ready. Additional orchestration beyond
single-agent adaptive handbook retrieval did not yet justify its added
complexity. The clearest next step is therefore not "more agents" in the
abstract, but better task-aligned evidence access, better error-sensitive
evaluation, and stronger analysis of when the system should defer rather than
predict.

### Limitations

The PMC corpus is an unfiltered Google-hosted mirror: by our earlier corpus
audit, roughly 11.5% of records would be excluded by stricter quality filters
such as retractions, non-commercial licenses, or pre-1990 content. In
addition, the local PMC article store keeps only the first 2,000 characters of
each article for prompt-time context, and `text-embedding-005` truncates
embedding inputs beyond 2,048 tokens [20]. Together, these design choices can
underweight evidence that appears later in long full-text articles.
The BigQuery snapshot may also lag the canonical NIH/PMC Open Access list.

A separate limitation concerns missing vital signs. In the final generation
pipeline, missing vitals are rendered explicitly to the model as
"Missing/not recorded," but retrieval-side query representations do not always
normalize missingness in the same way. As a result, performance on
missing-vitals cases may reflect both genuine robustness to incomplete triage
information and residual retrieval artifacts, rather than a cleanly isolated
effect of missing data.

### Future work

#### Expected triage cost metric

The current metric stack evaluates safety via directional error rates (`under_triage_rate`,
`over_triage_rate`) but does not assign differential penalties by error magnitude. An
`expected_triage_cost` metric — computed from a clinician-approved asymmetric cost matrix —
would encode clinical harm directly: an ESI-3 patient triaged as ESI-5 is far more dangerous
than one triaged as ESI-4. This is the cleanest way to collapse the safety dimension into a
single comparable scalar.

**How it would improve the analysis:**
- Makes promotion decisions between configs with similar directional error rates but different
  error *magnitudes* (e.g., two configs with 15% under-triage but different severity
  distributions of that under-triage)
- Allows cost-sensitive threshold tuning for production deployment
- Could align evaluation with triage reliability literature on severity-sensitive
  agreement weights [TODO: add source]

**Why deferred:** Designing and validating the cost matrix requires clinical collaborator
input (emergency medicine physician review of relative harm weights per ESI-level transition).
The matrix should be pre-specified before any evaluation, not fit post-hoc. Without clinical
grounding, arbitrary weights would add false precision rather than genuine signal.

**Implementation:** retrocomputable from existing run CSVs — each row contains per-case
y_true and y_pred pairs. Once a cost matrix is defined, all historical runs can be
re-evaluated without re-running experiments.

#### Other future work

- Proper chunking and re-embedding for full-article retrieval (currently truncated at 2048 tokens)
- Metadata filtering by journal and publication year to improve ED relevance
- Migrate to dedicated vector store (Qdrant/Pinecone) if production latency becomes a requirement
- Failure case analysis against Nature paper baseline — which conditions/presentations were
hardest? See issue #6.

---

## Bibliography

**[1]** Gaber F, Shaik M, Allega F, et al. "Evaluating large language model workflows in
clinical decision support for triage and referral and diagnosis." *npj Digital Medicine*.
2025;8:263. doi:10.1038/s41746-025-01684-1.
[https://www.nature.com/articles/s41746-025-01684-1](https://www.nature.com/articles/s41746-025-01684-1)
— Primary external benchmark comparison. Defines the MIMIC-IV-Ext triage setup and reports
exact/range accuracy for Claude-family workflows, but does not report kappa or human
inter-rater benchmarks.

**[2]** Johnson A, Bulgarelli L, Pollard T, Celi LA, Mark R, Horng S. *MIMIC-IV-ED*
(version 2.2). *PhysioNet*. 2023. RRID:SCR_007345. doi:10.13026/5ntk-km72.
[https://physionet.org/content/mimic-iv-ed/2.2/](https://physionet.org/content/mimic-iv-ed/2.2/)
— Source of the ED cases underlying MIMIC-IV-Ext. Accessed via the medLLMbenchmark
creation script [3].

**[3]** BIMSBbioinfo/medLLMbenchmark.
[https://github.com/BIMSBbioinfo/medLLMbenchmark](https://github.com/BIMSBbioinfo/medLLMbenchmark)
— Upstream benchmark repository. Source of the MIMIC-IV-Ext dataset creation script and
the Claude triage prompt template adapted for this project.

**[4]** PhysioNet. "Responsible Use of MIMIC Data with Online Services Like GPT."
*PhysioNet News*, 18 April 2023. Accessed 24 February 2026.
[https://physionet.org/news/post/gpt-responsible-use/](https://physionet.org/news/post/gpt-responsible-use/)
— Establishes permitted LLM providers for MIMIC-credentialed data. Justifies selection
of Google Gemini (via Vertex AI) and Anthropic Claude as compliant providers.

**[5]** Gilboy N, Tanabe P, Travers DA, Rosenau AM, Eitel DR. *Emergency Severity Index,
Version 4: Implementation Handbook*. AHRQ Publication No. 05-0046-2. Rockville, MD:
Agency for Healthcare Research and Quality. May 2005.
Downloaded from: [https://sgnor.ch/fileadmin/user_upload/Dokumente/Downloads/Esi_Handbook.pdf](https://sgnor.ch/fileadmin/user_upload/Dokumente/Downloads/Esi_Handbook.pdf)
— Contains 60 expert-classified ESI triage cases (30 practice + 30 competency) used as
the source for few-shot examples and the tool-use case bank.

**[6]** Mirhaghi A, Heydari A, Mazlom R, Hasanzadeh F. "Reliability of the Emergency Severity
Index: Meta-analysis." *Sultan Qaboos Univ Med J*. 2015;15(1):e71–e77.
PMC: [https://pmc.ncbi.nlm.nih.gov/articles/PMC4318610/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4318610/)
— Meta-analysis of 19 studies (40,579 cases). Nurse–expert κ = 0.732 (95% CI: 0.625–0.812),
expert–expert κ = 0.900. Used to contextualize model performance.

**[7]** Kim JH, Kim SK, Choi J, Lee Y. "Reliability of ChatGPT for performing triage task in the
emergency department using the Korean Triage and Acuity Scale." *Digital Health*. 2024;10.
PMC: [https://pmc.ncbi.nlm.nih.gov/articles/PMC10798071/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10798071/)
— ChatGPT 3.5 κ = 0.320 vs human inter-rater κ = 0.646. Best practice of reporting LLM κ
alongside clinician κ on the same cases.

**[8]** Görgülü et al. "Emergency department triaging using ChatGPT based on emergency severity
index principles." *Scientific Reports*. 2024;14.
[https://www.nature.com/articles/s41598-024-73229-7](https://www.nature.com/articles/s41598-024-73229-7)
— GPT-4 κ = 0.899 vs triage team κ = 0.893. Model matched clinician-level agreement.

**[9]** Ivanov et al. "Evaluating LLM-based generative AI tools in emergency triage." *American
Journal of Emergency Medicine*. 2024.
[https://www.sciencedirect.com/science/article/abs/pii/S0735675724007071](https://www.sciencedirect.com/science/article/abs/pii/S0735675724007071)
— ChatGPT κ = 0.537, triage nurses κ = 0.477, Copilot κ = 0.472 (all vs EM physician).

**[10]** Feinstein AR, Cicchetti DV. "High agreement but low kappa: I. The problems of two
paradoxes." *J Clin Epidemiol*. 1990;43(6):543–549.
— Describes the "high agreement but low kappa" paradoxes. Motivates the shift from
QWK-primary to multilevel metric stack.

**[11]** Warrens MJ. "Some Paradoxical Results for the Quadratically Weighted Kappa."
*Psychometrika*. 2012.
[https://doi.org/10.1007/s11336-012-9258-4](https://doi.org/10.1007/s11336-012-9258-4)
— Formally questions QWK's suitability as a single ordinal agreement summary. Supports
not relying on QWK as the sole ordinal agreement metric.

**[12]** Liu NF, Lin K, Hewitt J, et al. "Lost in the Middle: How Language Models Use Long
Contexts." *Transactions of the Association for Computational Linguistics*. 2024;12:157–173.
— Context-placement effects. Relevant to the finding that more context consistently
degraded performance.

**[13]** Lewis P, Perez E, Piktus A, et al. "Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks." *Advances in Neural Information Processing Systems*.
2020;33.
[https://papers.nips.cc/paper_files/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html](https://papers.nips.cc/paper_files/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)
— Foundational RAG citation. Supports the general framing of combining parametric generation
with non-parametric retrieval.

**[14]** Yao S, Zhao J, Yu D, Du N, Shafran I, Narasimhan KR, Cao Y. "ReAct: Synergizing
Reasoning and Acting in Language Models." *International Conference on Learning
Representations (ICLR)*. 2023.
[https://openreview.net/forum?id=WE_vluYUL-X](https://openreview.net/forum?id=WE_vluYUL-X)
— Canonical citation for interleaving model reasoning with external actions, including
search-style tool use.

**[15]** Schick T, Dwivedi-Yu J, Dessì R, et al. "Toolformer: Language Models Can Teach
Themselves to Use Tools." *Advances in Neural Information Processing Systems*. 2023;36.
[https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html)
— Tool-use citation. Supports the design choice of giving the model explicit external tools
rather than only passive context injection.

**[16]** Gwet KL. "Computing inter-rater reliability and its variance in the presence of
high agreement." *British Journal of Mathematical and Statistical Psychology*.
2008;61(Pt 1):29–48. doi:10.1348/000711006X126600.
[https://pubmed.ncbi.nlm.nih.gov/18482474/](https://pubmed.ncbi.nlm.nih.gov/18482474/)
— Primary methodological source for the AC-family agreement coefficients and their behavior
under high-agreement settings.

**[17]** Quarfoot D, Levine RA. "How Robust Are Multirater Interrater Reliability Indices
to Changes in Frequency Distribution?" *The American Statistician*. 2016;70(4):373–384.
doi:10.1080/00031305.2016.1141708.
[https://doi.org/10.1080/00031305.2016.1141708](https://doi.org/10.1080/00031305.2016.1141708)
— Supports preferring Gwet AC2 over kappa-family metrics under class imbalance and
prevalence shift.

**[18]** Sax DR, Warton EM, Mark DG, Reed ME. "Emergency Department Triage Accuracy and
Delays in Care for High-Risk Conditions." *JAMA Network Open*. 2025;8(5):e258498.
doi:10.1001/jamanetworkopen.2025.8498.
[https://pubmed.ncbi.nlm.nih.gov/40314952/](https://pubmed.ncbi.nlm.nih.gov/40314952/)
— Outcome-based justification for separating under-triage from over-triage: under-triage was
associated with delayed diagnostic and treatment orders for high-risk ED conditions.

**[19]** Wuerz RC, Milne LW, Eitel DR, Travers D, Gilboy N. "Reliability and validity of a
new five-level triage instrument." *Academic Emergency Medicine*. 2000;7(3):236–242.
doi:10.1111/j.1553-2712.2000.tb01066.x.
[https://pubmed.ncbi.nlm.nih.gov/10730830/](https://pubmed.ncbi.nlm.nih.gov/10730830/)
— Foundational validation study for the ESI framework itself.

**[20]** Google Cloud. "Text embeddings API." *Generative AI on Vertex AI* documentation.
[https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api)
— Official model reference for `text-embedding-005`; documents the model as specialized for
English and code tasks, with up to 768 output dimensions and a 2048-token maximum sequence length.

**[21]** Suamchaiyaphum K, Jones AR, Markaki A. "Triage Accuracy of Emergency Nurses: An
Evidence-Based Review." *Journal of Emergency Nursing*. 2024;50(1):44–54.
doi:10.1016/j.jen.2023.10.001.
[https://pubmed.ncbi.nlm.nih.gov/37930287/](https://pubmed.ncbi.nlm.nih.gov/37930287/)
— Review article used here to motivate the stakes of triage accuracy and the persistence of
both under-triage and over-triage in emergency nursing practice.

---

## Appendix: Ground-truth label reliability: ESI inter-rater agreement

### The problem: MIMIC-IV ESI labels are single-nurse assignments

The Emergency Severity Index (ESI) is a validated five-level triage instrument [19], but
the MIMIC-IV-ED triage labels are assigned by one triage nurse per patient. There is no
inter-rater reliability data reported for the MIMIC-IV-ED dataset itself — no second nurse
independently triaged the same patients, and no adjudication process is documented.

### Gaber et al. (2025) do not address this

Gaber et al. [1] treat MIMIC-IV ESI labels as ground truth without validation. Their only
concession is the asymmetric range accuracy metric, which forgives over-triage by one level
but penalizes under-triage. This bakes in a clinical prior ("err on the side of caution")
but is not empirically justified against the MIMIC-IV labels themselves.

They acknowledge that ESI assignment "can vary due to the clinician's intuition and
experience" and that using clinical notes "may introduce bias since triage determination is
typically made prior to the patient being fully assessed by a physician," but they do not
quantify or control for label noise.

### ESI inter-rater reliability benchmarks from published literature

Mirhaghi et al. [6] — meta-analysis of 19 studies (40,579 cases, 6 countries, 2000–2013):


| Agreement type | Pooled κ | 95% CI      |
| -------------- | -------- | ----------- |
| Overall        | 0.791    | 0.752–0.825 |
| Expert–expert  | 0.900    | 0.570–0.980 |
| Nurse–expert   | 0.732    | 0.625–0.812 |
| Weighted κ     | 0.796    | 0.751–0.834 |
| Unweighted κ   | 0.770    | 0.674–0.841 |
| Intra-rater    | 0.873    | 0.801–0.921 |


Individual study kappa values ranged from 0.46 (moderate) to 0.98 (almost perfect).

### Interpretive bounds for model κ

- **κ < 0.625** → below the lower CI of nurse–expert agreement; clear room to improve.
- **κ ∈ [0.625, 0.812]** → within nurse–expert range; model performs at clinician level.
- **κ > 0.812** → exceeds typical nurse–expert agreement; approaching expert–expert territory.

Early experiments are well below the floor, so label noise is not yet the binding constraint.

### LLM-triage literature: κ reported alongside human agreement

Gaber et al. [1] do not report kappa or human inter-rater benchmarks. Recent LLM-triage
papers increasingly do, and the comparison strengthens interpretation:

**Kim et al. [7]** — ChatGPT on Korean Triage and Acuity Scale (202 virtual cases).
Directly compared LLM κ against human inter-rater κ on the same cases.


| Rater                       | κ vs gold standard |
| --------------------------- | ------------------ |
| Human raters (3 paramedics) | 0.646              |
| ChatGPT 4.0                 | 0.523              |
| ChatGPT 3.5                 | 0.320              |


Note: ChatGPT 3.5 κ = 0.320 is nearly identical to our E00.5 baseline (κ = 0.338).

**Görgülü et al. [8]** — GPT-4 on ESI triage (Scientific Reports). Reported model and
clinician κ side by side against an expert gold standard.


| Rater       | κ vs gold standard |
| ----------- | ------------------ |
| Triage team | 0.893              |
| GPT-4       | 0.899              |


Here the model matched clinician-level agreement — but the key methodological point is that
both are reported together, giving the reader a direct frame of reference.

**Ivanov et al. [9]** — ChatGPT, Copilot, and triage nurses vs EM physician.


| Rater         | κ vs physician |
| ------------- | -------------- |
| ChatGPT       | 0.537          |
| Triage nurses | 0.477          |
| Copilot       | 0.472          |

**Takeaway:** Reporting κ alongside human agreement is emerging as best practice in the
LLM-triage literature. It contextualizes model performance against the inherent disagreement
in the task, rather than treating a raw κ value in isolation. We adopt this framing.

### Implications for this project

1. **Kappa ceiling.** Our measured model κ is bounded by the reliability of the labels. If
   nurse–expert agreement is κ ≈ 0.73, then a model achieving κ = 0.73 against single-nurse
   labels would be performing at the level of a second expert — not "only moderate agreement."
   Label noise is not yet the binding constraint but will become relevant as model performance
   improves.
2. **Benchmark target.** The nurse–expert pooled κ of 0.732 (95% CI: 0.625–0.812) is a
  meaningful benchmark: a model reaching this range would be performing at clinician-level
   agreement with the triage nurse. The expert–expert κ of 0.900 represents a soft upper
   bound for what any system could achieve against single-nurse labels.
3. **Asymmetric range accuracy is a workaround, not a solution.** Gaber's [1] range metric
  compensates for label noise by allowing over-triage flexibility, but it conflates two
   distinct phenomena: (a) the model correctly identifying higher acuity that the nurse missed,
   and (b) the model systematically over-triaging. Without ground-truth adjudication, these
   are indistinguishable.
4. **Paper framing.** Results should report κ alongside the published inter-rater benchmarks
  to contextualize model performance. A statement like "model κ = X against single-nurse
   labels (published nurse–expert κ = 0.73)" gives the reader the necessary frame of reference.
