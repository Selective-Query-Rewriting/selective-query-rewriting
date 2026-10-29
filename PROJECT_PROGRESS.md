# PROJECT_PROGRESS.md

> **Authoritative project progress document — update this file as implementation proceeds.**
> A new teammate must be able to understand the entire project from this single file.
>
> Last updated: 2026-10-29 | Audit pass by lead research engineer

---

## 1. Project Overview

**Title (provisional):** *When Is Rewriting Worth Its Cost? Oracle Headroom and Cost-Aware Gating for Conversational Retrieval*

**Institution:** Ramaiah Institute of Technology, Department of Computer Science and Engineering
**Course:** 7th Semester – Skill Enhancement Lab: Generative AI (CSL75)
**Period:** Aug 2026 – Dec 2026
**Supervisor:** Dr. Sangeetha V

**Team:**
| Name | Roll No |
|---|---|
| Kirti Sharma | 1MS23CS089 |
| Laksha Ravi | 1MS23CS096 |
| Mayur Shetty | 1MS23CS107 |

**Repository:** https://github.com/Selective-Query-Rewriting/selective-query-rewriting.git
**Branch:** `main` (clean, up to date with origin)
**Git commits (3 total):**
1. `9ba545f` – Initial commit
2. `f0df3ae` – Add dataset inspection and BM25 retrieval foundation
3. `dfd7edb` – Add query formulation, LLM rewriting, and caching pipeline

---

## 2. Research Problem and Motivation

Conversational Retrieval-Augmented Generation (RAG) systems often rewrite every user turn into a standalone query using dialogue history before retrieval. Rewriting resolves pronouns and ellipsis, but:

- **Unnecessary cost**: self-contained queries are rewritten anyway (wasting LLM calls, tokens, latency).
- **Harmful rewrites**: a rewrite can over-specify, drift intent, or introduce noise — reducing retrieval quality compared with the original query.
- **Average scores hide failures**: a rewriting model may improve mean retrieval while harming specific queries.
- **Existing gates**: prior work proposes heuristic or classifier-based gates but lacks systematic help/tie/hurt analysis, oracle headroom quantification, and rigorous cost–quality trade-off reporting.

**Core question:** *Can a low-cost, pre-retrieval gate decide when an LLM rewrite is worth its latency and cost, improving conversational retrieval quality while reducing unnecessary rewriting?*

---

## 3. Research Objectives and Questions

### Primary research question
How much does conversational query rewriting help, leave unchanged, or harm retrieval, and can a lightweight pre-retrieval gate capture useful quality–cost trade-offs by rewriting only selected queries?

### Objectives
1. **Analyse** QReCC to determine when query rewriting improves, leaves unchanged, or degrades retrieval; classify harmful rewrites by cause.
2. **Compare** three practical query formulations (KEEP, CONCAT, REWRITE) on the same retrieval infrastructure.
3. **Quantify oracle headroom**: how much could a hypothetical perfect gate improve over always-rewrite?
4. **Design and implement** a lightweight pre-retrieval gate using cheap query/history features.
5. **Evaluate the gate** against simple controls (always-keep, always-rewrite, CONCAT, heuristic gate, matched-rate random gate, oracle upper bound).
6. **Measure the quality–cost trade-off**: retrieval metrics alongside rewrite rate, LLM calls, tokens, and latency.

### Out of scope for the core implementation (optional extensions)
- CLARIFY action (flagging ambiguous turns and simulating user reply)
- FUSE as a gate output (currently only CONCAT as fixed baseline)
- Two-stage retrieval-informed gate (Stage 2 using retrieval confidence)
- Conformal risk control on the gate threshold
- TopiOCQA secondary dataset
- Dense / hybrid retrievers
- Answer-quality evaluation (F1, faithfulness)
- Ambiguity stress tests on corrupted history

> **[IMPORTANT]** The original synopsis describes a more ambitious system (KEEP/REWRITE/FUSE/CLARIFY, two-stage cascade, conformal risk control). The revised direction (Before vs. Now document) narrowed scope to: KEEP/CONCAT/REWRITE formulations + heuristic/random/learned gate + oracle headroom + cost analysis. The optional extensions above remain in the synopsis but are not required for Phases 4–7.

---

## 4. Research Methodology

### Formulation strategies
| Strategy | Description | LLM call? |
|---|---|---|
| **KEEP** | Return the original question unchanged (trimmed) | No |
| **CONCAT** | Prepend up to 3 prior questions to the current question | No |
| **REWRITE** | Generate a standalone question using Groq + `openai/gpt-oss-20b` | Yes |

### Evaluation pipeline
1. Generate KEEP, CONCAT, and REWRITE formulations for every eligible turn.
2. Retrieve top-100 passages using BM25 (SQLite FTS5) for each formulation.
3. Compute MRR@10, Recall@10, Recall@100, nDCG@10 per turn.
4. Label each turn as **helpful / tied / harmful** (REWRITE vs. KEEP, using configured tie margin).
5. Compute oracle scores and headroom.
6. Train and evaluate gate models; compare with controls.
7. Report quality alongside efficiency metrics.

### Experimental controls
| Method | Description |
|---|---|
| Always KEEP | Never rewrite — lower bound |
| Always REWRITE | Always rewrite — cost-blind upper baseline |
| CONCAT | Fixed history concatenation |
| Heuristic gate | Rule-based (pronoun/ellipsis/length) |
| Random gate (matched rate) | Same rewrite rate as learned gate, random selection |
| **Learned gate** | Main contribution: lightweight classifier on pre-retrieval features |
| Oracle | Analytical upper bound; not deployable |

### Critical research constraints
- **Never** use QReCC `Truth_rewrite` / gold reference rewrites as generated REWRITE outputs or as gate features.
- **Never** use retrieval scores, test labels, or oracle outcomes as gate inputs (pre-retrieval features only).
- **Conversation-level splitting** to prevent leakage across turns from the same conversation.
- **Fixed corpus/index/retriever/relevance labels** across all formulations for fair comparison.
- **Report failures honestly**: document failed rewrites, missing judgments, denominators.

---

## 5. System Architecture and Component Responsibilities

```
                    ┌─────────────────────────────────────────┐
                    │         ConversationTurn (schema.py)    │
                    │   question, history, relevant_ids, ...  │
                    └──────────────┬──────────────────────────┘
                                   │
             ┌─────────────────────┼─────────────────────────┐
             │                     │                          │
             ▼                     ▼                          ▼
     formulate_keep()      formulate_concat()      formulate_with_cache()
     (formulate.py)        (formulate.py)           ↓ RewriteCache
                                                  (cache.py → Groq API)

             └──────────────┬──────────────────────────────────┘
                            │ formulated query string
                            ▼
                  BM25Retriever.retrieve()
                  (retrieve.py → SQLite FTS5)
                            │
                            ▼ [(passage_id, score), ...]
                  compute_metrics()
                  (metrics.py)  → MRR, Recall, nDCG
                            │
                            ▼ {unique_id: score}
                  label_all_outcomes()      oracle_score_per_turn()
                  (outcomes.py)             (oracle.py)
                            │                      │
                            ▼                      ▼
                  helpful/tied/harmful      oracle headroom
                            │
                            ▼
    ┌──────────────────────────────────────────────────────────┐
    │                    Gating (Phase 5)                       │
    │  features.py  →  heuristic.py / random_gate.py          │
    │                  train_gate.py (learned classifier)       │
    └──────────────────────────────────────────────────────────┘
```

---

## 6. Repository Structure

```
selective-query-rewriting/
├── configs/
│   └── experiment.yaml          # Experiment hyperparameters and paths
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schema.py            # ConversationTurn dataclass (STABLE)
│   │   └── load_qrecc.py        # Dataset loader (JSON + JSONL formats)
│   ├── rewriting/
│   │   ├── __init__.py
│   │   ├── formulate.py         # KEEP, CONCAT, REWRITE formulation
│   │   └── cache.py             # Groq rewrite generation + JSONL cache
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── build_index.py       # SQLite FTS5 BM25 index builder
│   │   └── retrieve.py          # BM25Retriever class
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # MRR, Recall@k, nDCG@k (binary relevance)
│   │   ├── outcomes.py          # helpful/tied/harmful labeling
│   │   └── oracle.py            # Oracle headroom calculation
│   └── gating/
│       ├── __init__.py
│       ├── features.py          # STUB – pre-retrieval feature extraction
│       ├── heuristic.py         # STUB – rule-based gate
│       ├── random_gate.py       # STUB – matched-rate random baseline
│       └── train_gate.py        # STUB – learned classifier
├── scripts/
│   ├── run_formulation_pilot.py # Generate KEEP/CONCAT/REWRITE for pilot
│   ├── test_rewrite_pipeline.py # Integration smoke test (needs GROQ_API_KEY)
│   └── diagnose_rewrite.py      # Debug single rewrite call
├── tests/
│   ├── __init__.py
│   ├── test_schema.py           # ConversationTurn tests
│   ├── test_load_qrecc.py       # Dataset loader tests (synthetic data)
│   ├── test_formulate.py        # KEEP/CONCAT/REWRITE formulation tests
│   ├── test_formulate_with_cache.py  # Cache integration tests (mocked)
│   ├── test_rewrite_cache.py    # RewriteCache unit tests (mocked)
│   ├── test_metrics.py          # MRR/Recall/nDCG tests
│   ├── test_outcomes.py         # Help/tie/hurt labeling tests
│   └── test_oracle.py           # Oracle headroom tests
├── docs/
│   └── experiments/
│       └── phase3_pilot.md      # Phase 3 pilot results record
├── check_references.py          # One-off corpus coverage checker (requires data/)
├── check_turn_coverage.py       # Eligible turn coverage analysis (requires data/)
├── debug_reference.py           # One-off reference debug script (requires data/)
├── .gitignore                   # Excludes data/, outputs/, .rewrite_cache/, etc.
├── pyproject.toml               # Project metadata + pytest config
├── requirements.txt             # Python dependencies (pinned)
├── README.md                    # User-facing setup and overview
├── DATASET.md                   # Dataset documentation (QReCC)
└── PROJECT_PROGRESS.md          # ← THIS FILE (authoritative progress record)

# NOT committed to git (generated / local-only):
data/
├── qrecc/
│   ├── train.jsonl              # ~63,501 turns, 10,823 conversations
│   └── test.jsonl               # ~16,451 turns, 2,775 conversations
├── passages/raw/
│   ├── train-000.parquet        # ~3M source rows across 3 shards
│   ├── train-001.parquet
│   └── train-002.parquet
├── processed/
│   ├── train_eligible_turns.jsonl  # 27,594 eligible training turns
│   └── test_eligible_turns.jsonl   # ~7,672 eligible test turns
└── index/
    └── bm25_fts5_dedup_full.sqlite  # ~2,981,202 unique passages

outputs/
├── results/
│   └── formulation_pilot.jsonl  # 136 turns × {keep,concat,rewrite} formulations
└── plots/                       # (not yet created)

.rewrite_cache/
└── rewrites.jsonl               # Cached Groq rewrites (~47 KB, 134 entries)
```

---

## 7. Technology Stack and Dependencies

| Component | Library / Tool | Version constraint |
|---|---|---|
| Language | Python | ≥ 3.10 |
| Data handling | pandas, numpy | ≥ 2.0, ≥ 1.26 |
| Retrieval | SQLite FTS5 (built-in) | – |
| Parquet I/O | pyarrow | ≥ 14.0 |
| LLM rewriting | openai SDK (Groq endpoint) | unlocked (openai pkg) |
| ML gate | scikit-learn | ≥ 1.4, < 2.0 |
| Configuration | pyyaml | ≥ 6.0 |
| Progress bars | tqdm | ≥ 4.66 |
| BM25 (unused) | rank-bm25 | ≥ 0.2.2 |
| Testing | pytest | ≥ 8.0 |
| Plots (planned) | matplotlib, seaborn | (commented out in requirements.txt) |

**API:** Groq (`https://api.groq.com/openai/v1`) — requires `GROQ_API_KEY` environment variable.
**Model used for rewriting:** `openai/gpt-oss-20b`

> [!NOTE]
> `rank-bm25` is listed in `requirements.txt` but is **not used** — retrieval is done via SQLite FTS5. It should be removed or clearly marked optional to avoid confusion.
>
> `openai` is used (imported in `cache.py`) but was commented out in `requirements.txt` as "optional/future". This is a **bug**: `openai` must be in the active dependencies.

---

## 8. Phase-by-Phase Implementation Plan

| Phase | Title | Status |
|---|---|---|
| 0 | Repository setup and experiment contract | ✅ Complete |
| 1 | Dataset loading, schema, and coverage analysis | ✅ Complete |
| 2 | BM25 retrieval foundation and index construction | ✅ Complete |
| 3 | Query formulation baselines and pilot generation | ✅ Complete (pilot done) |
| 4 | Per-query retrieval evaluation, outcome labels, oracle | 🔲 Not started |
| 5 | Pre-retrieval features, gating models, controls | 🔲 Not started |
| 6 | Efficiency measurement and error analysis | 🔲 Not started |
| 7 | Frozen test protocol, final results, paper reporting | 🔲 Not started |

---

## 9. Phase 0–3: Implementation Details and Verified Status

### Phase 0 — Repository and Experiment Contract ✅

**What was intended:** Set up a clean Python project with configuration, shared schema, and test harness.

**What is implemented:**
- `pyproject.toml` — project metadata + pytest configuration (`pythonpath = ["."]`, `testpaths = ["tests"]`)
- `configs/experiment.yaml` — all hyperparameters in one file (run_name, dataset paths, formulation flags, retrieval top_k, evaluation metrics, rewrite settings, gating seed, output dirs)
- `.gitignore` — correctly excludes `data/`, `outputs/`, `.rewrite_cache/`, venv, caches

**Unresolved decisions flagged in config:**
- `primary_metric: "ndcg@10"` — marked `[UNRESOLVED]`, team agreement needed before oracle/gate work
- `tie_margin: 0.01` — marked `[UNRESOLVED]`, agreement needed before generating outcome labels
- `target_rewrite_rate: null` — to be set after KEEP/REWRITE analysis

**Status: Complete. Decisions above must be confirmed before Phase 4 runs.**

---

### Phase 1 — Dataset Loading and Coverage Analysis ✅

**What was intended:** Load QReCC, define shared schema, check passage-reference coverage.

**What is implemented:**
- [`src/data/schema.py`](file:///c:/Users/mayur/OneDrive/Desktop/selective-query-rewriting/src/data/schema.py) — `ConversationTurn` dataclass with `conversation_id`, `turn_id`, `question`, `relevant_passage_ids`, `conversation_history`, `rewritten_question`, `split`; `unique_id` property returns `{conversation_id}_{turn_id}`
- [`src/data/load_qrecc.py`](file:///c:/Users/mayur/OneDrive/Desktop/selective-query-rewriting/src/data/load_qrecc.py) — handles official JSON array format AND SCAI JSONL format; passage ID normalization (Markdown link extraction + backslash unescaping); `load_turns()`, `load_train_and_test()`, `print_dataset_stats()`
- [`check_turn_coverage.py`](file:///c:/Users/mayur/OneDrive/Desktop/selective-query-rewriting/check_turn_coverage.py) — generates `data/processed/{train,test}_eligible_turns.jsonl` and coverage reports (requires local data files)
- [`check_references.py`](file:///c:/Users/mayur/OneDrive/Desktop/selective-query-rewriting/check_references.py) — cross-checks reference IDs against corpus Parquet files

**Dataset statistics (from DATASET.md, collected during prior run):**

| Split | Turns | Conversations | Turns with refs | Turns with all refs available |
|---|---:|---:|---:|---:|
| Train | 63,501 | 10,823 | 29,596 | 27,594 |
| Test | 16,451 | 2,775 | 8,209 | 7,672 |

Reference-entry coverage: **89.92%** (train), **89.30%** (test) against the 3-shard local corpus.

**Tests:** `tests/test_schema.py`, `tests/test_load_qrecc.py` — synthetic data, no real dataset required.

**Status: Complete.** Coverage numbers are specific to the local 3-shard corpus; recalculate if corpus changes.

---

### Phase 2 — BM25 Retrieval Foundation ✅

**What was intended:** Build a disk-backed BM25 index over the passage corpus; implement retrieval.

**What is implemented:**
- [`src/retrieval/build_index.py`](file:///c:/Users/mayur/OneDrive/Desktop/selective-query-rewriting/src/retrieval/build_index.py) — SQLite FTS5 with `porter unicode61` tokenizer; deterministic duplicate-handling (first-valid-occurrence by normalized ID); records metadata in `index_metadata` table; CLI with `--corpus-dir`, `--index-path`, `--max-passages`, `--overwrite`
- [`src/retrieval/retrieve.py`](file:///c:/Users/mayur/OneDrive/Desktop/selective-query-rewriting/src/retrieval/retrieve.py) — `BM25Retriever` class (context manager); `retrieve(query, top_k) → [(passage_id, score)]`; scores are negated SQLite BM25 (higher = better)

**Full index stats (from prior build):**
- 2,981,202 unique passages indexed
- 17,085 empty/invalid rows skipped
- 1,713 duplicate-ID rows skipped

**Status: Complete.** Index exists at `data/index/bm25_fts5_dedup_full.sqlite` (not in git).

> [!IMPORTANT]
> Before Phase 4 evaluation, run a passage-ID compatibility smoke test: retrieve a few pilot turns, print relevant IDs and top retrieved IDs, verify they share the same format. Widespread zero-hit matches indicate an ID format mismatch.

---

### Phase 3 — Query Formulation Baselines and Pilot ✅

**What was intended:** Generate KEEP, CONCAT, and REWRITE formulations for pilot conversations; validate the pipeline.

**What is implemented:**
- [`src/rewriting/formulate.py`](file:///c:/Users/mayur/OneDrive/Desktop/selective-query-rewriting/src/rewriting/formulate.py) — `formulate_keep()`, `formulate_concat(max_history_turns=3)`, `formulate_rewrite(cached_rewrite)`, `formulate()` dispatch, `formulate_with_cache()`
- [`src/rewriting/cache.py`](file:///c:/Users/mayur/OneDrive/Desktop/selective-query-rewriting/src/rewriting/cache.py) — `RewriteCache`: SHA-256 cache key over full input; Groq API call via OpenAI SDK; retry on `finish_reason="length"` with 4× token budget; JSONL append-mode persistence; empty rewrites not cached
- [`scripts/run_formulation_pilot.py`](file:///c:/Users/mayur/OneDrive/Desktop/selective-query-rewriting/scripts/run_formulation_pilot.py) — reads config + eligible turns, generates all three formulations, writes `outputs/results/formulation_pilot.jsonl`
- [`docs/experiments/phase3_pilot.md`](file:///c:/Users/mayur/OneDrive/Desktop/selective-query-rewriting/docs/experiments/phase3_pilot.md) — pilot run record

**Pilot results (first 50 training conversations):**

| Metric | Value |
|---|---|
| Conversations | 50 |
| Total turns loaded | 389 |
| Eligible turns | 136 |
| Successful rewrites | 134 |
| Failed rewrites | 2 |
| Rewrite success rate | 98.5% |
| KEEP missing | 0 |
| CONCAT missing | 0 |
| Tests passing | 108 |

**Failed rewrites:**
- `21_3`: "When does that change?" — retry exhausted 1,024-token budget with reasoning, no usable text
- `38_3`: "What happened?" — same failure mode

**Failure policy (must be applied consistently):**
- Do NOT invent replacement rewrites.
- KEEP and CONCAT are available for all 136 turns.
- REWRITE evaluation must use a matched subset (134 turns with successful rewrites).
- Report denominators explicitly: all-turn results for KEEP/CONCAT; 134-turn subset for REWRITE comparisons.
- Keep generation failures distinct from retrieval failures.

**Status: Pilot complete.** Final sign-off requires applying the failure-handling policy consistently in Phase 4.

---

## 10. Phases 4–7: Requirements and Planned Implementation

> [!WARNING]
> Do NOT begin Phase 4 implementation without confirming:
> 1. Primary metric (currently `ndcg@10` in config — needs team agreement)
> 2. Tie margin (currently `0.01` — needs team agreement)
> 3. Failure policy (documented above — needs team sign-off)
> 4. Passage ID compatibility (run smoke test first)

### Phase 4 — Per-Query Retrieval Evaluation, Outcome Labels, Oracle 🔲

**Goal:** Evaluate each formulation on the fixed BM25 setup; compute retrieval metrics per turn; classify rewrites as helpful/tied/harmful; quantify oracle headroom.

**Inputs available:**
- `outputs/results/formulation_pilot.jsonl` — 136 turns × {keep, concat, rewrite} formulations
- `data/index/bm25_fts5_dedup_full.sqlite` — full BM25 index
- `src/retrieval/retrieve.py` — BM25Retriever
- `src/evaluation/metrics.py` — compute_metrics()
- `src/evaluation/outcomes.py` — label_all_outcomes()
- `src/evaluation/oracle.py` — oracle_score_per_turn()

**Steps:**
1. Verify passage-ID compatibility on a tiny smoke test (3–5 pilot turns).
2. Write `scripts/run_retrieval_eval.py` (or equivalent): for each row in pilot JSONL, retrieve top-100 for keep and concat (all 136 turns) and for rewrite (134 turns with rewrite_status=success). Save ranked IDs and scores.
3. Compute MRR@10, Recall@10, Recall@100, nDCG@10 per turn using `compute_metrics()`.
4. Write per-query output to `outputs/results/retrieval_per_query.jsonl` — fields: `turn_id`, `conversation_id`, `action`, `formulated_query`, `relevant_ids`, `retrieved_ids`, `metrics`, `run_id`, `rewrite_status`.
5. Compute macro-average metrics with denominators; write `outputs/results/formulation_summary.json`.
6. Call `label_all_outcomes(keep_scores, rewrite_scores, tie_margin)` for turns with both keep and rewrite scores. Save to `outputs/results/outcome_labels.jsonl`.
7. Call `oracle_score_per_turn(score_sets)` and `oracle_headroom_summary()`. Save to `outputs/results/oracle_summary.json`.
8. Add/extend tests; run `python -m pytest -q`.

**Deliverables:**
- `outputs/results/retrieval_per_query.jsonl`
- `outputs/results/formulation_summary.json`
- `outputs/results/outcome_labels.jsonl`
- `outputs/results/oracle_summary.json`
- Updated tests passing

**Acceptance criteria:**
- ID compatibility verified on smoke test
- Empty-judgment policy documented and applied
- Denominators explicit in all summary files
- Failure turns tracked separately from retrieval failures
- All configured metrics computed and reported

---

### Phase 5 — Pre-Retrieval Features, Gate Models, Controls 🔲

**Goal:** Implement and evaluate the gate. Determine whether inexpensive features can predict when rewriting helps.

**Planned feature set (all pre-retrieval — no retrieval results):**
- `query_length` — token count of current question
- `has_pronoun` — binary; detects 'he', 'she', 'it', 'they', 'that', 'this', 'those', 'these'
- `has_ellipsis` — binary; detects ellipsis/context-dependent patterns
- `turn_position` — turn_id value
- `lexical_overlap` — Jaccard similarity between query tokens and history tokens

> [!IMPORTANT]
> Feature set must be finalized after reviewing the help/tie/hurt outcome distribution from Phase 4. Do not implement the gate before Phase 4 outcomes are available.

**Steps:**
1. Implement [`src/gating/features.py`](file:///c:/Users/mayur/OneDrive/Desktop/selective-query-rewriting/src/gating/features.py) — `extract_features(turn: ConversationTurn) → dict`
2. Add unit tests for edge cases (first turn, empty history, all pronouns, etc.)
3. Implement [`src/gating/heuristic.py`](file:///c:/Users/mayur/OneDrive/Desktop/selective-query-rewriting/src/gating/heuristic.py) — rule-based gate (pronoun/ellipsis/length); tune thresholds on train/dev only
4. Implement [`src/gating/random_gate.py`](file:///c:/Users/mayur/OneDrive/Desktop/selective-query-rewriting/src/gating/random_gate.py) — random gate with `seed=42` (from config); matched to learned gate's rewrite rate
5. Create conversation-level train/dev/test split (suggested ~70/15/15 by conversation ID)
6. Implement [`src/gating/train_gate.py`](file:///c:/Users/mayur/OneDrive/Desktop/selective-query-rewriting/src/gating/train_gate.py) — logistic regression or gradient boosting on features; labels from training-split Phase 4 outcomes; tune on dev; evaluate on held-out test
7. Write comparison table: always-KEEP, always-REWRITE, CONCAT, heuristic, random (matched), learned, oracle
8. Plot quality vs. rewrite rate

**Deliverables:**
- Implemented features, heuristic, random, and learned gate modules
- Tests for each
- `outputs/results/gate_predictions.jsonl`
- Comparison table and quality-vs-rate plot

---

### Phase 6 — Efficiency Measurement and Error Analysis 🔲

**Goal:** Quantify the cost of observed retrieval quality; explain why rewrites help, tie, or hurt.

**Efficiency metrics to collect:**
- Rewrite rate (REWRITE decisions / eligible turns) with numerator/denominator
- API call count; failed calls; retries; cache hits vs. misses
- Prompt tokens; completion tokens (from Groq response usage)
- Latency per call and aggregate (mean; p95 if sample permits)
- Estimated monetary cost (only if dated public pricing available — state assumptions)

**Error taxonomy (to code manually on a reproducible sample):**
- Lost key terms (names, dates, entities disappear)
- Unsupported specificity (rewrite adds unjustified details)
- Wrong entity/referent (pronoun resolved to wrong entity)
- Meaning drift (intent or answer type changes)
- No useful change (paraphrase; no retrieval improvement)
- Generation failure (empty response, timeout, exhausted retries)
- Retrieval harm despite plausible wording

**Deliverables:**
- `outputs/results/efficiency_log.jsonl`
- Efficiency summary table
- Error taxonomy with counts and examples (including `21_3` and `38_3`)

---

### Phase 7 — Frozen Test Protocol, Final Results, Paper Reporting 🔲

**Goal:** Freeze all decisions, run held-out test set, produce paper-ready tables and plots.

**Before looking at test results, freeze:**
- Dataset version/split manifest
- Corpus/index metadata
- Formulation functions and prompt/model/temperature
- Cache policy
- Primary metric and tie margin
- Gate feature schema and training procedure
- Thresholds and seeds
- Failure policy

**Required paper-ready outputs:**
- Formulation metrics table (KEEP, CONCAT, REWRITE — all denominators explicit)
- Helpful/tied/harmful distribution table
- Oracle headroom table
- Gate comparison table (all controls included)
- Quality vs. rewrite rate plot
- Efficiency summary
- Error taxonomy table
- `outputs/results/final_summary.json` and `outputs/plots/`

**Run manifest fields:** commit hash, Python/library versions, model ID, prompt version, seed, index metadata, timestamp, output paths.

---

## 11. Important Design Decisions and Rationale

| Decision | Rationale |
|---|---|
| SQLite FTS5 for BM25 (not rank-bm25) | Disk-backed, handles millions of passages, deterministic, no dependency on in-memory loading |
| Groq API via OpenAI SDK compatibility | Cost-effective; allows caching; SDK abstracts the HTTP layer |
| SHA-256 cache key over full input | Deterministic, collision-resistant; invalidates cache when model/prompt changes |
| KEEP as primary baseline (not CONCAT) | KEEP is cost-free and preserves exact user intent; CONCAT is an intermediate |
| Binary relevance in nDCG | QReCC passage labels are binary (present/absent); graded relevance not confirmed |
| Conversation-level split (not turn-level) | Turns within a conversation share entities/context — turn-level split would leak |
| Pilot: first 50 conversations only | Limits API cost during development; Phase 4 can expand to all eligible turns |
| No CLARIFY in core scope | Requires simulating user replies (additional LLM cost); deprioritized for core experiment |

---

## 12. Dataset and Data-Processing Status

See [`DATASET.md`](DATASET.md) for full details. Summary:

| Item | Status |
|---|---|
| QReCC dataset present locally | ✅ (train.jsonl, test.jsonl — not committed) |
| Passage corpus present locally | ✅ (3 Parquet shards — not committed) |
| Eligible turns generated | ✅ data/processed/train_eligible_turns.jsonl (27,594 turns) |
| BM25 index built | ✅ data/index/bm25_fts5_dedup_full.sqlite (2,981,202 passages) |
| Dataset terms/license verified | ⚠️ CC BY-SA 3.0 reported; must be confirmed against official record before publication |
| Dataset exact version documented | ⚠️ Pending — record Zenodo record version and download date |

---

## 13. Query Formulation and Rewriting Approaches

### KEEP
Returns `turn.question.strip()`. No API calls. Deterministic.

### CONCAT
Returns `" ".join(recent_history[-max_history_turns:]) + " " + question`. Default `max_history_turns=3`. Falls back to KEEP when history is empty or all whitespace.

### REWRITE
Uses `RewriteCache.get_rewrite()`:
- System prompt: *"Rewrite the current question as a standalone question using the conversation history. Preserve the original meaning. Return only the final question, with no explanation. If the history does not identify a referent, do not invent one."*
- Model: `openai/gpt-oss-20b` via Groq
- `max_tokens=256`; `temperature=0.0`
- Retry once at `max(4×256, 1024)=1024` tokens if response is empty due to token exhaustion
- Caches successful rewrites by SHA-256 key; empty rewrites raise `ValueError` and are not cached
- `prompt_version="standalone-v1"` (included in cache key for cache invalidation)

---

## 14. Retrieval and Evaluation Methodology

### BM25 Retrieval
- Index: SQLite FTS5, `porter unicode61` tokenizer
- Query tokenization: `re.findall(r"\w+", query.lower())` → tokens quoted and joined with OR
- Returns: `[(passage_id, -raw_bm25_score), ...]` (higher score = better match)
- Default `top_k=100` for Recall@100; top-10 subset for MRR@10 and nDCG@10

### Metrics (`compute_metrics()`)
- **MRR@k** — reciprocal rank of first relevant document in top-k
- **Recall@k** — fraction of relevant documents found in top-k
- **nDCG@k** — normalized discounted cumulative gain (binary relevance)
- Returns 0.0 when `relevant_ids` is empty (does NOT treat as all-relevant)

### Outcome labeling
- **Helpful**: `rewrite_score - keep_score > tie_margin`
- **Tied**: `|rewrite_score - keep_score| <= tie_margin`
- **Harmful**: `keep_score - rewrite_score > tie_margin`
- Current config: `tie_margin=0.01` (**[UNRESOLVED] — team agreement required**)

### Oracle
- For each turn: `oracle_score = max(score over all included formulations)`
- Oracle headroom: `oracle_mean - baseline_mean` (baseline = always-rewrite or always-keep)
- Oracle is an analysis upper bound; it is NOT a deployable system

---

## 15. Experimental Configurations and Reproducibility Requirements

| Setting | Current value | Status |
|---|---|---|
| Primary metric | `ndcg@10` | **[UNRESOLVED]** — needs team agreement |
| Tie margin | `0.01` | **[UNRESOLVED]** — needs team agreement |
| Top-k retrieval | `100` | Agreed |
| Rewrite model | `openai/gpt-oss-20b` | Agreed |
| Temperature | `0.0` | Agreed |
| Max tokens (rewrite) | `256` (retry: `1024`) | Agreed |
| Prompt version | `standalone-v1` | Agreed |
| Random seed (gating) | `42` | Agreed |
| Pilot conversations | `50` | Agreed |
| CONCAT max history | `3` | Agreed |
| Split unit | Conversation-level | Agreed |

**Reproducibility checklist:**
- [ ] Record Groq model ID exactly (model alias may change)
- [ ] Record Python and dependency versions in run manifest
- [ ] Record index metadata (from `BM25Retriever.metadata()`)
- [ ] Confirm primary metric and tie margin before generating outcome labels
- [ ] Confirm conversation-level split before fitting any gate model
- [ ] Do not use test-split outcomes during gate training or threshold tuning

---

## 16. Tests Executed and Their Actual Outcomes

| Test file | Tests | Last known outcome |
|---|---|---|
| `test_schema.py` | 10 | Passed (part of 108-test run) |
| `test_load_qrecc.py` | ~30 | Passed (synthetic data) |
| `test_formulate.py` | ~20 | Passed |
| `test_formulate_with_cache.py` | 4 | Passed (mocked cache) |
| `test_rewrite_cache.py` | 4 | Passed (mocked Groq client) |
| `test_metrics.py` | ~18 | Passed |
| `test_outcomes.py` | 10 | Passed |
| `test_oracle.py` | 8 | Passed |
| **Total** | **108** | **All passed (Phase 3 run)** |

> Tests do NOT require real dataset files, real API keys, or the BM25 index. All external calls are mocked.

**Run command (from repo root with venv active):**
```bash
python -m pytest -q
```

---

## 17. Known Issues, Limitations, and Risks

### Code issues
| Issue | Location | Severity |
|---|---|---|
| `openai` package used but listed as "optional" in requirements.txt | `requirements.txt` vs `cache.py` | High — breaks fresh install |
| `rank-bm25` listed as dependency but never used | `requirements.txt` | Low — unnecessary dependency |
| `features.py`, `heuristic.py`, `random_gate.py`, `train_gate.py` are stubs that raise `NotImplementedError` at module import | `src/gating/` | Expected — Phase 5 work |
| Config paths (`data/raw/qrecc`, `data/index/bm25_qrecc`) are stale in experiment.yaml | `configs/experiment.yaml` | Medium — actual paths differ; verify before running scripts |
| Passage-ID format compatibility unverified | Phase 4 prerequisite | High — must check before retrieval evaluation |

### Research issues
| Issue | Risk level |
|---|---|
| Primary metric and tie margin unconfirmed | High — must agree before outcome labels |
| Conversation-level split procedure not yet implemented | High — Phase 5 blocker |
| Oracle headroom may be small, leaving little room for gate improvement | Medium — affects novelty framing |
| 89% reference coverage means ~11% of reference passages are missing from local corpus | Medium — affects absolute metric values |
| Pilot of 136 turns may be too small for statistically stable conclusions | Medium — final test uses full eligible set |
| Binary relevance assumption in nDCG may understate quality differences | Low — QReCC labels are binary |

### Documentation issues
| Issue | Action needed |
|---|---|
| Three root-level one-off scripts (`check_references.py`, `check_turn_coverage.py`, `debug_reference.py`) could be moved to `scripts/` | Low priority — no functional impact |
| Dataset license/version not formally recorded | Before publication |

---

## 18. Commands Required to Set Up, Test, and Run the Project

### Initial setup
```bash
# 1. Clone
git clone https://github.com/Selective-Query-Rewriting/selective-query-rewriting.git
cd selective-query-rewriting

# 2. Create venv (Python 3.10+)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install openai   # Must install manually until requirements.txt is fixed

# 4. Set API key
set GROQ_API_KEY=your_key_here   # Windows
export GROQ_API_KEY=your_key_here  # macOS/Linux
```

### Dataset and index (if not already present)
```bash
# Place dataset files at:
#   data/qrecc/train.jsonl
#   data/qrecc/test.jsonl
#   data/passages/raw/train-000.parquet, train-001.parquet, train-002.parquet

# Build smoke-test index (10,000 passages)
python src/retrieval/build_index.py --index-path data/index/bm25_fts5_smoke.sqlite

# Build full index (~2.98M passages — takes significant time/disk)
python src/retrieval/build_index.py --max-passages 0

# Generate eligible turn files
python check_turn_coverage.py
```

### Run tests
```bash
python -m pytest -q        # All 108 tests (no external dependencies required)
python -m pytest -v        # Verbose output
```

### Run pilot (Phase 3)
```bash
# Generates outputs/results/formulation_pilot.jsonl
python scripts/run_formulation_pilot.py
```

---

## 19. Git Collaboration and Branching Workflow

- **Main branch:** `main` — stable, up to date with origin
- **No feature branches currently exist**
- **Do not commit:** `data/`, `outputs/`, `.rewrite_cache/`, `.venv/`, `*.sqlite`, `*.db`, API keys
- **Recommended:** Create a feature branch before starting each phase (`git checkout -b phase-4-retrieval-eval`)
- **Commit small, focused changes** that can be reviewed independently
- **Do not force-push or rewrite shared history** without explicit authorization

---

## 20. Completed Milestones and Evidence

| Milestone | Evidence |
|---|---|
| Repository structure established | `pyproject.toml`, `configs/experiment.yaml`, directory layout |
| `ConversationTurn` schema stabilized | `src/data/schema.py`; 10 passing tests in `test_schema.py` |
| QReCC loading (JSON + JSONL) | `src/data/load_qrecc.py`; ~30 passing tests; synthetic-data only |
| Passage-reference coverage measured | `DATASET.md` §8; `check_turn_coverage.py` run |
| BM25 index built (2.98M passages) | `data/index/bm25_fts5_dedup_full.sqlite`; metadata in DB |
| BM25 retriever implemented | `src/retrieval/retrieve.py`; unit tests pass |
| KEEP, CONCAT, REWRITE formulations | `src/rewriting/formulate.py`; 20+ passing tests |
| Groq rewrite cache implemented | `src/rewriting/cache.py`; 4 passing tests (mocked client) |
| Metrics (MRR, Recall, nDCG) | `src/evaluation/metrics.py`; 18+ passing tests |
| Help/tie/hurt outcome labeling | `src/evaluation/outcomes.py`; 10 passing tests |
| Oracle headroom calculation | `src/evaluation/oracle.py`; 8 passing tests |
| Phase 3 pilot run (136 turns) | `outputs/results/formulation_pilot.jsonl`; 134/136 rewrites; 108 tests |

---

## 21. Remaining Tasks in Priority Order

| Priority | Task | Phase | Blocker? |
|---|---|---|---|
| 1 | Confirm primary metric and tie margin with team | Pre-Phase-4 | Yes — blocks outcome labeling |
| 2 | Fix `openai` dependency in `requirements.txt` | Immediate | Yes — breaks fresh install |
| 3 | Run passage-ID compatibility smoke test | Phase 4 | Yes — blocks retrieval eval |
| 4 | Implement retrieval evaluation script | Phase 4 | No (after #1, #3) |
| 5 | Compute per-query metrics and save to JSONL | Phase 4 | No |
| 6 | Generate outcome labels and oracle headroom | Phase 4 | No |
| 7 | Add Phase 4 tests | Phase 4 | No |
| 8 | Define conversation-level train/dev/test split | Phase 5 | Yes — blocks gate training |
| 9 | Implement `features.py` | Phase 5 | After Phase 4 outcomes |
| 10 | Implement `heuristic.py` | Phase 5 | After #9 |
| 11 | Implement `random_gate.py` | Phase 5 | After learned gate rate known |
| 12 | Implement `train_gate.py` | Phase 5 | After #9, #10 |
| 13 | Build comparison table (all baselines) | Phase 5 | After gate eval |
| 14 | Efficiency log collection | Phase 6 | After Phase 4 |
| 15 | Error taxonomy analysis | Phase 6 | After Phase 4 |
| 16 | Frozen test protocol + final evaluation | Phase 7 | After all above |
| 17 | Paper-ready tables, plots, and writeup | Phase 7 | After #16 |

---

## 22. Open Questions and Decisions Requiring Human Input

| # | Question | Impact | Owner |
|---|---|---|---|
| OQ-1 | **Primary metric**: team must confirm `ndcg@10` vs. another metric before outcome labels are generated | High — affects all downstream outcomes and gate labels | Team agreement |
| OQ-2 | **Tie margin**: team must confirm `0.01` (in config) before outcome labels are generated | High — affects helpful/tied/harmful distribution | Team agreement |
| OQ-3 | **Pilot vs. full eligible set**: will Phase 4 evaluate only the 136-turn pilot or all 27,594 eligible training turns? | Scope — pilot is faster but smaller | Team decision |
| OQ-4 | **Conversation-level split ratios**: what are the train/dev/test proportions and how are conversations assigned? | Gate training | Team decision |
| OQ-5 | **Failure handling in averages**: when REWRITE fails, should the failed turn be excluded from averages or scored as 0? | Metric integrity | Team decision |
| OQ-6 | **CONCAT as gate output vs. fixed baseline**: should the gate choose between KEEP and REWRITE only, or also CONCAT? | Gate design | Team decision |
| OQ-7 | **Full corpus scope**: the local corpus has 3 Parquet shards (~3M passages); the original QReCC corpus is 54M passages. Will the evaluation use the local 3-shard corpus? | Comparability with published results | Team decision |
| OQ-8 | **`openai` package version**: what version is installed in the working venv? | Reproducibility | Check with `pip show openai` |
| OQ-9 | **Dataset license confirmation**: verify CC BY-SA 3.0 applies to the local JSONL files before publication | Legal | Check Zenodo record |
| OQ-10 | **Removing stale `rank-bm25` dependency**: safe to remove from requirements.txt? | Cleanliness | Confirm no other code uses it |

---

## 23. Changelog

| Date | Milestone |
|---|---|
| 2026-10-29 | **Stage A audit complete**: full codebase inspection, all three attached documents read, `PROJECT_PROGRESS.md` created from scratch |
| (prior) | Phase 3 pilot complete: 136 turns, 134/136 rewrites, 108 tests passing |
| (prior) | Phase 2 complete: BM25 index built (2,981,202 passages) |
| (prior) | Phase 1 complete: QReCC loader, schema, coverage analysis |
| (prior) | Phase 0 complete: repository, config, .gitignore |

---

## Documentation Consolidation Notes

Current Markdown files in the repository:

| File | Purpose | Action |
|---|---|---|
| `README.md` | User-facing setup and overview | **Keep** — essential |
| `DATASET.md` | QReCC dataset documentation | **Keep** — detailed dataset record |
| `docs/experiments/phase3_pilot.md` | Phase 3 pilot run record | **Keep** — historical experimental record |
| `PROJECT_PROGRESS.md` | **This file** — authoritative project record | **Keep** — the canonical source |

No files need to be deleted. The four Markdown files serve distinct purposes and are non-redundant.

---

*End of PROJECT_PROGRESS.md*
