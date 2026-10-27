# Selective Query Rewriting for Conversational Retrieval

> **Provisional research title:** *When Is Rewriting Worth Its Cost? Oracle Headroom and Cost-Aware Gating for Conversational Retrieval*

## Overview

Conversational search often involves follow-up questions that depend on earlier turns. For example, a question such as "Who directed it?" may be difficult to retrieve against when searched without its conversational context.

Query rewriting can convert such questions into standalone queries, but rewriting every question introduces additional cost and may sometimes hurt retrieval quality.

This project investigates whether a lightweight, pre-retrieval gate can determine **when rewriting is worth its cost**, balancing retrieval effectiveness against rewriting overhead.

## Research Approach

We compare three query-formulation strategies:

| Strategy | Description |
|---|---|
| **KEEP** | Use the original question unchanged. |
| **CONCAT** | Combine the question with relevant conversation history. |
| **REWRITE** | Generate a standalone query using an LLM. |

The research will examine retrieval performance, per-query help/tie/hurt outcomes, oracle headroom, and the quality–cost trade-off of selective rewriting.

The gate will be compared against simple baselines and matched-rate controls. Its effectiveness will be established through experiments rather than assumed in advance.

## Project Status

**Current stage: Retrieval foundation implemented; research experiments are ongoing.**

- [x] Repository setup and initial configuration
- [x] QReCC dataset inspection and schema checks
- [x] Passage-reference coverage analysis
- [x] SQLite FTS5 BM25 index construction
- [x] Passage-ID normalization and duplicate handling
- [x] Initial test suite — 100 tests passed in the last recorded run
- [ ] Evaluate KEEP, CONCAT, and REWRITE
- [ ] Analyze help/tie/hurt outcomes and oracle headroom
- [ ] Implement and evaluate selective gating
- [ ] Measure rewriting costs and quality–cost trade-offs

## Repository Structure

```text
selective-query-rewriting/
├── configs/          # Experiment configuration
├── src/
│   ├── data/         # Dataset loading and schema
│   ├── retrieval/    # BM25 index and retrieval
│   ├── rewriting/   # Query formulation and rewrite caching
│   ├── evaluation/  # Metrics, outcomes, and oracle analysis
│   └── gating/       # Gate features and decision strategies
├── tests/            # Automated tests
├── data/             # Local datasets and indexes (not committed)
├── outputs/          # Generated experiment results (not committed)
├── DATASET.md        # Dataset documentation
├── requirements.txt  # Python dependencies
└── README.md
```

This is a high-level overview; refer to the actual source files for the current implementation.

## Setup

### 1. Clone the repository

Replace `<repo-url>` with the repository's GitHub URL.

```bash
git clone <repo-url>
cd selective-query-rewriting
```

### 2. Create a virtual environment

Python 3.10 or later is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the QReCC files and passage corpus in the expected local directories:

```text
data/
├── qrecc/
│   ├── train.jsonl
│   └── test.jsonl
└── passages/
    └── raw/
        └── *.parquet
```

See [`DATASET.md`](DATASET.md) for dataset details and preparation instructions.

## Building the BM25 Index

The project uses SQLite FTS5 for disk-backed BM25 retrieval.

### Smoke-test index

Build a smaller index for quick validation:

```bash
python src/retrieval/build_index.py \
  --index-path data/index/bm25_fts5_smoke.sqlite
```

This indexes up to 10,000 unique passages.

### Full index

Build the index over all valid, unique passages:

```bash
python src/retrieval/build_index.py --max-passages 0
```

The default output path is:

```text
data/index/bm25_fts5_dedup_full.sqlite
```

The builder processes Parquet files in sorted filename order and retains the first valid occurrence of each normalized passage ID. It refuses to overwrite an existing index unless `--overwrite` is explicitly supplied.

The full index requires substantial disk space and build time. If a valid full index already exists locally, it does not need to be rebuilt.

## Running Tests

From the repository root, run:

```bash
python -m pytest -q
```

The last recorded run completed with 100 passing tests. Run the test suite again after making code changes.

## Reproducibility

For reliable comparisons, experiments should use consistent retrieval settings, documented query-formulation rules, and clearly defined evaluation metrics.

Keep training, validation, and test data separate. Do not use test outcomes to train the gate or tune its decision threshold.

Record relevant configuration, model settings, retrieval metrics, rewriting frequency, and cost measurements for each experiment.

## Notes

- Datasets, generated indexes, virtual environments, API keys, and large experiment artifacts must not be committed to Git.
- LLM rewriting may require an API key supplied through an environment variable. Never hardcode credentials.
- Oracle results represent an analytical upper bound, not a deployable prediction-time strategy.
- The novelty of the proposed approach must be established through literature review and experimental evidence.

## Dataset and References

Dataset provenance, citation details, and applicable licensing information are documented in [`DATASET.md`](DATASET.md). Verify relevant dataset and method citations before publication.
