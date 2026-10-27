# Dataset Documentation

## 1. Overview

This project uses the **Question Rewriting in Conversational Context (QReCC)** dataset to study query formulation and retrieval in conversational search.

The dataset contains conversational questions, previous conversational context, reference rewrites, answers, and reference passage information. These fields support experiments comparing original questions, context-augmented questions, and rewritten standalone queries.

The project also uses a local passage corpus in Parquet format to build a disk-backed SQLite FTS5 BM25 retrieval index.

**Documentation status:** Local file layout, dataset fields, and passage-reference coverage have been inspected. Dataset provenance and licensing details should be checked against the original sources before publication.

## 2. Dataset Source and Citation

The existing project documentation identifies the following sources for QReCC:

| Field | Details |
|---|---|
| Dataset | Question Rewriting in Conversational Context (QReCC) |
| Paper | Anantha et al., *Open-Domain Question Answering Goes Conversational via Question Rewriting* |
| Publication | NAACL 2021 |
| arXiv | https://arxiv.org/abs/2010.04898 |
| GitHub | https://github.com/apple-aiml-research/ml-qrecc |
| Zenodo | https://zenodo.org/records/4769775 |
| DOI | 10.5281/zenodo.4769775 |
| License | CC BY-SA 3.0, as reported in the previous project documentation; verify against the applicable dataset record and files before redistribution |

### BibTeX

```bibtex
@inproceedings{anantha2021qrecc,
  title={Open-Domain Question Answering Goes Conversational via Question Rewriting},
  author={Anantha, Raviteja and Vakulenko, Svitlana and Tu, Zhucheng and Longpre, Shayne and Pulman, Stephen and Chappidi, Srinivas},
  booktitle={Proceedings of NAACL},
  year={2021}
}
```

The citation above follows the existing project documentation. Verify the complete bibliographic details against the original publication before submitting a paper.

## 3. Local Dataset Layout

The current project expects the following layout:

```text
data/
├── qrecc/
│   ├── train.jsonl
│   └── test.jsonl
├── passages/
│   └── raw/
│       ├── train-000.parquet
│       ├── train-001.parquet
│       └── train-002.parquet
├── processed/
└── index/
    └── bm25_fts5_dedup_full.sqlite
```

The JSONL files contain QReCC conversation turns. The Parquet files contain the passage corpus used by the retrieval index builder.

The `processed/` and `index/` directories contain generated artifacts and can be recreated when the required source data is available.

**Do not commit the dataset, generated index, or other large data artifacts to Git.**

## 4. Dataset Schema

The locally inspected QReCC records contain fields including the following:

| Field | Description |
|---|---|
| `Conversation_no` | Identifier grouping turns into conversations |
| `Turn_no` | Turn number within the conversation |
| `Conversation_source` | Source category associated with the conversation |
| `Context` | Conversational history |
| `Question` | Original user question |
| `Truth_rewrite` | Reference standalone rewrite |
| `Truth_answer` | Reference answer |
| `Answer_URL` | Answer-source URL or reference |
| `Truth_passages` | Reference passage information used for retrieval evaluation |

The exact fields available can vary by dataset variant. Use the actual local files and the shared schema implementation as the authority when loading records.

### Conversational context

The `Context` field provides information from earlier turns that may help interpret the current question.

For example:

- Previous question: "Tell me about Inception."
- Current question: "Who directed it?"
- Reference rewrite: "Who directed the movie Inception?"

The KEEP, CONCAT, and REWRITE strategies should apply consistent, documented rules for using this history.

Do not assume that every dataset variant represents context in exactly the same format. Validate the structure during data loading.

## 5. Dataset Statistics

The following counts were recorded during the project's local dataset inspection.

| Split | Turns | Conversations | Turns with reference passages |
|---|---:|---:|---:|
| Train | 63,501 | 10,823 | 29,596 |
| Test | 16,451 | 2,775 | 8,209 |
| **Total** | **79,952** | **13,598** | **37,805** |

Additional observations:

- Train records without reference passages: 33,905.
- Test records without reference passages: 8,242.
- The counts above describe the local files inspected and should be regenerated if the dataset changes.

These figures should not be assumed to apply to every published QReCC distribution or variant.

## 6. Passage Corpus

The project uses Parquet passage shards under:

```text
data/passages/raw/
```

The local files inspected were:

- `train-000.parquet`
- `train-001.parquet`
- `train-002.parquet`

Together, these files contain approximately 3 million source rows.

This is the local corpus used by the current implementation. It should not be confused with the larger passage corpus described in some original QReCC distribution documentation.

### Passage fields

The current index builder reads these columns from each Parquet shard:

- `id`: Passage identifier.
- `contents`: Passage text.

The builder reads the shards in sorted filename order and indexes valid passage identifiers and non-empty passage contents.

### Passage ID normalization

The index builder normalizes passage IDs before inserting them into SQLite.

For Markdown-formatted IDs whose final component contains a URL link, it extracts the URL from the link target. It then removes supported Markdown escape sequences and trims surrounding whitespace.

This normalization is important because reference identifiers and corpus identifiers must be represented consistently when matching retrieved passages to relevance judgments.

## 7. Index Construction and Duplicate Handling

The retrieval foundation uses SQLite FTS5 with BM25 ranking.

The default full-index path is:

```text
data/index/bm25_fts5_dedup_full.sqlite
```

The builder uses the `porter unicode61` tokenizer and records index metadata, including the indexing configuration and duplicate-handling policy.

### Duplicate policy

Passage shards are processed in sorted filename order.

1. Rows with missing or empty IDs or contents are skipped.
2. IDs are normalized before duplicate detection.
3. The first valid occurrence of a normalized ID is retained.
4. Later occurrences of the same normalized ID are skipped, even if their contents differ.

This policy makes index construction deterministic for a fixed set of input files and their ordering.

The full index previously built for this project contained **2,981,202 unique normalized passage IDs**. The builder reported 17,085 empty or invalid rows skipped and 1,713 duplicate-ID rows skipped.

These counts are specific to the inspected local corpus and should be rechecked if the source files change.

### Build commands

Build a smoke-test index:

```bash
python src/retrieval/build_index.py \
  --index-path data/index/bm25_fts5_smoke.sqlite
```

Build the full index:

```bash
python src/retrieval/build_index.py --max-passages 0
```

The builder does not overwrite an existing index unless `--overwrite` is specified. A full index can require several gigabytes of storage and significant build time.

## 8. Reference-Passage Coverage

Reference-passage coverage was checked by matching the normalized reference identifiers against IDs available in the local passage corpus.

The recorded results are:

| Measure | Train | Test |
|---|---:|---:|
| Total turns | 63,501 | 16,451 |
| Turns with reference passages | 29,596 | 8,209 |
| Turns without reference passages | 33,905 | 8,242 |
| Turns with all references available | 27,594 | 7,672 |
| Turns with some references missing | 1,493 | 381 |
| Turns with no references available | 509 | 156 |
| Total reference entries | 34,675 | 10,507 |
| Matched reference entries | 31,179 | 9,383 |
| Reference-entry coverage | 89.92% | 89.30% |

These figures describe the current local corpus and reference files.

### Implications for evaluation

Missing reference passages can affect retrieval metrics.

The evaluation implementation must explicitly define how it handles:

- Turns without reference passages.
- Turns whose references are only partially available.
- Turns whose references are all unavailable.
- Duplicate reference identifiers.
- Queries with no relevant passage in the local corpus.

The number of evaluated queries and the reference-coverage rate should be reported alongside the retrieval metrics. Do not silently treat unavailable references as evidence that a retrieval strategy failed.

Reference coverage alone does not establish retrieval quality.

## 9. Data Splits and Experimental Use

The project distinguishes between training data, validation data, and the final test evaluation.

Conversation-level splitting is important because turns from the same conversation can share context and entities. Splitting individual turns independently could introduce information leakage.

The split policy should therefore:

1. Keep all turns from a conversation in the same split.
2. Fit learned gates and tune thresholds using training and validation data only.
3. Keep final test outcomes separate from model selection.
4. Record the split procedure and random seed.

The availability of reference passage judgments must be checked separately for each split. The existence of answer text or a reference rewrite does not automatically provide passage-level relevance labels.

If additional pseudo-relevance judgments are generated, document the method and evaluate its limitations.

## 10. Download and Reproduction

Obtain the dataset from its original source and follow the applicable license and usage terms.

The current project expects the downloaded files to be placed under:

```text
data/qrecc/
data/passages/raw/
```

The exact download filenames and formats should be verified against the chosen dataset distribution before use. The original documentation describes JSON files, while the current implementation uses JSONL files; convert or adapt the files only through an explicit, tested preprocessing step.

After placing the files:

1. Inspect the input schema.
2. Run the dataset loader and validation tests.
3. Recalculate passage-reference coverage.
4. Build or validate the local BM25 index.
5. Record the dataset version and preprocessing configuration.

Generated files under `data/processed/` and `data/index/` should be reproducible from the source data and code wherever practical.

## 11. Related Dataset Variants

The original project documentation also mentions SCAI QReCC'21 shared-task files, including variants with fields such as `Truth_rewrite`, `Truth_passages`, and `Truth_answer`.

Dataset variants may differ in naming, record format, and available relevance judgments. The project should record the exact variant used for every experiment rather than assuming that files with similar names are interchangeable.

## 12. Limitations and Verification Checklist

Before final experiments or publication, verify the following:

- [ ] Record the exact source and version of the local dataset files.
- [ ] Verify the applicable license and attribution requirements.
- [ ] Confirm the field schema using the actual files.
- [ ] Recalculate turn counts and conversation counts.
- [ ] Recalculate passage-reference coverage against the current corpus.
- [ ] Document how unavailable reference passages are handled.
- [ ] Confirm the conversation-level split policy.
- [ ] Record the corpus version and duplicate-handling policy.
- [ ] Keep source data, indexes, and credentials out of Git.
- [ ] Verify all dataset and method citations before submission.

## 13. References

Anantha, R., Vakulenko, S., Tu, Z., Longpre, S., Pulman, S., and Chappidi, S. (2021). *Open-Domain Question Answering Goes Conversational via Question Rewriting*. Proceedings of NAACL.

- Paper: https://arxiv.org/abs/2010.04898
- Repository: https://github.com/apple-aiml-research/ml-qrecc
- Dataset record: https://zenodo.org/records/4769775

See the original publication and dataset record for authoritative citation, provenance, and licensing information.
