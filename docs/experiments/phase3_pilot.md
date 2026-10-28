# Phase 3 — Formulation Baselines and Pilot

## Objective
Generate three query formulations for eligible QReCC conversational turns: KEEP, CONCAT, and REWRITE.

## Configuration
- Dataset: QReCC training split
- Pilot conversations: First 50 conversations
- Eligible turns: 136
- Retrieval formulation methods: KEEP, CONCAT, REWRITE
- Rewrite provider: Groq
- Rewrite model: `openai/gpt-oss-20b`
- Rewrite cache: `.rewrite_cache/rewrites.jsonl`
- Temperature: 0.0
- Configured maximum output tokens: 256, with a retry at 1,024 tokens for empty responses caused by output-length exhaustion

## Results
- Total eligible turns: 136
- Successful rewrites: 134
- Failed rewrites: 2
- Rewrite success rate: 98.5%
- Missing KEEP formulations: 0
- Missing CONCAT formulations: 0
- Missing REWRITE formulations: 2
- Automated tests: 108 passed

## Failed Rewrites
- `21_3`: “When does that change?”
- `38_3`: “What happened?”

Both rewrite attempts exhausted the retry budget without returning usable final text. These failures are recorded rather than replaced with manually invented rewrites.

## Evaluation Policy
KEEP and CONCAT remain available for all 136 turns. REWRITE evaluation must explicitly document how the two failed generations are handled. Report rewrite-generation success and failure rates separately from retrieval performance. Do not silently treat failed rewrites as successful or omit them without reporting the resulting evaluation subset.

## Phase 3 Status
Pilot generation and structural validation are complete. Final sign-off depends on recording and consistently applying the failure-handling policy.
