"""LLM rewrite cache and call logger.

Status: STUB — do not use until the formulation module is tested and
the LLM provider / model are agreed upon in configs/experiment.yaml.

Owner: Person B

Responsibilities
----------------
- Read from and write to a persistent JSONL cache keyed by turn unique_id.
- Call the LLM API only when no cache entry exists.
- Log every LLM call: model, prompt, response, token counts, latency.
- Expose token usage for cost-aware evaluation.

API keys must be read from environment variables. Never hardcode them.
"""

from __future__ import annotations

# TODO (Person B): implement after formulate.py tests pass and the
#   LLM provider is agreed upon in configs/experiment.yaml.

raise NotImplementedError(
    "src/rewriting/cache.py is not yet implemented. "
    "See the module docstring for the planned interface."
)
