"""Pre-retrieval feature extraction for the gating model.

Status: STUB — implement after KEEP/CONCAT results are available.

Owner: Person B

Planned features (all pre-retrieval — no retrieval results used here)
-----------------------------------------------------------------------
- query_length          : number of tokens in the current question
- has_pronoun           : binary; detects 'he', 'she', 'it', 'they', 'that', etc.
- has_ellipsis          : binary; detect ellipsis patterns (e.g. "and what about...")
- turn_position         : turn_id (later turns may be more coreference-heavy)
- lexical_overlap       : Jaccard similarity between query tokens and history tokens

[UNRESOLVED] finalise the feature set after reviewing the distribution
of help/tie/hurt outcomes.
"""

from __future__ import annotations

# TODO (Person B): implement after outcomes are available.
raise NotImplementedError(
    "src/gating/features.py is not yet implemented."
)
