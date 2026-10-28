
"""Test the integrated query rewriting pipeline with Groq."""

from pathlib import Path
import yaml

from src.data.schema import ConversationTurn
from src.rewriting.cache import RewriteCache
from src.rewriting.formulate import formulate_with_cache


def main():
    # Load the experiment configuration.
    with open("configs/experiment.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    rewrite_config = config["rewriting"]

    cache_path = Path(rewrite_config["cache_file"])

    max_tokens = int(rewrite_config.get("max_tokens", 256))
    if max_tokens < 256:
        raise ValueError(
            "Set rewriting.max_tokens to 256 in "
            "configs/experiment.yaml before running this test."
        )

    # Initialize the cache and Groq client.
    rewrite_cache = RewriteCache(
        cache_file=str(cache_path),
        provider=rewrite_config["provider"],
        model=rewrite_config["model"],
        max_tokens=max_tokens,
        temperature=float(rewrite_config.get("temperature", 0.0)),
    )

    # Create one example conversational turn.
    turn = ConversationTurn(
        conversation_id="manual-pilot",
        turn_id=2,
        question="What about its disadvantages?",
        relevant_passage_ids=[],
        conversation_history=[
            "What are the advantages of solar energy?"
        ],
    )

    print("\nOriginal question:")
    print(turn.question)

    print("\nGenerating or retrieving the rewrite...")

    size_before = (
        cache_path.stat().st_size if cache_path.exists() else 0
    )

    # First call: generate a rewrite or retrieve an existing one.
    first_rewrite = formulate_with_cache(
        turn,
        method="rewrite",
        rewrite_cache=rewrite_cache,
    )

    size_after_first = (
        cache_path.stat().st_size if cache_path.exists() else 0
    )

    print("\nFirst rewrite:")
    print(first_rewrite)

    if size_after_first > size_before:
        print("\nA new cache entry was saved.")
    else:
        print("\nAn existing cached rewrite was reused.")

    # Second call: the same input should reuse the saved result.
    second_rewrite = formulate_with_cache(
        turn,
        method="rewrite",
        rewrite_cache=rewrite_cache,
    )

    size_after_second = (
        cache_path.stat().st_size if cache_path.exists() else 0
    )

    print("\nSecond rewrite:")
    print(second_rewrite)

    assert first_rewrite == second_rewrite, (
        "The same input returned different rewrites."
    )

    assert size_after_second == size_after_first, (
        "The second call unexpectedly added another cache entry."
    )

    print("\nSUCCESS: Integrated rewriting and cache reuse work.")


if __name__ == "__main__":
    main()
