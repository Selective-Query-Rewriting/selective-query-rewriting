
from src.data.schema import ConversationTurn
from src.rewriting.cache import RewriteCache

turn = ConversationTurn(
    conversation_id="10",
    turn_id=1,
    question="Where was she born?",
    relevant_passage_ids=[],
    conversation_history=[],
)

cache = RewriteCache(
    cache_file=".rewrite_cache/diagnostic.jsonl",
    provider="groq",
    model="openai/gpt-oss-20b",
    max_tokens=256,
    temperature=0.0,
)

try:
    rewrite = cache.get_rewrite(
        turn.unique_id,
        turn.question,
        turn.conversation_history,
    )
    print("Original question:", turn.question)
    print("Generated rewrite:", rewrite)

except ValueError as error:
    print("Diagnosis:", error)
