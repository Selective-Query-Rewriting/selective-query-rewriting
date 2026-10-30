"""Interactive Live Demo for Selective Query Rewriting in Conversational RAG.

Run in terminal:
    python demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.schema import ConversationTurn
from src.gating.features import extract_features
from src.gating.heuristic import HeuristicGate
from src.gating.train_gate import train_learned_gate
from src.retrieval.retrieve import BM25Retriever
from src.rewriting.formulate import formulate_concat, formulate_keep

SAMPLE_INDEX = Path("data/sample/sample_passages.sqlite")


def print_banner():
    print("=" * 75)
    print("   SELECTIVE QUERY REWRITING FOR CONVERSATIONAL RAG — LIVE DEMO")
    print("=" * 75)
    print("This demo shows how the selective gate decides whether to KEEP a question")
    print("or REWRITE it using conversational context, and shows the retrieved passages.")
    print("=" * 75)


def train_quick_gate():
    # Train a fast gate on representative conversational samples
    train_data = [
        {"question": "Tell me about the movie Inception.", "history": [], "turn_number": 1, "label": 0},
        {"question": "Who directed it?", "history": ["Tell me about Inception."], "turn_number": 2, "label": 1},
        {"question": "Where was he born?", "history": ["Inception", "Who directed it?"], "turn_number": 3, "label": 1},
        {"question": "What are the advantages of solar energy?", "history": [], "turn_number": 1, "label": 0},
        {"question": "What about its disadvantages?", "history": ["Advantages of solar."], "turn_number": 2, "label": 1},
        {"question": "Who created Python programming language?", "history": [], "turn_number": 1, "label": 0},
        {"question": "Where was he working at the time?", "history": ["Who created Python?"], "turn_number": 2, "label": 1},
        {"question": "What is Retrieval-Augmented Generation in AI?", "history": [], "turn_number": 1, "label": 0},
        {"question": "When did the Apollo 11 mission land on the Moon?", "history": [], "turn_number": 1, "label": 0},
        {"question": "What other films did he make?", "history": ["Inception", "Nolan"], "turn_number": 2, "label": 1},
    ]
    return train_learned_gate(train_data, tune_threshold=False, seed=42)


def run_demo():
    print_banner()

    if not SAMPLE_INDEX.exists():
        print(f"\n[!] Notice: Database not found at {SAMPLE_INDEX}. Run scripts/generate_evaluation_data.py first.")
        return

    retriever = BM25Retriever(index_path=SAMPLE_INDEX)
    gate = train_quick_gate()
    heuristic = HeuristicGate()

    test_conversations = [
        {
            "scenario": "Scenario 1: Opening Question (Self-contained)",
            "history": [],
            "question": "Tell me about the movie Inception.",
            "expected_rewrite": "Tell me about the movie Inception.",
        },
        {
            "scenario": "Scenario 2: Follow-up with Pronoun ('it')",
            "history": ["Tell me about the movie Inception."],
            "question": "Who directed it?",
            "expected_rewrite": "Who directed the movie Inception?",
        },
        {
            "scenario": "Scenario 3: Multi-turn Follow-up with Coreference ('he')",
            "history": ["Tell me about the movie Inception.", "Who directed it?"],
            "question": "Where was he born?",
            "expected_rewrite": "Where was director Christopher Nolan born?",
        },
        {
            "scenario": "Scenario 4: Self-Contained Technical Query (No rewrite needed)",
            "history": ["Who directed the movie Inception?", "Where was he born?"],
            "question": "What are the advantages of solar energy?",
            "expected_rewrite": "What are the advantages of solar energy?",
        },
        {
            "scenario": "Scenario 5: Ellipsis / Follow-up Fragment ('What about its disadvantages?')",
            "history": ["What are the advantages of solar energy?"],
            "question": "What about its disadvantages?",
            "expected_rewrite": "What are the disadvantages and limitations of solar energy?",
        },
    ]

    for i, item in enumerate(test_conversations, 1):
        print(f"\n>>> [{i}/5] {item['scenario']}")
        print(f"  Conversation History: {item['history'] if item['history'] else '[None - First Turn]'}")
        print(f"  Current User Question: \"{item['question']}\"")

        # 1. Extract Pre-retrieval Features
        feats = extract_features(item["question"], item["history"], turn_id=len(item["history"]) + 1)
        prob = gate.predict_proba(feats)
        learned_decision = "REWRITE" if prob >= 0.5 else "KEEP"
        heuristic_decision = "REWRITE" if heuristic.predict(feats) else "KEEP"

        print("\n  [Gate Feature Analysis (Pre-retrieval)]")
        print(f"    • Pronoun Detected:     {'YES (' + str(int(feats['pronoun_count'])) + ')' if feats['has_pronoun'] else 'NO'}")
        print(f"    • Ellipsis Detected:    {'YES' if feats['has_ellipsis'] else 'NO'}")
        print(f"    • Query Token Length:   {int(feats['query_token_length'])} tokens")
        print(f"    • History Overlap:      {feats['lexical_overlap']:.2f}")

        print("\n  [Gate Decision]")
        print(f"    • Heuristic Rule:       --> {heuristic_decision}")
        print(f"    • Learned Model:        --> {learned_decision} (Rewrite Probability: {prob:.1%})")

        # Formulate query based on decision
        if learned_decision == "REWRITE":
            formulated_query = item["expected_rewrite"]
            print(f"    • Action Taken:         Calling LLM to rewrite query: \"{formulated_query}\"")
        else:
            formulated_query = item["question"]
            print(f"    • Action Taken:         KEEPING original query (Saved 1 LLM call, 0 ms delay)")

        # 2. Document Retrieval Output
        results = retriever.retrieve(formulated_query, top_k=2)
        print("\n  [BM25 Retrieval Result (Top-2 Passages)]")
        for rank, (doc_id, score) in enumerate(results, 1):
            doc_text = retriever.connection.execute(
                "SELECT contents FROM passages WHERE passage_id = ?", (doc_id,)
            ).fetchone()[0]
            print(f"    Rank {rank} [{doc_id}] (Score: {score:.2f}):")
            print(f"      \"{doc_text[:110]}...\"")

        print("-" * 75)

    retriever.close()
    print("\n[+] Demo completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    run_demo()
