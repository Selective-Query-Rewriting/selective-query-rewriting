"""tests/test_gating.py — Unit tests for Phase 5 gating models and feature extraction."""

from pathlib import Path
from src.data.schema import ConversationTurn
from src.gating.features import extract_features, feature_names
from src.gating.heuristic import HeuristicGate
from src.gating.random_gate import RandomGate
from src.gating.train_gate import (
    LearnedGate,
    create_conversation_split,
    train_learned_gate,
)


def make_turn(conv_id="1", turn_id=1, question="What is Python?", history=None):
    return ConversationTurn(
        conversation_id=conv_id,
        turn_id=turn_id,
        question=question,
        relevant_passage_ids=[],
        conversation_history=history or [],
    )


class TestFeatures:
    def test_first_turn_features(self):
        turn = make_turn(conv_id="1", turn_id=1, question="Tell me about Christopher Nolan.", history=[])
        feats = extract_features(turn)

        assert feats["is_first_turn"] == 1.0
        assert feats["turn_position"] == 1.0
        assert feats["history_turn_count"] == 0.0
        assert feats["has_pronoun"] == 0.0
        assert feats["has_ellipsis"] == 0.0
        assert feats["lexical_overlap"] == 0.0

    def test_pronoun_detection(self):
        turn = make_turn(
            conv_id="1",
            turn_id=2,
            question="Who directed it?",
            history=["Tell me about Inception."],
        )
        feats = extract_features(turn)

        assert feats["is_first_turn"] == 0.0
        assert feats["turn_position"] == 2.0
        assert feats["has_pronoun"] == 1.0
        assert feats["pronoun_count"] >= 1.0

    def test_ellipsis_detection(self):
        turn = make_turn(
            conv_id="1",
            turn_id=3,
            question="What about his later movies?",
            history=["Tell me about Inception.", "Christopher Nolan directed it."],
        )
        feats = extract_features(turn)

        assert feats["has_ellipsis"] == 1.0

    def test_lexical_overlap(self):
        turn = make_turn(
            conv_id="2",
            turn_id=2,
            question="Tell me more about solar energy advantages.",
            history=["What is solar energy?"],
        )
        feats = extract_features(turn)

        assert feats["lexical_overlap"] > 0.0

    def test_feature_names_stable(self):
        keys = feature_names()
        assert "has_pronoun" in keys
        assert "is_first_turn" in keys
        assert "query_token_length" in keys


class TestHeuristicGate:
    def test_first_turn_not_rewritten(self):
        gate = HeuristicGate()
        turn = make_turn(turn_id=1, question="What is natural language processing?")
        assert gate.predict(turn) is False
        assert gate.predict_action(turn) == "keep"

    def test_pronoun_triggered_rewrite(self):
        gate = HeuristicGate()
        turn = make_turn(turn_id=2, question="When was it founded?", history=["Tell me about Apple."])
        assert gate.predict(turn) is True
        assert gate.predict_action(turn) == "rewrite"

    def test_ellipsis_triggered_rewrite(self):
        gate = HeuristicGate()
        turn = make_turn(turn_id=2, question="And what else?", history=["Tell me about Apple."])
        assert gate.predict(turn) is True

    def test_short_query_followup_rewrite(self):
        gate = HeuristicGate(short_query_threshold=4)
        turn = make_turn(turn_id=2, question="Why?", history=["Python was created by Guido."])
        assert gate.predict(turn) is True

    def test_long_explicit_followup_keeps(self):
        gate = HeuristicGate()
        turn = make_turn(
            turn_id=2,
            question="Can artificial intelligence create realistic images and synthetic voices without supervision?",
            history=["Tell me about machine learning."],
        )
        assert gate.predict(turn) is False


class TestRandomGate:
    def test_deterministic_seed(self):
        g1 = RandomGate(rewrite_rate=0.4, seed=123)
        g2 = RandomGate(rewrite_rate=0.4, seed=123)

        preds1 = [g1.predict() for _ in range(20)]
        preds2 = [g2.predict() for _ in range(20)]
        assert preds1 == preds2

    def test_extreme_rates(self):
        always = RandomGate(rewrite_rate=1.0)
        never = RandomGate(rewrite_rate=0.0)

        assert all(always.predict() for _ in range(10))
        assert not any(never.predict() for _ in range(10))


class TestConversationSplit:
    def test_no_conversation_leakage(self):
        # 10 conversations with 3 turns each
        turns = []
        for c in range(10):
            for t in range(1, 4):
                turns.append({"conversation_id": str(c), "turn_id": f"{c}_{t}"})

        splits = create_conversation_split(turns, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

        train_convs = {x["conversation_id"] for x in splits["train"]}
        val_convs = {x["conversation_id"] for x in splits["val"]}
        test_convs = {x["conversation_id"] for x in splits["test"]}

        # Check disjoint conversations
        assert len(train_convs & val_convs) == 0
        assert len(train_convs & test_convs) == 0
        assert len(val_convs & test_convs) == 0

        # All turns belonging to a conv must be in the same split
        assert len(splits["train"]) == len(train_convs) * 3
        assert len(splits["val"]) == len(val_convs) * 3
        assert len(splits["test"]) == len(test_convs) * 3


class TestLearnedGate:
    def test_training_and_serialization(self, tmp_path):
        train_data = [
            {"question": "Who directed it?", "history": ["Inception."], "turn_number": 2, "label": 1},
            {"question": "What about his movies?", "history": ["Inception."], "turn_number": 3, "label": 1},
            {"question": "Tell me about France history.", "history": [], "turn_number": 1, "label": 0},
            {"question": "Where is the Eiffel Tower situated?", "history": ["Paris."], "turn_number": 2, "label": 0},
        ]
        val_data = [
            {"question": "How did he die?", "history": ["Alan Turing."], "turn_number": 2, "label": 1},
            {"question": "What is quantum mechanics?", "history": [], "turn_number": 1, "label": 0},
        ]

        gate = train_learned_gate(train_data, val_turns=val_data, tune_threshold=True, seed=42)

        # Test inference
        prob = gate.predict_proba({"question": "Where was she born?", "history": ["Ada Lovelace."], "turn_number": 2})
        assert 0.0 <= prob <= 1.0

        action = gate.predict_action({"question": "Tell me about cars.", "history": [], "turn_number": 1})
        assert action in ("rewrite", "keep")

        # Test save & load
        model_file = tmp_path / "learned_gate.pkl"
        gate.save(model_file)
        assert model_file.exists()

        loaded_gate = LearnedGate.load(model_file)
        prob_loaded = loaded_gate.predict_proba({"question": "Where was she born?", "history": ["Ada Lovelace."], "turn_number": 2})
        assert prob_loaded == prob
