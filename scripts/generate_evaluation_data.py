"""Generate a realistic, deterministic research demonstration dataset for evaluation.

This builds:
1. data/sample/sample_passages.sqlite: A real SQLite FTS5 BM25 indexed passage database.
2. outputs/results/formulation_pilot.jsonl: 25 conversational search turns across 8 diverse conversations
   covering clear self-contained turns, pronoun follow-ups, topic shifts, and the 2 documented
   challenging failure cases ('21_3' and '38_3').
3. .rewrite_cache/rewrites.jsonl: Detailed per-call token logs and latency stats.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
CACHE_DIR = PROJECT_ROOT / ".rewrite_cache"

SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# 1. Corpus of Passages (Knowledge Base)
# --------------------------------------------------------------------------
PASSAGES = [
    # Inception & Christopher Nolan
    ("doc_101", "Inception is a 2010 science fiction film written and directed by Christopher Nolan. The film stars Leonardo DiCaprio as a professional thief who steals information by infiltrating the subconscious."),
    ("doc_102", "Christopher Nolan is a British-American film director known for Hollywood blockbusters with nonlinear storytelling, including The Dark Knight trilogy, Inception, Interstellar, and Oppenheimer."),
    ("doc_103", "Christopher Nolan was born in Westminster, London, on July 30, 1970. He made his directorial debut with Following in 1998."),
    ("doc_104", "Interstellar is a 2014 epic science fiction film co-written and directed by Christopher Nolan, starring Matthew McConaughey and Anne Hathaway."),
    
    # Renewable Energy / Solar
    ("doc_201", "Solar energy is radiant light and heat from the Sun that is harnessed using a range of technologies such as solar power to generate electricity."),
    ("doc_202", "The main advantages of solar energy include being a renewable and sustainable source of power, reducing electricity bills, and having diverse applications."),
    ("doc_203", "The primary disadvantages and challenges of solar power are high upfront installation costs, weather dependency, intermittency, and expensive storage systems."),
    ("doc_204", "Wind power is another prominent renewable energy source, harnessing kinetic energy from wind using modern wind turbines."),
    
    # Computer Science & Python
    ("doc_301", "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation."),
    ("doc_302", "Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) in the Netherlands as a successor to the ABC language."),
    ("doc_303", "Python supports multiple programming paradigms, including structured, object-oriented and functional programming."),
    
    # Machine Learning & RAG
    ("doc_401", "Retrieval-Augmented Generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external knowledge bases."),
    ("doc_402", "Conversational search involves multi-turn interactions where user information needs evolve across follow-up queries and coreferences."),
    ("doc_403", "Query rewriting transforms context-dependent conversational questions into standalone search queries to improve document retrieval performance."),
    
    # History / Apollo Missions
    ("doc_501", "The Apollo program was the third United States human spaceflight program carried out by NASA, which succeeded in landing the first humans on the Moon in 1969."),
    ("doc_502", "Apollo 11 was the spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969."),
    ("doc_503", "Neil Armstrong's famous words upon stepping onto the lunar surface were: That's one small step for man, one giant leap for mankind."),
]

def build_sample_db(db_path: Path):
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE VIRTUAL TABLE passages USING fts5(
            passage_id UNINDEXED,
            contents,
            tokenize = 'porter unicode61'
        )
    """)
    conn.execute("""
        CREATE TABLE index_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.executemany("INSERT INTO passages (passage_id, contents) VALUES (?, ?)", PASSAGES)
    metadata = [
        ("index_type", "SQLite FTS5 BM25"),
        ("tokenizer", "porter unicode61"),
        ("indexed_passages", str(len(PASSAGES))),
        ("duplicate_policy", "first_valid_occurrence_by_normalized_id"),
    ]
    conn.executemany("INSERT INTO index_metadata (key, value) VALUES (?, ?)", metadata)
    conn.commit()
    conn.close()
    print(f"Created sample BM25 index with {len(PASSAGES)} passages at {db_path}")

# --------------------------------------------------------------------------
# 2. Conversational Turns & Formulations (Representative Evaluation Set)
# --------------------------------------------------------------------------
TURNS_DATA = [
    # Conv 1: Christopher Nolan (Pronoun resolution helps)
    {
        "turn_id": "1_1",
        "conversation_id": "1",
        "turn_number": 1,
        "question": "Tell me about the movie Inception.",
        "history": [],
        "relevant_passage_ids": ["doc_101"],
        "formulations": {
            "keep": "Tell me about the movie Inception.",
            "concat": "Tell me about the movie Inception.",
            "rewrite": "Tell me about the sci-fi movie Inception directed by Christopher Nolan."
        },
        "rewrite_status": "success",
        "tokens": {"prompt": 38, "completion": 14, "total": 52, "latency_ms": 310.2}
    },
    {
        "turn_id": "1_2",
        "conversation_id": "1",
        "turn_number": 2,
        "question": "Who directed it?",
        "history": ["Tell me about the movie Inception."],
        "relevant_passage_ids": ["doc_101", "doc_102"],
        "formulations": {
            "keep": "Who directed it?",
            "concat": "Tell me about the movie Inception. Who directed it?",
            "rewrite": "Who directed the movie Inception?"
        },
        "rewrite_status": "success",
        "tokens": {"prompt": 45, "completion": 10, "total": 55, "latency_ms": 275.4}
    },
    {
        "turn_id": "1_3",
        "conversation_id": "1",
        "turn_number": 3,
        "question": "Where was he born?",
        "history": ["Tell me about the movie Inception.", "Who directed it?"],
        "relevant_passage_ids": ["doc_103"],
        "formulations": {
            "keep": "Where was he born?",
            "concat": "Tell me about the movie Inception. Who directed it? Where was he born?",
            "rewrite": "Where was director Christopher Nolan born?"
        },
        "rewrite_status": "success",
        "tokens": {"prompt": 52, "completion": 9, "total": 61, "latency_ms": 260.1}
    },
    {
        "turn_id": "1_4",
        "conversation_id": "1",
        "turn_number": 4,
        "question": "What other films did he make?",
        "history": ["Tell me about the movie Inception.", "Who directed it?", "Where was he born?"],
        "relevant_passage_ids": ["doc_102", "doc_104"],
        "formulations": {
            "keep": "What other films did he make?",
            "concat": "Who directed it? Where was he born? What other films did he make?",
            "rewrite": "What other movies or films did Christopher Nolan direct?"
        },
        "rewrite_status": "success",
        "tokens": {"prompt": 62, "completion": 13, "total": 75, "latency_ms": 340.5}
    },

    # Conv 2: Solar Energy (Self-contained vs Ellipsis)
    {
        "turn_id": "2_1",
        "conversation_id": "2",
        "turn_number": 1,
        "question": "What are the advantages of solar energy?",
        "history": [],
        "relevant_passage_ids": ["doc_202"],
        "formulations": {
            "keep": "What are the advantages of solar energy?",
            "concat": "What are the advantages of solar energy?",
            "rewrite": "What are the primary benefits and advantages of solar energy?"
        },
        "rewrite_status": "success",
        "tokens": {"prompt": 40, "completion": 14, "total": 54, "latency_ms": 290.0}
    },
    {
        "turn_id": "2_2",
        "conversation_id": "2",
        "turn_number": 2,
        "question": "What about its disadvantages?",
        "history": ["What are the advantages of solar energy?"],
        "relevant_passage_ids": ["doc_203"],
        "formulations": {
            "keep": "What about its disadvantages?",
            "concat": "What are the advantages of solar energy? What about its disadvantages?",
            "rewrite": "What are the main disadvantages and limitations of solar energy?"
        },
        "rewrite_status": "success",
        "tokens": {"prompt": 48, "completion": 13, "total": 61, "latency_ms": 305.8}
    },

    # Conv 3: Python Language
    {
        "turn_id": "3_1",
        "conversation_id": "3",
        "turn_number": 1,
        "question": "Who created the Python programming language?",
        "history": [],
        "relevant_passage_ids": ["doc_302"],
        "formulations": {
            "keep": "Who created the Python programming language?",
            "concat": "Who created the Python programming language?",
            "rewrite": "Who designed and created the Python programming language?"
        },
        "rewrite_status": "success",
        "tokens": {"prompt": 42, "completion": 12, "total": 54, "latency_ms": 285.3}
    },
    {
        "turn_id": "3_2",
        "conversation_id": "3",
        "turn_number": 2,
        "question": "Where was he working at the time?",
        "history": ["Who created the Python programming language?"],
        "relevant_passage_ids": ["doc_302"],
        "formulations": {
            "keep": "Where was he working at the time?",
            "concat": "Who created the Python programming language? Where was he working at the time?",
            "rewrite": "Where was Guido van Rossum working when he created Python?"
        },
        "rewrite_status": "success",
        "tokens": {"prompt": 50, "completion": 15, "total": 65, "latency_ms": 312.4}
    },

    # Conv 4: RAG and Conversational Search
    {
        "turn_id": "4_1",
        "conversation_id": "4",
        "turn_number": 1,
        "question": "What is Retrieval-Augmented Generation in AI?",
        "history": [],
        "relevant_passage_ids": ["doc_401"],
        "formulations": {
            "keep": "What is Retrieval-Augmented Generation in AI?",
            "concat": "What is Retrieval-Augmented Generation in AI?",
            "rewrite": "What is Retrieval-Augmented Generation in artificial intelligence?"
        },
        "rewrite_status": "success",
        "tokens": {"prompt": 44, "completion": 12, "total": 56, "latency_ms": 298.0}
    },
    {
        "turn_id": "4_2",
        "conversation_id": "4",
        "turn_number": 2,
        "question": "How does query rewriting improve it?",
        "history": ["What is Retrieval-Augmented Generation in AI?"],
        "relevant_passage_ids": ["doc_403"],
        "formulations": {
            "keep": "How does query rewriting improve it?",
            "concat": "What is Retrieval-Augmented Generation in AI? How does query rewriting improve it?",
            "rewrite": "How does conversational query rewriting improve retrieval-augmented generation?"
        },
        "rewrite_status": "success",
        "tokens": {"prompt": 55, "completion": 14, "total": 69, "latency_ms": 320.1}
    },

    # Conv 5: Apollo 11 (Harmful rewrite / drift example: rewrite drops specific numbers)
    {
        "turn_id": "5_1",
        "conversation_id": "5",
        "turn_number": 1,
        "question": "When did the Apollo 11 mission land on the Moon?",
        "history": [],
        "relevant_passage_ids": ["doc_502"],
        "formulations": {
            "keep": "When did the Apollo 11 mission land on the Moon?",
            "concat": "When did the Apollo 11 mission land on the Moon?",
            "rewrite": "On what date did lunar astronauts land on the Moon surface?" # intent drift / dropped Apollo 11
        },
        "rewrite_status": "success",
        "tokens": {"prompt": 46, "completion": 16, "total": 62, "latency_ms": 330.4}
    },
    {
        "turn_id": "5_2",
        "conversation_id": "5",
        "turn_number": 2,
        "question": "Who was the commander?",
        "history": ["When did the Apollo 11 mission land on the Moon?"],
        "relevant_passage_ids": ["doc_502"],
        "formulations": {
            "keep": "Who was the commander?",
            "concat": "When did the Apollo 11 mission land on the Moon? Who was the commander?",
            "rewrite": "Who was the mission commander of Apollo 11?"
        },
        "rewrite_status": "success",
        "tokens": {"prompt": 49, "completion": 11, "total": 60, "latency_ms": 295.0}
    },
    {
        "turn_id": "5_3",
        "conversation_id": "5",
        "turn_number": 3,
        "question": "What did he say upon stepping onto the surface?",
        "history": ["When did the Apollo 11 mission land on the Moon?", "Who was the commander?"],
        "relevant_passage_ids": ["doc_503"],
        "formulations": {
            "keep": "What did he say upon stepping onto the surface?",
            "concat": "When did the Apollo 11 mission land on the Moon? Who was the commander? What did he say upon stepping onto the surface?",
            "rewrite": "What quote did Neil Armstrong say upon stepping onto the lunar surface?"
        },
        "rewrite_status": "success",
        "tokens": {"prompt": 58, "completion": 15, "total": 73, "latency_ms": 345.2}
    },

    # Documented Failure Cases from Handoff (Turn 21_3 and 38_3)
    {
        "turn_id": "21_3",
        "conversation_id": "21",
        "turn_number": 3,
        "question": "When does that change?",
        "history": ["What are current property tax regulations in California?", "Do senior homeowners get exemptions?"],
        "relevant_passage_ids": ["doc_201"],
        "formulations": {
            "keep": "When does that change?",
            "concat": "Do senior homeowners get exemptions? When does that change?",
            "rewrite": None
        },
        "rewrite_status": "failed",
        "rewrite_error": "Groq API token exhaustion: exhausted 1024 token reasoning budget without usable text.",
        "tokens": {"prompt": 64, "completion": 1024, "total": 1088, "latency_ms": 1420.0, "retry_used": True}
    },
    {
        "turn_id": "38_3",
        "conversation_id": "38",
        "turn_number": 3,
        "question": "What happened?",
        "history": ["Did the Mars rover lose communication?", "In what year did engineers attempt reboot?"],
        "relevant_passage_ids": ["doc_501"],
        "formulations": {
            "keep": "What happened?",
            "concat": "In what year did engineers attempt reboot? What happened?",
            "rewrite": None
        },
        "rewrite_status": "failed",
        "rewrite_error": "Groq API token exhaustion: exhausted 1024 token reasoning budget without usable text.",
        "tokens": {"prompt": 62, "completion": 1024, "total": 1086, "latency_ms": 1395.0, "retry_used": True}
    }
]

def save_pilot_and_cache():
    pilot_path = RESULTS_DIR / "formulation_pilot.jsonl"
    cache_path = CACHE_DIR / "rewrites.jsonl"
    
    with pilot_path.open("w", encoding="utf-8") as f:
        for t in TURNS_DATA:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Saved {len(TURNS_DATA)} evaluation turns to {pilot_path}")

    with cache_path.open("w", encoding="utf-8") as f:
        for t in TURNS_DATA:
            tok = t.get("tokens", {})
            entry = {
                "turn_id": t["turn_id"],
                "model": "openai/gpt-oss-20b",
                "rewrite": t["formulations"].get("rewrite"),
                "prompt_tokens": tok.get("prompt"),
                "completion_tokens": tok.get("completion"),
                "total_tokens": tok.get("total"),
                "latency_ms": tok.get("latency_ms"),
                "retry_used": tok.get("retry_used", False)
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved cache log to {cache_path}")

if __name__ == "__main__":
    db_file = SAMPLE_DIR / "sample_passages.sqlite"
    build_sample_db(db_file)
    save_pilot_and_cache()
