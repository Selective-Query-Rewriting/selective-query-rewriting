
"""Groq-backed rewrite generation, persistent caching, and usage logging."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


def _stable_key(
    turn_id: str,
    model: str,
    prompt_version: str,
    question: str,
    conversation_history: list[str],
    max_tokens: int,
    temperature: float,
) -> str:
    """Create a cache key from the complete rewrite input and settings."""
    payload = {
        "turn_id": turn_id,
        "model": model,
        "prompt_version": prompt_version,
        "question": question.strip(),
        "conversation_history": [
            item.strip()
            for item in conversation_history
            if item and item.strip()
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RewriteCache:
    """Generate and persist LLM rewrites, reusing successful cached results."""

    def __init__(
        self,
        cache_file: str = ".rewrite_cache/rewrites.jsonl",
        provider: str = "groq",
        model: str = "openai/gpt-oss-20b",
        max_tokens: int = 256,
        temperature: float = 0.0,
        prompt_version: str = "standalone-v1",
    ) -> None:
        if provider.lower() != "groq":
            raise ValueError(
                "This implementation currently supports provider='groq'."
            )

        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is missing."
            )

        self.cache_path = Path(cache_file)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt_version = prompt_version

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        self.entries: dict[str, dict[str, Any]] = {}
        self.logs: list[dict[str, Any]] = []

        self._load_cache()

    def _load_cache(self) -> None:
        """Load valid JSONL entries and ignore malformed lines."""
        if not self.cache_path.exists():
            return

        with self.cache_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    entry = json.loads(line)
                    key = entry.get("cache_key")
                    rewrite = entry.get("rewrite")

                    if (
                        key
                        and isinstance(rewrite, str)
                        and rewrite.strip()
                    ):
                        self.entries[key] = entry

                except (json.JSONDecodeError, AttributeError):
                    continue

    def get_rewrite(
        self,
        turn_id: str,
        question: str,
        conversation_history: list[str],
    ) -> str:
        """Return a cached rewrite or generate and persist a new one."""

        key = _stable_key(
            turn_id=turn_id,
            model=self.model,
            prompt_version=self.prompt_version,
            question=question,
            conversation_history=conversation_history,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        # Reuse an existing successful rewrite.
        if key in self.entries:
            return self.entries[key]["rewrite"]

        history = "\n".join(
            item.strip()
            for item in conversation_history
            if item and item.strip()
        )

        if history:
            user_input = (
                f"Conversation history:\n{history}\n\n"
                f"Current question:\n{question.strip()}"
            )
        else:
            user_input = f"Current question:\n{question.strip()}"

        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the current question as a standalone "
                    "question using the conversation history. Preserve "
                    "the original meaning. Return only the final "
                    "question, with no explanation. If the history does "
                    "not identify a referent, do not invent one; preserve "
                    "the question's meaning."
                ),
            },
            {
                "role": "user",
                "content": user_input,
            },
        ]

        start = time.perf_counter()

        # First attempt uses the configured token budget.
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        choice = response.choices[0]
        rewrite = choice.message.content
        retry_used = False

        # Retry only if the model hit the token limit without returning text.
        if (
            (not rewrite or not rewrite.strip())
            and choice.finish_reason == "length"
        ):
            retry_max_tokens = max(self.max_tokens * 4, 1024)
            retry_used = True

            print(
                f"Retrying turn {turn_id} with "
                f"max_tokens={retry_max_tokens}"
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=retry_max_tokens,
                temperature=self.temperature,
            )

            choice = response.choices[0]
            rewrite = choice.message.content

        latency_ms = (time.perf_counter() - start) * 1000

        # Read usage from the final response, including any retry.
        usage = response.usage

        completion_tokens = (
            usage.completion_tokens if usage else None
        )
        total_tokens = (
            usage.total_tokens if usage else None
        )

        # Do not cache an empty response.
        if not rewrite or not rewrite.strip():
            message = choice.message
            reasoning = getattr(message, "reasoning", None)

            diagnostic = {
                "turn_id": turn_id,
                "model": response.model or self.model,
                "finish_reason": choice.finish_reason,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "has_reasoning": bool(reasoning),
                "reasoning_length": (
                    len(reasoning) if reasoning else 0
                ),
                "retry_used": retry_used,
            }

            raise ValueError(
                "Groq returned an empty rewrite. "
                f"Response diagnostics: {json.dumps(diagnostic)}"
            )

        rewrite = rewrite.strip()

        # Save the successful rewrite and its usage information.
        entry: dict[str, Any] = {
            "cache_key": key,
            "turn_id": turn_id,
            "model": response.model or self.model,
            "prompt_version": self.prompt_version,
            "rewrite": rewrite,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": round(latency_ms, 2),
            "finish_reason": choice.finish_reason,
            "retry_used": retry_used,
        }

        with self.cache_path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(entry, ensure_ascii=False) + "\n"
            )

        self.entries[key] = entry
        self.logs.append(entry.copy())

        return rewrite
