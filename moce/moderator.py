"""Moderator: turns a raw user request into a validated `Plan` by prompting
the "moderator" LLM to emit JSON matching the Plan schema, retrying with a
corrective prompt on validation failure."""
from __future__ import annotations

import json
import logging
from typing import Protocol

from pydantic import ValidationError

from moce.dag import DagError, validate_dag
from moce.schema import PLAN_EXAMPLE, PLAN_JSON_SCHEMA_HINT, Plan

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3

MODERATOR_SYSTEM_PROMPT = f"""You are the moderator in a multi-expert system. \
Given a user request, decompose it into a small set of content "blocks" that, \
together, fully answer the request. Each block has a type (text, code, \
structured, or image), a prompt for the specialist expert who will fill it, \
and may depend on other blocks (e.g. a text block explaining code may depend \
on the code block).

If the request asks for code, create a separate "code" block whose prompt \
asks the expert to write the actual, complete, working code (not a "text" \
block describing code). If the request asks for an explanation of that \
code, create a separate "text" block that depends on the code block and \
explains the *actual* code via {{{{block_id.output}}}}, rather than \
describing hypothetically what code "could" do. Never substitute a "text" \
block for what should be a "code" or "structured" block.

Respond with ONLY a single JSON object matching this schema, with no \
markdown fences, no commentary, and no extra text before or after the JSON:

{PLAN_JSON_SCHEMA_HINT}

{PLAN_EXAMPLE}
"""


class Generator(Protocol):
    """Anything with `ModelManager.generate`'s signature can act as the LLM
    backend for the moderator (allows easy mocking in tests)."""

    def generate(self, role: str, system_prompt: str, user_prompt: str, **kw) -> str:
        ...


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


class ModeratorError(RuntimeError):
    """Raised when the moderator fails to produce a valid Plan after all
    retries are exhausted."""


# Common near-miss block type names smaller models emit instead of the exact
# literals in Block.type. Normalized before schema validation so a single
# naming slip doesn't burn a whole retry cycle.
_TYPE_SYNONYMS: dict[str, str] = {
    "json": "structured",
    "data": "structured",
    "table": "structured",
    "summary": "structured",
    "prose": "text",
    "explanation": "text",
    "picture": "image",
    "diagram": "image",
    "svg": "image",
}


def _normalize_block_types(data: dict) -> dict:
    blocks = data.get("blocks")
    if not isinstance(blocks, list):
        return data
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if isinstance(block_type, str):
            normalized = block_type.strip().lower()
            block["type"] = _TYPE_SYNONYMS.get(normalized, normalized)
    return data


def generate_plan(
    generator: Generator,
    user_request: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Plan:
    """Ask the moderator model to produce a valid Plan for `user_request`,
    retrying with corrective feedback on JSON/schema/DAG errors."""
    user_prompt = user_request
    last_error: str | None = None

    for attempt in range(1, max_retries + 1):
        raw = generator.generate(
            role="moderator",
            system_prompt=MODERATOR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        cleaned = _strip_code_fences(raw)

        try:
            data = json.loads(cleaned)
            data = _normalize_block_types(data)
            plan = Plan.model_validate(data)
            validate_dag(plan)
            return plan
        except (json.JSONDecodeError, ValidationError, DagError) as exc:
            last_error = str(exc)
            logger.warning(
                "Moderator plan attempt %d/%d invalid: %s", attempt, max_retries, last_error
            )
            user_prompt = (
                f"{user_request}\n\n"
                f"Your previous response was invalid: {last_error}\n"
                "Respond again with ONLY a corrected JSON object matching the schema."
            )

    raise ModeratorError(
        f"moderator failed to produce a valid plan after {max_retries} attempts: {last_error}"
    )
