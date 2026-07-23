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

Whenever a block's "depends_on" list names another block, that block's \
"prompt" text MUST actually include the literal placeholder \
"{{{{block_id.output}}}}" for every id in "depends_on" — never just \
describe the dependency in words (e.g. never write "explain this code" \
without also including "{{{{code1.output}}}}" so the actual code is \
attached). Omitting the placeholder means the expert receives no content \
to act on and will respond that nothing was provided.

If the request asks for code, create a separate "code" block whose prompt \
asks the expert to write the actual, complete, working code (not a "text" \
block describing code). A "code" block's output must be PURE CODE ONLY — \
never ask a code block to also include an explanation, description, or \
commentary about the code; that always belongs in its own "text" block. \
If the request asks for an explanation/description of that code, create a \
separate "text" block that depends on the code block and explains the \
*actual* code via {{{{block_id.output}}}}, rather than describing \
hypothetically what code "could" do. Never substitute a "text" block for \
what should be a "code" or "structured" block. If the request asks for \
multiple pieces of code (e.g. several languages, several functions) each \
with its own description, create one "code" block and one dependent "text" \
block PER item, not a single combined block for all of them.

Every "text" block's prompt must make clear that the resulting text is only \
one section of a single larger document that other blocks (possibly other \
"text" blocks) also contribute to. When there are multiple similar items \
(e.g. one description per language, per function, per example), each \
"text" block's prompt must explicitly name/identify which specific item it \
covers (e.g. "describe the Python version above", "describe the C++ \
version above") and instruct the expert to write ONLY about that item, \
without re-introducing the overall topic, repeating shared background \
information, or restating points already made in a sibling block. This \
prevents near-duplicate or overlapping prose when the same kind of section \
is generated multiple times.

The top-level response MUST be a single JSON *object* with exactly two \
keys, "blocks" and "assembly_template" — NEVER a bare JSON array/list of \
blocks.

Respond with ONLY a single JSON object matching this schema, with no \
markdown fences, no commentary, and no extra text before or after the JSON. \
Your ENTIRE response must be NOTHING OTHER THAN that JSON object — no \
preamble, no explanation of your reasoning, no summary, and no trailing \
remarks of any kind, before or after it:

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


def _normalize_depends_on(data: dict) -> dict:
    """Some models reference a block's output_slot (or its own id/output_slot
    mixed up) inside another block's `depends_on` instead of the referenced
    block's id. Rewrite any depends_on entry that matches a known
    output_slot to the corresponding block id, so a single mix-up doesn't
    fail DAG validation and burn a whole retry cycle."""
    blocks = data.get("blocks")
    if not isinstance(blocks, list):
        return data

    slot_to_id: dict[str, str] = {}
    known_ids: set[str] = set()
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_id = block.get("id")
        output_slot = block.get("output_slot")
        if isinstance(block_id, str):
            known_ids.add(block_id)
        if isinstance(output_slot, str) and isinstance(block_id, str):
            slot_to_id[output_slot] = block_id

    for block in blocks:
        if not isinstance(block, dict):
            continue
        depends_on = block.get("depends_on")
        if not isinstance(depends_on, list):
            continue
        block["depends_on"] = [
            slot_to_id[dep] if dep not in known_ids and dep in slot_to_id else dep
            for dep in depends_on
            if isinstance(dep, str)
        ]
    return data


def _ensure_dependency_references(data: dict) -> dict:
    """Guarantee that every declared `depends_on` relationship is actually
    reflected in the block's `prompt` text as a `{{block_id.output}}`
    placeholder. Small models frequently set `depends_on` correctly (so the
    DAG/scheduling is right) but never mention the dependency in the prompt
    itself, leaving the expert with nothing to act on (e.g. "explain this
    code" with no code ever attached) and leaving the *plan itself* with no
    visible trace of the relationship (confusing when inspected via
    --show-plan/--verbose). Rather than relying solely on runtime injection
    in experts.py, fix the plan data itself here: for every dependency id
    missing its placeholder, append `{{{{dep_id.output}}}}` to the prompt.
    """
    blocks = data.get("blocks")
    if not isinstance(blocks, list):
        return data

    for block in blocks:
        if not isinstance(block, dict):
            continue
        depends_on = block.get("depends_on")
        prompt = block.get("prompt")
        if not isinstance(depends_on, list) or not isinstance(prompt, str):
            continue

        missing = [
            dep for dep in depends_on
            if isinstance(dep, str) and f"{{{{{dep}.output}}}}" not in prompt
        ]
        if missing:
            references = " ".join(f"{{{{{dep}.output}}}}" for dep in missing)
            block["prompt"] = f"{prompt.rstrip()}\n\n{references}"
    return data


def _parse_json_object(cleaned: str) -> dict:
    """Parse `cleaned` as a single JSON object, tolerating trailing garbage
    after the first complete object (some models emit extra text/a repeated
    object after the closing brace, which otherwise fails as "Extra data")."""
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        if exc.msg.startswith("Extra data"):
            parsed, _ = json.JSONDecoder().raw_decode(cleaned)
        else:
            raise
    return _normalize_top_level_shape(parsed)


def _normalize_top_level_shape(parsed) -> dict:
    """Some models emit a bare JSON array of blocks instead of the required
    {"blocks": [...], "assembly_template": "..."} object. Detect this and
    wrap it, synthesizing an assembly_template that concatenates every
    block's output_slot placeholder in order."""
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        output_slots = [
            block["output_slot"]
            for block in parsed
            if isinstance(block, dict) and isinstance(block.get("output_slot"), str)
        ]
        assembly_template = "\n\n".join(f"{{{{{slot}}}}}" for slot in output_slots)
        return {"blocks": parsed, "assembly_template": assembly_template}
    raise json.JSONDecodeError("moderator output is not a JSON object or array", str(parsed), 0)


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
        logger.debug("Moderator raw output (attempt %d/%d):\n%s", attempt, max_retries, raw)
        cleaned = _strip_code_fences(raw)

        try:
            data = _parse_json_object(cleaned)
            data = _normalize_block_types(data)
            data = _normalize_depends_on(data)
            data = _ensure_dependency_references(data)
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
