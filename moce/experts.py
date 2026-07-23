"""Experts: build per-block-type prompts, invoke the corresponding model, and
validate/clean the raw output so only the requested content type is kept
("no derailment")."""
from __future__ import annotations

import ast
import json
import logging
import os
import re
from pathlib import Path
from typing import Protocol

from moce.schema import Block, BlockResult

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3

# Directory where generated images are saved; the assembled document embeds
# a Markdown reference to the file at this path. Overridable via the
# MOCE_IMAGE_OUTPUT_DIR environment variable.
IMAGE_OUTPUT_DIR = Path(os.environ.get("MOCE_IMAGE_OUTPUT_DIR", "moce_output/images"))

_BOILERPLATE_PREFIXES = re.compile(
    r"^(?:(?:sure|okay|ok|certainly|here('|’)s|here is|of course|the answer is|the answer)"
    r"[,:!.\s-]*)+",
    re.IGNORECASE,
)

_SYSTEM_PROMPTS: dict[str, str] = {
    "text": (
        "You are a text-writing expert collaborating with other experts on a "
        "single document. Output ONLY the prose text requested for this block. "
        "Do not include headings, preambles, disclaimers, or any commentary "
        "about what you are doing. Do not repeat content from other blocks. "
        "This text is only ONE section of a larger document that other "
        "blocks also contribute to — if this section covers one of several "
        "similar items (e.g. one of multiple languages/examples/functions), "
        "write ONLY about the specific item named in the prompt below; do "
        "not re-introduce the overall topic, restate shared background, or "
        "repeat points that belong to sibling sections about other items. "
        "Directly produce the requested content itself — do not describe what "
        "could or would be written/created/shown; do not talk about the task "
        "in the abstract or hypothetically."
    ),
    "code": (
        "You are a coding expert collaborating with other experts on a single "
        "document. Output PURE CODE ONLY for this block: no markdown code "
        "fences, no explanations, no commentary, and no prose sentences of "
        "any kind before, after, or interleaved with the code. Any "
        "explanation of the code is handled by a separate text block "
        "elsewhere — do not include it here, even if the original request "
        "asked for a description. Write real, complete, working code — "
        "never a description of what the code would do. Code comments "
        "(using the target language's comment syntax) are fine, but plain "
        "prose paragraphs are not."
    ),
    "structured": (
        "You are a structured-data expert collaborating with other experts on "
        "a single document. Output ONLY valid JSON matching what is requested "
        "for this block. No markdown fences, no explanations, no trailing text."
    ),
}


class Generator(Protocol):
    def generate(self, role: str, system_prompt: str, user_prompt: str, **kw) -> str:
        ...

    def generate_image(self, role: str, prompt: str, output_path: str, **kw) -> str:
        ...


def _substitute_dependencies(prompt: str, context: dict[str, BlockResult]) -> str:
    def replace(match: re.Match) -> str:
        block_id = match.group(1)
        result = context.get(block_id)
        if result is None:
            return match.group(0)
        return result.validated_output

    return re.sub(r"\{\{(\w[\w-]*)\.output\}\}", replace, prompt)


def _append_missing_dependency_outputs(
    prompt: str, original_prompt: str, block: Block, context: dict[str, BlockResult]
) -> str:
    """Defensive safety net: the moderator sometimes lists a block in
    `depends_on` but forgets to actually reference its content via the
    `{{block_id.output}}` placeholder in the prompt text (e.g. a "text"
    block asks to "explain this code" without the code ever being attached).
    Without this, the expert would receive a prompt with nothing to act on
    and produce a response like "I'm sorry, but you haven't provided any "
    "code for me to describe." As a fallback, any dependency declared in
    `depends_on` whose placeholder is missing from the *original* (pre-
    substitution) prompt has its output appended explicitly, so the
    dependency's content always reaches the expert one way or another.
    """
    missing = [
        (dep_id, context[dep_id].validated_output)
        for dep_id in block.depends_on
        if dep_id in context and f"{{{{{dep_id}.output}}}}" not in original_prompt
    ]
    if not missing:
        return prompt

    appended = "\n\n".join(f"[{dep_id}]:\n{output}" for dep_id, output in missing)
    return f"{prompt}\n\n{appended}"


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


def _clean_text(text: str) -> str:
    return _BOILERPLATE_PREFIXES.sub("", text.strip(), count=1).strip()


def _extract_code(text: str) -> str:
    """Extract just the code from `text`, discarding any prose the model
    appended despite being told not to. This is fence-based extraction (not
    NLP-based prose stripping): small models frequently still emit a
    trailing explanation paragraph after the code, sometimes behind a
    matched ```/``` pair, sometimes behind a single trailing/stray ``` with
    no matching opener. Handles, in order:
      1. A proper ```lang\ncode\n``` fenced block anywhere in the text ->
         keep only the fenced content.
      2. A single unpaired ``` marker (e.g. code, then a lone closing
         fence, then prose) -> keep only the text before that marker.
      3. No fences at all -> return the text unchanged (nothing safe to
         trim without language-aware parsing).
    """
    text = text.strip()

    paired = re.search(r"```(?:\w+)?\n?(.*?)```", text, re.DOTALL)
    if paired:
        return paired.group(1).strip()

    if "```" in text:
        return text.split("```", 1)[0].strip()

    return text


# Aliases the `language` field may use for Python; only these are eligible
# for the ast.parse() syntax check below, since it can only ever recognize
# Python and would produce a misleading "invalid" message for any other
# target language.
_PYTHON_LANGUAGE_NAMES = {"python", "py", "python3"}


def _validate_code(text: str, language: str | None = None) -> str:
    code = _extract_code(text)

    # Only syntax-check when the block is (or may be) Python: `language` is
    # unset for legacy/unlabeled blocks (assume Python, the historical
    # default), or explicitly "python". Any other explicit language is
    # skipped entirely rather than logged, since ast.parse() can only ever
    # validate Python and would otherwise misleadingly claim non-Python code
    # (e.g. C++, JS) "is not valid Python" on every single run.
    if language is None or language.strip().lower() in _PYTHON_LANGUAGE_NAMES:
        try:
            ast.parse(code)
        except SyntaxError:
            logger.debug("code block (language=%s) is not valid Python", language)
    return code


def _validate_structured(text: str) -> str:
    cleaned = _strip_code_fences(text)
    data = json.loads(cleaned)  # raises json.JSONDecodeError if invalid
    return json.dumps(data)


def _run_image_block(generator: Generator, block: Block, context: dict[str, BlockResult]) -> BlockResult:
    """Generate an image for `block` via the generator's `generate_image`,
    returning a Markdown image reference to the saved file as the block's
    validated output."""
    prompt = _substitute_dependencies(block.prompt, context)
    prompt = _append_missing_dependency_outputs(prompt, block.prompt, block, context)
    if block.constraints:
        prompt = f"{prompt}\nAdditional constraints: {block.constraints}"

    output_path = IMAGE_OUTPUT_DIR / f"{block.id}.png"
    try:
        saved_path = generator.generate_image(role="image", prompt=prompt, output_path=str(output_path))
    except Exception as exc:  # noqa: BLE001 - surface any generation failure as an invalid block
        logger.warning("Image block '%s' failed to generate: %s", block.id, exc)
        return BlockResult(
            block_id=block.id,
            status="invalid",
            error_message=f"image generation failed: {exc}",
        )

    alt_text = block.constraints or block.id
    return BlockResult(
        block_id=block.id,
        raw_output=saved_path,
        validated_output=f"![{alt_text}]({saved_path})",
        status="ok",
    )


def run_block(
    generator: Generator,
    block: Block,
    context: dict[str, BlockResult],
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> BlockResult:
    """Run the expert for `block`, substituting dependency outputs into its
    prompt, validating the result, and retrying on validation failure up to
    `max_retries` times before marking the block invalid."""
    if block.type == "image":
        return _run_image_block(generator, block, context)

    system_prompt = _SYSTEM_PROMPTS[block.type]
    if block.constraints:
        system_prompt = f"{system_prompt}\nAdditional constraints: {block.constraints}"

    user_prompt = _substitute_dependencies(block.prompt, context)
    user_prompt = _append_missing_dependency_outputs(user_prompt, block.prompt, block, context)
    last_error: str | None = None

    for attempt in range(1, max_retries + 1):
        raw = generator.generate(
            role=block.type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        logger.debug(
            "Block '%s' raw output (attempt %d/%d):\n%s", block.id, attempt, max_retries, raw
        )
        try:
            if block.type == "code":
                validated = _validate_code(raw, block.language)
            elif block.type == "structured":
                validated = _validate_structured(raw)
            else:  # text
                validated = _clean_text(raw)

            return BlockResult(
                block_id=block.id,
                raw_output=raw,
                validated_output=validated,
                status="ok",
                retries=attempt - 1,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            logger.warning(
                "Block '%s' attempt %d/%d failed validation: %s",
                block.id, attempt, max_retries, last_error,
            )
            user_prompt = (
                f"{user_prompt}\n\n"
                f"Your previous response was invalid: {last_error}\n"
                "Respond again with ONLY the corrected content, nothing else."
            )

    return BlockResult(
        block_id=block.id,
        status="invalid",
        retries=max_retries,
        error_message=f"failed validation after {max_retries} attempts: {last_error}",
    )
