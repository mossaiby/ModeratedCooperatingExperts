"""Assembler: substitutes each block's validated output into the plan's
assembly template to produce the final document."""
from __future__ import annotations

import re

from moce.schema import BlockResult, Plan


def assemble(plan: Plan, results: dict[str, BlockResult]) -> str:
    """Fill `plan.assembly_template`'s `{{output_slot}}` placeholders with the
    corresponding block's validated output (or an error marker if the block
    failed)."""
    content_by_key: dict[str, str] = {}
    for block in plan.blocks:
        result = results.get(block.id)
        if result is None:
            content = f"[ERROR: block '{block.id}' was never executed]"
        elif result.status == "ok":
            content = result.validated_output
        else:
            content = f"[ERROR: block '{block.id}' failed validation: {result.error_message}]"

        content_by_key[block.output_slot] = content
        # Also index by block id: moderator models sometimes reference a
        # block's own id in assembly_template (dependency syntax) instead of
        # its declared output_slot. output_slot is the authoritative key if
        # the two happen to collide.
        content_by_key.setdefault(block.id, content)

    def replace(match: re.Match) -> str:
        key = match.group(1)
        return content_by_key.get(key, match.group(0))

    # Tolerate a moderator model mistakenly appending a dotted suffix (e.g.
    # "{{slot.output}}" or "{{slot.output_slot}}"), confusing this template's
    # "{{output_slot}}" syntax with the block-dependency "{{block_id.output}}"
    # syntax used inside block prompts. Only the leading identifier is used
    # to look up the slot's content; any dotted suffix is ignored.
    return re.sub(r"\{\{(\w[\w-]*)(?:\.\w[\w-]*)?\}\}", replace, plan.assembly_template)
