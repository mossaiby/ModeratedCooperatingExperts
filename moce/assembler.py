"""Assembler: substitutes each block's validated output into the plan's
assembly template to produce the final document."""
from __future__ import annotations

import re

from moce.schema import BlockResult, Plan


def assemble(plan: Plan, results: dict[str, BlockResult]) -> str:
    """Fill `plan.assembly_template`'s `{{output_slot}}` placeholders with the
    corresponding block's validated output (or an error marker if the block
    failed)."""
    slot_to_content: dict[str, str] = {}
    for block in plan.blocks:
        result = results.get(block.id)
        if result is None:
            slot_to_content[block.output_slot] = f"[ERROR: block '{block.id}' was never executed]"
        elif result.status == "ok":
            slot_to_content[block.output_slot] = result.validated_output
        else:
            slot_to_content[block.output_slot] = (
                f"[ERROR: block '{block.id}' failed validation: {result.error_message}]"
            )

    def replace(match: re.Match) -> str:
        slot = match.group(1)
        return slot_to_content.get(slot, match.group(0))

    return re.sub(r"\{\{(\w[\w-]*)\}\}", replace, plan.assembly_template)
