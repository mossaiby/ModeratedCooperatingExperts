"""Core data models for MoCE: Block, Plan, BlockResult.

The moderator LLM produces a `Plan` (as JSON) describing an ordered set of
`Block`s with a dependency graph. Each block is later filled in by a
type-specific expert model, producing a `BlockResult`.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

BlockType = Literal["text", "code", "structured", "image"]
BlockStatus = Literal["ok", "invalid", "error"]

# JSON schema fragment shown to the moderator LLM so it knows the exact shape
# of output we expect back. Kept here (rather than duplicated in prompts) so
# it always matches the real Pydantic models.
PLAN_JSON_SCHEMA_HINT = """
{
  "blocks": [
    {
      "id": "string, unique identifier for this block",
      "type": "one of: text, code, structured, image",
      "depends_on": ["list of block ids this block depends on, may be empty"],
      "prompt": "the exact instructions for the expert that will fill this block; may reference other blocks' output via {{block_id.output}}",
      "output_slot": "string name of the placeholder in assembly_template this block fills",
      "constraints": "optional extra constraints string, or null"
    }
  ],
  "assembly_template": "string containing {{output_slot}} placeholders for every block's output_slot, defining the final document layout"
}
""".strip()


class Block(BaseModel):
    id: str = Field(min_length=1)
    type: BlockType
    depends_on: list[str] = Field(default_factory=list)
    prompt: str = Field(min_length=1)
    output_slot: str = Field(min_length=1)
    constraints: Optional[str] = None

    @field_validator("depends_on")
    @classmethod
    def _no_self_dependency(cls, v: list[str], info) -> list[str]:
        block_id = info.data.get("id")
        if block_id is not None and block_id in v:
            raise ValueError(f"block '{block_id}' cannot depend on itself")
        return v


class Plan(BaseModel):
    blocks: list[Block] = Field(min_length=1)
    assembly_template: str = Field(min_length=1)

    @field_validator("blocks")
    @classmethod
    def _unique_ids(cls, v: list[Block]) -> list[Block]:
        ids = [b.id for b in v]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate block ids: {sorted(dupes)}")
        return v

    def block_map(self) -> dict[str, Block]:
        return {b.id: b for b in self.blocks}


class BlockResult(BaseModel):
    block_id: str
    raw_output: str = ""
    validated_output: str = ""
    status: BlockStatus = "error"
    retries: int = 0
    error_message: Optional[str] = None
