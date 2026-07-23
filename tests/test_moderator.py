import pytest

from moce.moderator import ModeratorError, generate_plan


VALID_PLAN_JSON = """
{
  "blocks": [
    {"id": "b1", "type": "text", "depends_on": [], "prompt": "write", "output_slot": "s1"}
  ],
  "assembly_template": "{{s1}}"
}
"""


class ScriptedGenerator:
    def __init__(self, responses):
        self._responses = list(responses)

    def generate(self, role, system_prompt, user_prompt, **kw):
        return self._responses.pop(0)


def test_generate_plan_success():
    gen = ScriptedGenerator([VALID_PLAN_JSON])
    plan = generate_plan(gen, "do something")
    assert plan.blocks[0].id == "b1"


def test_generate_plan_strips_code_fences():
    gen = ScriptedGenerator([f"```json\n{VALID_PLAN_JSON}\n```"])
    plan = generate_plan(gen, "do something")
    assert plan.blocks[0].id == "b1"


def test_generate_plan_retries_on_invalid_json():
    gen = ScriptedGenerator(["not json at all", VALID_PLAN_JSON])
    plan = generate_plan(gen, "do something", max_retries=3)
    assert plan.blocks[0].id == "b1"


def test_generate_plan_normalizes_type_synonyms():
    json_synonym_plan = """
    {
      "blocks": [
        {"id": "b1", "type": "json", "depends_on": [], "prompt": "write", "output_slot": "s1"}
      ],
      "assembly_template": "{{s1}}"
    }
    """
    gen = ScriptedGenerator([json_synonym_plan])
    plan = generate_plan(gen, "do something")
    assert plan.blocks[0].type == "structured"


def test_generate_plan_normalizes_depends_on_output_slot_reference():
    """Model referenced another block's output_slot name in depends_on
    instead of that block's id."""
    plan_json = """
    {
      "blocks": [
        {"id": "code1", "type": "code", "depends_on": [], "prompt": "write code", "output_slot": "reverse_string_function"},
        {"id": "text1", "type": "text", "depends_on": ["reverse_string_function"], "prompt": "explain {{code1.output}}", "output_slot": "explanation"}
      ],
      "assembly_template": "{{reverse_string_function}} {{explanation}}"
    }
    """
    gen = ScriptedGenerator([plan_json])
    plan = generate_plan(gen, "do something")
    assert plan.blocks[1].depends_on == ["code1"]


def test_generate_plan_raises_after_max_retries():
    gen = ScriptedGenerator(["nope", "still nope"])
    with pytest.raises(ModeratorError):
        generate_plan(gen, "do something", max_retries=2)


def test_generate_plan_tolerates_trailing_extra_data():
    """Some models emit valid JSON followed by trailing garbage (e.g. a
    repeated/partial object), which otherwise fails as "Extra data"."""
    gen = ScriptedGenerator([VALID_PLAN_JSON + "\n{\"blocks\": [}"])
    plan = generate_plan(gen, "do something")
    assert plan.blocks[0].id == "b1"


def test_generate_plan_wraps_bare_block_array():
    """Some models emit a bare JSON array of blocks instead of the required
    {"blocks": [...], "assembly_template": "..."} object."""
    bare_array_json = """
    [
      {"id": "code1", "type": "code", "depends_on": [], "prompt": "write code", "output_slot": "code_slot"},
      {"id": "text1", "type": "text", "depends_on": ["code1"], "prompt": "explain {{code1.output}}", "output_slot": "text_slot"}
    ]
    """
    gen = ScriptedGenerator([bare_array_json])
    plan = generate_plan(gen, "do something")
    assert [b.id for b in plan.blocks] == ["code1", "text1"]
    assert "{{code_slot}}" in plan.assembly_template
    assert "{{text_slot}}" in plan.assembly_template


def test_generate_plan_injects_missing_dependency_reference():
    """Regression: a block correctly lists a dependency in `depends_on`, but
    the moderator forgot to include the {{block_id.output}} placeholder in
    the prompt text itself. The Plan's prompt text must be corrected to
    include the reference, not just left as-is (the printed plan should
    show the relationship, and the expert should have something to act on)."""
    plan_json = """
    {
      "blocks": [
        {"id": "code1", "type": "code", "depends_on": [], "prompt": "write code", "output_slot": "code_slot"},
        {"id": "text1", "type": "text", "depends_on": ["code1"], "prompt": "Explain what this code does.", "output_slot": "text_slot"}
      ],
      "assembly_template": "{{code_slot}}\\n\\n{{text_slot}}"
    }
    """
    gen = ScriptedGenerator([plan_json])
    plan = generate_plan(gen, "do something")
    text_block = plan.block_map()["text1"]
    assert "{{code1.output}}" in text_block.prompt
