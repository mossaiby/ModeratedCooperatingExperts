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


def test_generate_plan_raises_after_max_retries():
    gen = ScriptedGenerator(["nope", "still nope"])
    with pytest.raises(ModeratorError):
        generate_plan(gen, "do something", max_retries=2)
