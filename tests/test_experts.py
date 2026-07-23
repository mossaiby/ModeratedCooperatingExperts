import json

from moce.experts import run_block
from moce.schema import Block, BlockResult


class ScriptedGenerator:
    """Test double that returns a scripted sequence of responses for `generate`,
    ignoring which role/prompt was passed."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, role, system_prompt, user_prompt, **kw):
        self.calls.append((role, system_prompt, user_prompt))
        return self._responses.pop(0)


def make_block(**overrides):
    defaults = dict(
        id="b1", type="text", depends_on=[], prompt="write something", output_slot="slot1",
    )
    defaults.update(overrides)
    return Block(**defaults)


def test_text_block_strips_boilerplate():
    gen = ScriptedGenerator(["Sure, here's the answer: Paris is the capital of France."])
    block = make_block(type="text")
    result = run_block(gen, block, {})
    assert result.status == "ok"
    assert result.validated_output == "Paris is the capital of France."


def test_code_block_strips_fences():
    gen = ScriptedGenerator(["```python\nprint('hi')\n```"])
    block = make_block(type="code")
    result = run_block(gen, block, {})
    assert result.status == "ok"
    assert result.validated_output == "print('hi')"


def test_structured_block_validates_json():
    gen = ScriptedGenerator(['{"a": 1, "b": 2}'])
    block = make_block(type="structured")
    result = run_block(gen, block, {})
    assert result.status == "ok"
    assert json.loads(result.validated_output) == {"a": 1, "b": 2}


def test_structured_block_retries_then_succeeds():
    gen = ScriptedGenerator(["not json", '{"ok": true}'])
    block = make_block(type="structured")
    result = run_block(gen, block, {}, max_retries=3)
    assert result.status == "ok"
    assert result.retries == 1
    assert len(gen.calls) == 2


def test_structured_block_fails_after_max_retries():
    gen = ScriptedGenerator(["nope", "still nope"])
    block = make_block(type="structured")
    result = run_block(gen, block, {}, max_retries=2)
    assert result.status == "invalid"
    assert result.retries == 2


def test_image_block_is_stubbed():
    gen = ScriptedGenerator([])
    block = make_block(type="image")
    result = run_block(gen, block, {})
    assert result.status == "invalid"
    assert "not implemented" in result.error_message


def test_dependency_substitution():
    gen = ScriptedGenerator(["derived output"])
    context = {
        "code1": BlockResult(block_id="code1", validated_output="print(1)", status="ok")
    }
    block = make_block(id="b2", prompt="Explain: {{code1.output}}")
    run_block(gen, block, context)
    _, _, user_prompt = gen.calls[0]
    assert "print(1)" in user_prompt
