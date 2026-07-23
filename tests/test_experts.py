import json
from pathlib import Path

from moce.experts import run_block
from moce.schema import Block, BlockResult


class ScriptedGenerator:
    """Test double that returns a scripted sequence of responses for `generate`,
    ignoring which role/prompt was passed."""

    def __init__(self, responses, image_error=None):
        self._responses = list(responses)
        self.calls = []
        self.image_calls = []
        self._image_error = image_error

    def generate(self, role, system_prompt, user_prompt, **kw):
        self.calls.append((role, system_prompt, user_prompt))
        return self._responses.pop(0)

    def generate_image(self, role, prompt, output_path, **kw):
        self.image_calls.append((role, prompt, output_path))
        if self._image_error is not None:
            raise self._image_error
        return output_path


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


def test_code_block_discards_trailing_prose_after_stray_fence():
    """Regression: small models sometimes emit raw code (no opening fence),
    then a single stray closing "```", then a prose explanation paragraph —
    the explanation must not leak into the code block's output."""
    raw = (
        "#include <iostream>\n"
        "int main() { std::cout << \"hi\"; return 0; }\n"
        "```\n"
        "\n"
        "This code prints hi to the console."
    )
    gen = ScriptedGenerator([raw])
    block = make_block(type="code", language="cpp")
    result = run_block(gen, block, {})
    assert result.status == "ok"
    assert "This code prints" not in result.validated_output
    assert "#include <iostream>" in result.validated_output


def test_code_block_extracts_fenced_block_not_at_start():
    """Regression: model emits some leading whitespace/prose before the
    fence and prose after the closing fence; only the fenced content should
    be kept."""
    raw = "Sure!\n```python\nprint('hi')\n```\nThis prints hi."
    gen = ScriptedGenerator([raw])
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


def test_image_block_generates_and_returns_markdown_reference():
    gen = ScriptedGenerator([])
    block = make_block(type="image", prompt="a red fox")
    result = run_block(gen, block, {})
    expected_path = str(Path("moce_output/images/b1.png"))
    assert result.status == "ok"
    assert result.validated_output == f"![b1]({expected_path})"
    assert gen.image_calls[0][1] == "a red fox"


def test_image_block_reports_generation_failure():
    gen = ScriptedGenerator([], image_error=RuntimeError("out of memory"))
    block = make_block(type="image")
    result = run_block(gen, block, {})
    assert result.status == "invalid"
    assert "out of memory" in result.error_message


def test_dependency_substitution():
    gen = ScriptedGenerator(["derived output"])
    context = {
        "code1": BlockResult(block_id="code1", validated_output="print(1)", status="ok")
    }
    block = make_block(id="b2", prompt="Explain: {{code1.output}}")
    run_block(gen, block, context)
    _, _, user_prompt = gen.calls[0]
    assert "print(1)" in user_prompt
