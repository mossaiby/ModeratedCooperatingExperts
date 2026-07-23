from moce.assembler import assemble
from moce.orchestrator import run_plan
from moce.schema import Block, BlockResult, Plan


class RecordingGenerator:
    """Returns text/code/structured-shaped output based on role so
    experts.run_block's validators succeed, and records call order."""

    def __init__(self):
        self.order = []

    def generate(self, role, system_prompt, user_prompt, **kw):
        self.order.append(role)
        if role == "structured":
            return '{"result": true}'
        if role == "code":
            return "print('hello')"
        return "some text output"


def test_run_plan_respects_dependency_order():
    plan = Plan(
        blocks=[
            Block(id="code1", type="code", prompt="write code", output_slot="code_slot"),
            Block(
                id="text1",
                type="text",
                depends_on=["code1"],
                prompt="explain {{code1.output}}",
                output_slot="text_slot",
            ),
        ],
        assembly_template="{{code_slot}}\n{{text_slot}}",
    )
    gen = RecordingGenerator()
    results = run_plan(gen, plan, max_workers=1)

    assert results["code1"].status == "ok"
    assert results["text1"].status == "ok"
    assert gen.order.index("code") < gen.order.index("text")


def test_assemble_fills_template_and_reports_errors():
    plan = Plan(
        blocks=[
            Block(id="b1", type="text", prompt="p", output_slot="slot1"),
            Block(id="b2", type="text", prompt="p", output_slot="slot2"),
        ],
        assembly_template="A:{{slot1}} B:{{slot2}}",
    )
    results = {
        "b1": BlockResult(block_id="b1", validated_output="hello", status="ok"),
        "b2": BlockResult(block_id="b2", status="invalid", error_message="bad output"),
    }
    doc = assemble(plan, results)
    assert "A:hello" in doc
    assert "ERROR" in doc and "bad output" in doc
