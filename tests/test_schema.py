import pytest
from pydantic import ValidationError

from moce.dag import DagError, topological_generations, validate_dag
from moce.schema import Block, Plan


def make_block(**overrides):
    defaults = dict(
        id="b1",
        type="text",
        depends_on=[],
        prompt="write something",
        output_slot="slot1",
    )
    defaults.update(overrides)
    return Block(**defaults)


def test_block_self_dependency_rejected():
    with pytest.raises(ValidationError):
        make_block(id="b1", depends_on=["b1"])


def test_plan_duplicate_ids_rejected():
    with pytest.raises(ValidationError):
        Plan(
            blocks=[make_block(id="b1"), make_block(id="b1")],
            assembly_template="{{slot1}}",
        )


def test_plan_valid():
    plan = Plan(
        blocks=[make_block(id="b1"), make_block(id="b2", depends_on=["b1"], output_slot="slot2")],
        assembly_template="{{slot1}} {{slot2}}",
    )
    assert plan.block_map()["b2"].depends_on == ["b1"]


def test_dag_unknown_dependency_rejected():
    plan = Plan(
        blocks=[make_block(id="b1", depends_on=["missing"])],
        assembly_template="{{slot1}}",
    )
    with pytest.raises(DagError):
        validate_dag(plan)


def test_dag_cycle_rejected():
    plan = Plan(
        blocks=[
            make_block(id="b1", depends_on=["b2"]),
            make_block(id="b2", depends_on=["b1"], output_slot="slot2"),
        ],
        assembly_template="{{slot1}} {{slot2}}",
    )
    with pytest.raises(DagError):
        validate_dag(plan)


def test_topological_generations_order():
    plan = Plan(
        blocks=[
            make_block(id="b1"),
            make_block(id="b2", depends_on=["b1"], output_slot="slot2"),
            make_block(id="b3", output_slot="slot3"),
        ],
        assembly_template="{{slot1}} {{slot2}} {{slot3}}",
    )
    generations = topological_generations(plan)
    assert {"b1", "b3"} == set(generations[0])
    assert generations[1] == ["b2"]
