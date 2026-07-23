"""Orchestrator: schedules block execution according to the plan's dependency
DAG, running each "generation" of independent, ready blocks concurrently.

True parallelism is limited by how many models can be resident in GPU memory
at once (see `ModelManager`'s LRU cache); the thread pool size is
configurable so users on a single consumer GPU can set it to 1 to avoid
thrashing model loads, while users with more VRAM (or CPU-only setups) can
increase it for real concurrency.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from moce.dag import topological_generations
from moce.experts import run_block
from moce.schema import BlockResult, Plan

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 1


def run_plan(
    generator: Any,
    plan: Plan,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, BlockResult]:
    """Execute every block in `plan`, respecting dependency order, and return
    a mapping of block_id -> BlockResult."""
    generations = topological_generations(plan)
    block_map = plan.block_map()
    results: dict[str, BlockResult] = {}

    for generation in generations:
        blocks = [block_map[block_id] for block_id in generation]
        logger.info("Running generation with %d block(s): %s", len(blocks), generation)

        if max_workers <= 1 or len(blocks) == 1:
            for block in blocks:
                results[block.id] = run_block(generator, block, results)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(run_block, generator, block, dict(results)): block.id
                    for block in blocks
                }
                for future, block_id in futures.items():
                    results[block_id] = future.result()

    return results
