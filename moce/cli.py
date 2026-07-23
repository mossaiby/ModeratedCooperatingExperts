"""CLI entrypoint for MoCE."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from moce.assembler import assemble
from moce.model_manager import ModelManager, configure_model_logging
from moce.moderator import ModeratorError, generate_plan
from moce.orchestrator import run_plan

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"


@click.group()
@click.option("--verbose", is_flag=True, help="Enable verbose logging and intermediate output.")
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug diagnostics: raw model outputs, DEBUG-level logging, and "
    "un-silenced transformers/huggingface_hub logging. Implies --verbose.",
)
@click.pass_context
def main(ctx: click.Context, verbose: bool, debug: bool) -> None:
    """Moderated Cooperating Experts CLI."""
    ctx.ensure_object(dict)
    verbose = verbose or debug
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug
    logging.basicConfig(
        level=logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )
    configure_model_logging(verbose=verbose, debug=debug)


@main.command()
@click.argument("user_prompt")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_CONFIG_PATH,
    show_default=True,
    help="Path to the model configuration YAML file.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Only generate and print the moderator's plan; skip expert execution.",
)
@click.option(
    "--max-workers",
    type=int,
    default=1,
    show_default=True,
    help="Max concurrent expert threads per dependency generation.",
)
@click.option(
    "--verbose",
    "verbose_flag",
    is_flag=True,
    help="Also print the moderator's plan and each block's output.",
)
@click.option(
    "--show-plan",
    is_flag=True,
    help="Also print the moderator's plan, without the rest of --verbose's output.",
)
@click.option(
    "--debug",
    "debug_flag",
    is_flag=True,
    help="Enable debug diagnostics: raw model outputs, DEBUG-level logging, and "
    "un-silenced transformers/huggingface_hub logging. Implies --verbose.",
)
@click.pass_context
def run(
    ctx: click.Context,
    user_prompt: str,
    config_path: Path,
    dry_run: bool,
    max_workers: int,
    verbose_flag: bool,
    show_plan: bool,
    debug_flag: bool,
) -> None:
    """Run the full moderator -> experts -> assembler pipeline for USER_PROMPT."""
    debug = ctx.obj.get("debug", False) or debug_flag
    verbose = ctx.obj.get("verbose", False) or verbose_flag or debug
    logging.getLogger().setLevel(logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING))
    configure_model_logging(verbose=verbose, debug=debug)
    manager = ModelManager.from_yaml(config_path)

    try:
        plan = generate_plan(manager, user_prompt)
    except ModeratorError as exc:
        click.echo(f"Moderator failed: {exc}", err=True)
        sys.exit(1)

    if verbose or dry_run or show_plan:
        click.echo("=== Plan ===")
        click.echo(json.dumps(plan.model_dump(), indent=2))

    if dry_run:
        return

    results = run_plan(manager, plan, max_workers=max_workers)

    if verbose:
        click.echo("\n=== Block Results ===")
        for block_id, result in results.items():
            click.echo(f"--- {block_id} ({result.status}) ---")
            click.echo(result.validated_output or result.error_message or "")
            if debug and result.raw_output:
                click.echo(f"[raw output, {result.retries} retries]")
                click.echo(result.raw_output)

    click.echo("\n=== Final Document ===")
    click.echo(assemble(plan, results))


if __name__ == "__main__":
    main()
