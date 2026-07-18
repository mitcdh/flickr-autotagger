"""Command-line interface for safe analysis and explicit Flickr updates."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from .analysis import OpenAIImageAnalyzer
from .config import ConfigurationError, Settings
from .flickr_gateway import FlickrGateway
from .pipeline import AutotaggerPipeline, RunSummary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Flickr metadata with an OpenAI-compatible vision model."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply generated metadata to Flickr. Without this flag the run is a dry-run.",
    )
    parser.add_argument(
        "--apply-plan",
        type=Path,
        metavar="PATH",
        help="Apply analyses from a previous JSON or JSONL plan without calling OpenAI.",
    )
    parser.add_argument("--photoset-id", action="append", help="Process only this photoset ID.")
    parser.add_argument("--limit", type=int, help="Maximum number of new OpenAI analyses.")
    parser.add_argument("--max-cost", type=float, help="Stop after reaching this recorded cost.")
    parser.add_argument("--concurrency", type=int, help="Concurrent OpenAI analyses.")
    parser.add_argument("--prompt-file", type=Path, help="Custom prompt template file.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any photo fails.")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def _apply_overrides(settings: Settings, arguments: argparse.Namespace) -> Settings:
    values = {}
    if arguments.photoset_id:
        values["flickr_photoset_ids"] = tuple(arguments.photoset_id)
    if arguments.limit is not None:
        if arguments.limit < 1:
            raise ConfigurationError("--limit must be greater than zero")
        values["max_photos"] = arguments.limit
    if arguments.max_cost is not None:
        if arguments.max_cost <= 0:
            raise ConfigurationError("--max-cost must be greater than zero")
        values["max_total_cost"] = arguments.max_cost
    if arguments.concurrency is not None:
        if arguments.concurrency < 1:
            raise ConfigurationError("--concurrency must be greater than zero")
        values["analysis_concurrency"] = arguments.concurrency
    if arguments.prompt_file is not None:
        values["prompt_file"] = arguments.prompt_file
    if arguments.no_resume:
        values["resume"] = False
    if arguments.strict:
        values["strict"] = True
    updated = replace(settings, **values)
    updated.validate(require_openai=not bool(arguments.apply_plan), require_flickr=True)
    return updated


def _report(summary: RunSummary, output_file: Path, *, dry_run: bool) -> None:
    mode = "Dry-run complete" if dry_run else "Run complete"
    print(
        f"{mode}: analyzed={summary.analyzed}, updated={summary.updated}, "
        f"skipped={summary.skipped}, failed={summary.failed}, "
        f"OpenAI cost=${summary.total_cost:.4f}"
    )
    if summary.skip_reasons:
        reasons = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(
                summary.skip_reasons.items(), key=lambda item: (-item[1], item[0])
            )
        )
        print(f"Skipped by reason: {reasons}")
    print(f"Run report saved to {output_file}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if arguments.apply and arguments.apply_plan:
        parser.error("--apply and --apply-plan cannot be used together")
    try:
        settings = _apply_overrides(Settings.from_env(), arguments)
        flickr = FlickrGateway(settings)
        analyzer = None if arguments.apply_plan else OpenAIImageAnalyzer(settings)
        pipeline = AutotaggerPipeline(settings, flickr, analyzer)
        if arguments.apply_plan:
            summary = pipeline.apply_plan(arguments.apply_plan)
            dry_run = False
        else:
            summary = asyncio.run(pipeline.run(apply=arguments.apply))
            dry_run = not arguments.apply
        _report(summary, settings.updated_metadata_file, dry_run=dry_run)
        return 2 if settings.strict and summary.failed else 0
    except (ConfigurationError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
