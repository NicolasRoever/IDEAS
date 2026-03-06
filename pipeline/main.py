"""Orchestrator: runs all pipeline stages in sequence."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.generator import generate_seeds, load_directions
from pipeline.taste_judge import judge_seeds
from pipeline.novelty_checker import check_novelty
from pipeline.cost_tracker import CostTracker


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def load_research_vision(path: Path) -> str:
    return path.read_text()


def write_json(data, path: Path):
    path.write_text(json.dumps(data, indent=2))
    print(f"  Wrote {path}")


def next_run_dir(output_root: Path) -> tuple[Path, str]:
    """Return (run_dir, run_id) for the next sequential run, e.g. run_0001."""
    output_root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(p.name.split("_")[1])
        for p in output_root.iterdir()
        if p.is_dir() and p.name.startswith("run_") and p.name.split("_")[1].isdigit()
    ]
    next_id = max(existing, default=0) + 1
    run_id = f"{next_id:04d}"
    return output_root / f"run_{run_id}", run_id


def generate_report(seeds: list[dict]) -> str:
    lines = ["# Research Idea Seeds — Top 10\n"]
    for i, seed in enumerate(seeds, 1):
        verdict = seed.get("novelty_verdict", "")
        nearest = seed.get("nearest_paper", "")
        distinction = seed.get("novelty_distinction", "")
        novelty_line = f"{verdict}"
        if nearest:
            novelty_line += f" — nearest paper: {nearest}"
        if distinction:
            novelty_line += f". {distinction}"

        lines.append(f"### #{i}: {seed['title']}")
        lines.append(f"**Question:** {seed['question']}")
        lines.append(f"**Key insight:** {seed['insight']}")
        lines.append(f"**Identification:** {seed.get('identification', '')}")
        lines.append(f"**Fit score:** {seed.get('taste_probability', '')}/100")
        lines.append(f"**Why this fits you:** {seed.get('taste_reasoning', '')}")
        lines.append(f"**Biggest risk:** {seed.get('taste_risk', '')}")
        lines.append(f"**Novelty:** {novelty_line}")
        lines.append("---\n")

    return "\n".join(lines)


def write_summary(
    path: Path,
    run_id: str,
    config: dict,
    started_at: datetime,
    finished_at: datetime,
    stage_stats: dict,
    tracker: CostTracker,
) -> None:
    duration_seconds = (finished_at - started_at).total_seconds()
    summary = {
        "run_id": run_id,
        "version": config.get("version", "unknown"),
        "date": started_at.strftime("%Y-%m-%d"),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round(duration_seconds, 1),
        "config": config,
        "stage_stats": stage_stats,
        "cost": {
            "by_stage": tracker.by_stage(),
            "total_usd": round(tracker.total_cost(), 6),
        },
    }
    path.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {path}")


def main():
    load_dotenv(ROOT / ".env")

    config_path = ROOT / "config.yaml"
    config = load_config(config_path)

    anthropic_key = os.environ.get(config["api_key_env"])
    if not anthropic_key:
        print(f"ERROR: {config['api_key_env']} not set in environment or .env")
        sys.exit(1)
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key

    research_vision = load_research_vision(ROOT / "inputs" / "research_vision.md")
    paper_reactions = (ROOT / "inputs" / "paper_reactions.md").read_text()
    directions = load_directions(ROOT / "inputs" / "directions.md")

    run_dir, run_id = next_run_dir(ROOT / "output")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run ID: {run_id}  →  {run_dir}\n")

    tracker = CostTracker()
    started_at = datetime.now(timezone.utc)

    # Stage 1: Generate seeds
    print("=== Stage 1: Generating seeds ===")
    raw_seeds = generate_seeds(research_vision, directions, config, tracker)
    write_json(raw_seeds, run_dir / "seeds_raw.json")
    print(f"Generated {len(raw_seeds)} seeds\n")

    # Stage 2: Taste judging
    print("=== Stage 2: Taste judging ===")
    scored_seeds = judge_seeds(raw_seeds, research_vision, paper_reactions, config, tracker)
    write_json(scored_seeds, run_dir / "seeds_scored.json")
    score_threshold = config["judging"]["score_threshold"]
    max_survivors = config["judging"]["max_survivors"]
    survivors = [s for s in scored_seeds if s["taste_probability"] >= score_threshold][:max_survivors]
    print(f"Passed taste filter: {len(survivors)}/{len(scored_seeds)} seeds\n")

    # Stage 3: Novelty check
    print("=== Stage 3: Novelty check ===")
    final_seeds = check_novelty(survivors, config, tracker)
    write_json(final_seeds, run_dir / "seeds_final.json")
    print(f"Passed novelty check: {len(final_seeds)} seeds\n")

    # Report
    print("=== Generating report ===")
    (run_dir / "report.md").write_text(generate_report(final_seeds))
    print(f"  Wrote {run_dir / 'report.md'}")

    # Summary
    finished_at = datetime.now(timezone.utc)
    stage_stats = {
        "seeds_generated": len(raw_seeds),
        "seeds_passed_taste": len(survivors),
        "seeds_passed_novelty": len(final_seeds),
        "top_n": min(len(final_seeds), config["output"]["top_n"]),
    }
    write_summary(run_dir / "summary.json", run_id, config, started_at, finished_at, stage_stats, tracker)

    print(
        f"\nGenerated {len(raw_seeds)} seeds → "
        f"{len(survivors)} passed taste filter (≥{score_threshold}) → "
        f"{len(final_seeds)} passed novelty → "
        f"top {stage_stats['top_n']} in report.md"
    )
    print(f"\n{tracker.summary()}")


if __name__ == "__main__":
    main()
