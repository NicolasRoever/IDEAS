"""Orchestrator: runs all pipeline stages in sequence."""

import json
import os
import sys
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Allow running as: python pipeline/run.py from IDEAS/ root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.generator import generate_seeds
from pipeline.taste_judge import judge_seeds
from pipeline.novelty_checker import check_novelty


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_taste_profile(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_research_vision(path: Path) -> str:
    with open(path) as f:
        return f.read()


def write_json(data, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Wrote {path}")


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
        lines.append(f"**Why this fits you:** {seed.get('taste_reasoning', '')}")
        lines.append(f"**Biggest risk:** {seed.get('taste_risk', '')}")
        lines.append(f"**Novelty:** {novelty_line}")
        lines.append("---\n")

    return "\n".join(lines)


def main():
    load_dotenv(ROOT / ".env")

    # Set API keys from env vars
    config_path = ROOT / "config.yaml"
    config = load_config(config_path)

    anthropic_key = os.environ.get(config["api_key_env"])
    if not anthropic_key:
        print(f"ERROR: {config['api_key_env']} not set in environment or .env")
        sys.exit(1)
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key

    # Load inputs
    taste_profile = load_taste_profile(ROOT / "inputs" / "taste_profile.json")
    research_vision = load_research_vision(ROOT / "inputs" / "research_vision.md")

    # Create output directory
    batch_dir = ROOT / "output" / f"batch_{date.today().isoformat()}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {batch_dir}\n")

    # Stage 1: Generate seeds
    print("=== Stage 1: Generating seeds ===")
    raw_seeds = generate_seeds(research_vision, taste_profile, config)
    write_json(raw_seeds, batch_dir / "seeds_raw.json")
    print(f"Generated {len(raw_seeds)} seeds\n")

    # Stage 2: Taste judging
    print("=== Stage 2: Taste judging ===")
    scored_seeds = judge_seeds(raw_seeds, research_vision, taste_profile, config)
    write_json(scored_seeds, batch_dir / "seeds_scored.json")
    print(f"Passed taste filter: {len(scored_seeds)} seeds\n")

    # Stage 3: Novelty check
    print("=== Stage 3: Novelty check ===")
    final_seeds = check_novelty(scored_seeds, config)
    write_json(final_seeds, batch_dir / "seeds_final.json")
    print(f"Passed novelty check: {len(final_seeds)} seeds\n")

    # Generate report
    print("=== Generating report ===")
    report_text = generate_report(final_seeds)
    report_path = batch_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"  Wrote {report_path}")

    # Summary
    print(
        f"\nGenerated {len(raw_seeds)} seeds → "
        f"{len(scored_seeds)} passed taste filter → "
        f"{len(final_seeds)} passed novelty → "
        f"top {min(len(final_seeds), config['output']['top_n'])} in report.md"
    )


if __name__ == "__main__":
    main()
