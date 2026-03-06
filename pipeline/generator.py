"""Stage 1: Generate diverse research idea seeds via per-direction calls."""

import re
from pathlib import Path
from typing import Optional
import anthropic
from pipeline.cost_tracker import CostTracker


SHARED_INSTRUCTION = (
    "Think extremely outside the box. The goal is to generate ideas that are "
    "surprising and non-obvious, not incremental or obvious extensions of existing work. "
    "Don't be afraid to propose ideas that might be risky or fail — the goal is to generate "
    "a diverse set of seeds, not to only generate 'safe' ideas. They should have top-5 "
    "journal potential. Be careful not to be constrained by the researcher's vision — "
    "they want to be surprised. Some of the best ideas will challenge or stretch their "
    "current thinking.\n\n"
)


def load_directions(path: Path) -> list[tuple[str, str]]:
    """Parse directions.md into a list of (name, prompt) tuples.

    Each direction starts with a ## header (the name) followed by the prompt body.
    """
    text = path.read_text()
    directions = []
    blocks = re.split(r"^##\s+", text, flags=re.MULTILINE)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        first_newline = block.find("\n")
        if first_newline == -1:
            continue
        name = block[:first_newline].strip()
        prompt = block[first_newline:].strip()
        if name and prompt:
            directions.append((name, prompt))
    return directions


def _parse_seeds(text: str, strategy: str, id_offset: int) -> list[dict]:
    """Parse seeds from a strategy response."""
    # Split on numbered idea boundaries
    idea_pattern = re.compile(
        r"(?:^|\n)\s*(?:\*\*)?(\d+)(?:\*\*)?\.\s*(?:\*\*)?(?:TITLE|Idea)?(?:\*\*)?:?",
        re.MULTILINE,
    )
    matches = list(idea_pattern.finditer(text))

    if len(matches) >= 2:
        idea_blocks = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            idea_blocks.append(text[start:end])
    else:
        title_splits = re.split(r"\n(?=(?:\*\*)?TITLE(?:\*\*)?:)", text)
        idea_blocks = [b for b in title_splits if b.strip()]

    seeds = []
    for i, block in enumerate(idea_blocks):
        seed = _parse_single_seed(block, strategy, id_offset + i)
        if seed:
            seeds.append(seed)
    return seeds


def _parse_single_seed(block: str, strategy: str, idx: int) -> Optional[dict]:
    """Extract fields from a single idea block."""
    all_headers = (
        r"TITLE|QUESTION|THEORETICAL CONTRIBUTION|EMPIRICAL DESIGN|KILLER QUESTION"
    )

    def extract_field(label: str) -> str:
        pattern = rf"{label}:\s*(.+?)(?=\n\s*(?:{all_headers}):|$)"
        m = re.search(pattern, block, re.IGNORECASE | re.DOTALL)
        if m:
            val = m.group(1).strip()
            val = re.sub(r"\*\*", "", val)
            return val.strip()
        return ""

    title = extract_field("TITLE")
    if not title:
        return None

    return {
        "id": f"seed_{idx + 1:03d}",
        "strategy": strategy,
        "title": title,
        "question": extract_field("QUESTION"),
        "insight": extract_field("THEORETICAL CONTRIBUTION"),
        "empirical_data": extract_field("EMPIRICAL DESIGN"),
        "killer_question": extract_field("KILLER QUESTION"),
    }


def generate_seeds(
    research_vision: str,
    directions: list[tuple[str, str]],
    config: dict,
    tracker: CostTracker,
    verbose: bool = True,
) -> list[dict]:
    """Run one LLM call per direction and return all seeds."""
    client = anthropic.Anthropic()
    model = config["model_routing"]["generation"]
    temperature = config["generation"]["temperature"]

    all_seeds = []
    id_offset = 0

    for strategy_name, strategy_prompt in directions:
        if verbose:
            print(f"  Generating: {strategy_name}...")

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=temperature,
            system=research_vision,
            messages=[{"role": "user", "content": SHARED_INSTRUCTION + strategy_prompt}],
        )

        tracker.record(
            "generation", model,
            response.usage.input_tokens, response.usage.output_tokens,
        )

        seeds = _parse_seeds(response.content[0].text, strategy_name, id_offset)

        if verbose:
            print(f"    Parsed {len(seeds)} seeds")

        all_seeds.extend(seeds)
        id_offset += len(seeds)

    # Re-assign clean sequential IDs
    for i, seed in enumerate(all_seeds):
        seed["id"] = f"seed_{i + 1:03d}"

    return all_seeds
