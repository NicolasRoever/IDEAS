"""Stage 1: Generate 50 diverse research idea seeds via 5 strategy calls."""

import random
import re
from typing import Optional
import anthropic
from pipeline.cost_tracker import CostTracker


STRATEGIES = [
    (
        "mechanism_first",
        "Propose 10 research ideas that start from a behavioral or economic mechanism and find a surprising empirical setting to test it.",
    ),
    (
        "puzzle_first",
        "Propose 10 research ideas that start from a surprising empirical fact, pattern, or real-world observation that existing theory doesn't explain well. Work backwards to a research question and identification approach.",
    ),
    (
        "method_first",
        "Propose 10 research ideas built around an elegant identification strategy — a natural experiment, institutional feature, data structure or surprising new method that enables unusually clean causal inference on an important question.",
    ),
    (
        "cross_pollination",
        "Propose 10 research ideas that take an insight or method from one field of economics (or adjacent field: psychology, CS, political science) and apply it to a different domain where it hasn't been used.",
    ),
    (
        "contrarian",
        "Propose 10 research ideas that challenge a widely held assumption in economics, test a 'sacred cow,' or investigate something the field treats as settled but probably shouldn't.",
    ),
]

SHARED_INSTRUCTION = """
For each idea, provide exactly:
- TITLE: A working paper title (specific, not generic)
- QUESTION: The core research question in one sentence
- INSIGHT: What is the key intellectual move or contribution? Why is this not obvious? (2-3 sentences)
- EMPIRICAL DATA: What kind of empirical data/strategy could support the argument (1-2 sentences)

Be specific. "Does X affect Y" is too vague. Name actual institutional contexts, actual data sources where plausible, actual mechanisms. Every idea should be concrete enough that a PhD student could evaluate its feasibility.

Number each idea 1-10.
"""


def _format_papers_for_context(papers: list[dict]) -> str:
    lines = []
    for p in papers:
        lines.append(f"Title: {p['title']}")
        lines.append(f"Abstract: {p['abstract']}")
        lines.append("")
    return "\n".join(lines)


def _parse_seeds(text: str, strategy: str, id_offset: int) -> list[dict]:
    """Parse 10 seeds from a strategy response."""
    seeds = []

    # Split on numbered ideas: look for patterns like "1.", "2.", etc. at start of line or after blank line
    # Try to split on idea boundaries
    idea_pattern = re.compile(
        r"(?:^|\n)\s*(?:\*\*)?(\d+)(?:\*\*)?\.\s*(?:\*\*)?(?:TITLE|Idea)?(?:\*\*)?:?",
        re.MULTILINE,
    )

    # Find all idea boundaries
    matches = list(idea_pattern.finditer(text))

    if len(matches) >= 2:
        # Extract text between boundaries
        idea_blocks = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            idea_blocks.append(text[start:end])
    else:
        # Fallback: split on TITLE: pattern
        title_splits = re.split(r"\n(?=(?:\*\*)?TITLE(?:\*\*)?:)", text)
        idea_blocks = [b for b in title_splits if b.strip()]

    for i, block in enumerate(idea_blocks):
        seed = _parse_single_seed(block, strategy, id_offset + i)
        if seed:
            seeds.append(seed)

    return seeds


def _parse_single_seed(block: str, strategy: str, idx: int) -> Optional[dict]:
    """Extract fields from a single idea block."""
    def extract_field(pattern, text):
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            val = m.group(1).strip()
            # Clean up markdown bold markers
            val = re.sub(r'\*\*', '', val)
            # Trim at next field header
            val = re.split(r'\n(?:TITLE|QUESTION|INSIGHT|EMPIRICAL DATA|IDENTIFICATION):', val, maxsplit=1)[0]
            return val.strip()
        return ""

    title = extract_field(r'TITLE:\s*(.+?)(?=\n(?:QUESTION|INSIGHT|EMPIRICAL DATA)|$)', block)
    question = extract_field(r'QUESTION:\s*(.+?)(?=\n(?:TITLE|INSIGHT|EMPIRICAL DATA)|$)', block)
    insight = extract_field(r'INSIGHT:\s*(.+?)(?=\n(?:TITLE|QUESTION|EMPIRICAL DATA)|$)', block)
    empirical = extract_field(r'EMPIRICAL\s*DATA:\s*(.+?)(?=\n(?:TITLE|QUESTION|INSIGHT)|$)', block)

    if not title:
        return None

    seed_id = f"seed_{idx + 1:03d}"
    return {
        "id": seed_id,
        "strategy": strategy,
        "title": title,
        "question": question,
        "insight": insight,
        "empirical_data": empirical,
    }


def generate_seeds(
    research_vision: str,
    taste_profile: list[dict],
    config: dict,
    tracker: CostTracker,
    verbose: bool = True,
) -> list[dict]:
    """Run 5 strategy calls and return up to 50 seeds."""
    client = anthropic.Anthropic()
    model = config["model_routing"]["generation"]
    temperature = config["generation"]["temperature"]

    # Sample 20 papers for generation context
    sampled = random.sample(taste_profile, min(20, len(taste_profile)))
    papers_context = _format_papers_for_context(sampled)

    all_seeds = []
    id_offset = 0

    for strategy_name, strategy_prompt in STRATEGIES:
        if verbose:
            print(f"  Generating: {strategy_name}...")

        system_msg = f"""You are a creative research idea generator for an economist. Be extremnely bold and imaginative. Propose ideas that are surprising and non-obvious, not incremental or obvious extensions of existing work. Don't be afraid to propose ideas that might be risky or fail — the goal is to generate a diverse set of seeds, not to only generate "safe" ideas. They should have top-5 potential. Here is the economist's research vision and intellectual identity:

{research_vision}

Here are sample papers from their reading list and his reaction to them to understand their intellectual world:

{papers_context}
"""

        user_msg = f"""{strategy_prompt}

{SHARED_INSTRUCTION}"""

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=temperature,
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}],
        )

        tracker.record("generation", model, response.usage.input_tokens, response.usage.output_tokens)
        raw_text = response.content[0].text
        seeds = _parse_seeds(raw_text, strategy_name, id_offset)

        if verbose:
            print(f"    Parsed {len(seeds)} seeds")

        all_seeds.extend(seeds)
        id_offset += len(seeds)

    # Re-assign sequential IDs
    for i, seed in enumerate(all_seeds):
        seed["id"] = f"seed_{i + 1:03d}"

    return all_seeds
