"""Stage 2: Score each seed against the full taste profile."""

import re
import anthropic
from pipeline.cost_tracker import CostTracker


def _format_taste_profile(taste_profile: list[dict]) -> str:
    lines = []
    for p in taste_profile:
        lines.append(f"[{p['id']}] {p['title']}")
        lines.append(f"Abstract: {p['abstract']}")
        lines.append(f"Reaction: {p['reaction']}")
        lines.append("")
    return "\n".join(lines)


def _format_seed(seed: dict) -> str:
    return (
        f"TITLE: {seed['title']}\n"
        f"QUESTION: {seed['question']}\n"
        f"INSIGHT: {seed['insight']}\n"
        f"EMPIRICAL DATA: {seed.get('empirical_data', seed.get('identification', ''))}"
    )


def _parse_judgment(text: str) -> tuple[int, str, str]:
    """Extract SCORE, REASONING, RISK from judgment response."""
    score = 0
    reasoning = ""
    risk = ""

    score_match = re.search(r'SCORE:\s*(\d)', text, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))

    reasoning_match = re.search(
        r'REASONING:\s*(.+?)(?=\n(?:RISK|SCORE)|$)', text, re.IGNORECASE | re.DOTALL
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    risk_match = re.search(
        r'RISK:\s*(.+?)(?=\n(?:REASONING|SCORE)|$)', text, re.IGNORECASE | re.DOTALL
    )
    if risk_match:
        risk = risk_match.group(1).strip()

    return score, reasoning, risk


def judge_seeds(
    seeds: list[dict],
    research_vision: str,
    taste_profile: list[dict],
    config: dict,
    tracker: CostTracker,
    verbose: bool = True,
) -> list[dict]:
    """Score each seed individually; return filtered, sorted survivors."""
    client = anthropic.Anthropic()
    model = config["model_routing"]["judging"]
    score_threshold = config["judging"]["score_threshold"]
    max_survivors = config["judging"]["max_survivors"]

    taste_profile_text = _format_taste_profile(taste_profile)

    system_msg = f"""You are evaluating research idea seeds for a specific economist. Your job is to predict whether they would find an idea exciting enough to pursue.

Here is their research vision:
{research_vision}

Here are 30+ papers they have rated, with their unfiltered reactions:
{taste_profile_text}"""

    scored_seeds = []

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"  Judging {seed['id']} ({i+1}/{len(seeds)}): {seed['title'][:60]}...")

        seed_text = _format_seed(seed)

        user_msg = f"""Now evaluate this idea seed:

{seed_text}

Respond with:
- SCORE: 1-5 (1 = they would find this boring/generic, 5 = they would be genuinely excited)
- REASONING: 2-3 sentences explaining why this researcher specifically would or wouldn't like this idea. Reference their taste profile where relevant.
- RISK: The single biggest reason this idea might not work or might not be interesting."""

        response = client.messages.create(
            model=model,
            max_tokens=512,
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}],
        )

        tracker.record("judging", model, response.usage.input_tokens, response.usage.output_tokens)
        raw_text = response.content[0].text
        score, reasoning, risk = _parse_judgment(raw_text)

        scored_seed = {
            "id": seed["id"],
            "strategy": seed["strategy"],
            "title": seed["title"],
            "question": seed["question"],
            "insight": seed["insight"],
            "identification": seed.get("empirical_data", seed.get("identification", "")),
            "taste_score": score,
            "taste_reasoning": reasoning,
            "taste_risk": risk,
        }
        scored_seeds.append(scored_seed)

    # Filter and sort
    survivors = [s for s in scored_seeds if s["taste_score"] >= score_threshold]
    survivors.sort(key=lambda x: x["taste_score"], reverse=True)
    survivors = survivors[:max_survivors]

    if verbose:
        print(f"  {len(survivors)}/{len(seeds)} seeds passed taste filter (score >= {score_threshold})")

    return survivors
