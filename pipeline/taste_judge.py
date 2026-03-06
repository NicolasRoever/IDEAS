"""Stage 2: Score each seed against the taste profile (paper_reactions.md)."""

import re
import anthropic
from pipeline.cost_tracker import CostTracker


def _format_seed(seed: dict) -> str:
    lines = [
        f"TITLE: {seed['title']}",
        f"QUESTION: {seed['question']}",
        f"INSIGHT: {seed['insight']}",
        f"EMPIRICAL DESIGN: {seed.get('empirical_data', seed.get('identification', ''))}",
    ]
    if seed.get("killer_question"):
        lines.append(f"KILLER QUESTION: {seed['killer_question']}")
    return "\n".join(lines)


def _parse_judgment(text: str) -> tuple[int, str, str]:
    """Extract PROBABILITY, REASONING, RISK from judgment response."""
    probability = 0
    reasoning = ""
    risk = ""

    prob_match = re.search(r'PROBABILITY:\s*(\d+)', text, re.IGNORECASE)
    if prob_match:
        probability = min(100, max(0, int(prob_match.group(1))))

    reasoning_match = re.search(
        r'REASONING:\s*(.+?)(?=\n(?:RISK|PROBABILITY)|$)', text, re.IGNORECASE | re.DOTALL
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    risk_match = re.search(
        r'RISK:\s*(.+?)(?=\n(?:REASONING|PROBABILITY)|$)', text, re.IGNORECASE | re.DOTALL
    )
    if risk_match:
        risk = risk_match.group(1).strip()

    return probability, reasoning, risk


def judge_seeds(
    seeds: list[dict],
    research_vision: str,
    paper_reactions: str,
    config: dict,
    tracker: CostTracker,
    verbose: bool = True,
) -> list[dict]:
    """Score each seed individually; return filtered, sorted survivors."""
    client = anthropic.Anthropic()
    model = config["model_routing"]["judging"]
    score_threshold = config["judging"]["score_threshold"]
    max_survivors = config["judging"]["max_survivors"]

    system_msg = f"""You are a research taste evaluator for a specific economist.
Your job: given a research idea seed, estimate the probability (0-100) that this
researcher would actually pursue this paper — meaning both that they find it
personally exciting AND that it has realistic top-5 journal potential (AER, QJE,
JPE, REStud, Econometrica).

Here is their detailed taste profile, derived from reactions to ~30 papers:
{paper_reactions}"""

    scored_seeds = []

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"  Judging {seed['id']} ({i+1}/{len(seeds)}): {seed['title'][:60]}...")

        seed_text = _format_seed(seed)

        user_msg = f"""Evaluate this research idea seed:

{seed_text}

Respond with exactly:
PROBABILITY: [0-100 integer]
REASONING: [2-3 sentences: why this researcher specifically would or wouldn't pursue this. Reference their taste profile. Address both personal taste and top-5 potential.]
RISK: [The single biggest reason this idea would not make it — either the researcher wouldn't find it exciting enough, or it's not top-5 material.]"""

        response = client.messages.create(
            model=model,
            max_tokens=512,
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}],
        )

        tracker.record("judging", model, response.usage.input_tokens, response.usage.output_tokens)
        raw_text = response.content[0].text
        probability, reasoning, risk = _parse_judgment(raw_text)

        if verbose:
            status = "PASS" if probability >= score_threshold else "FAIL"
            print(f"    Probability: {probability} → {status}")

        scored_seed = {
            "id": seed["id"],
            "strategy": seed["strategy"],
            "title": seed["title"],
            "question": seed["question"],
            "insight": seed["insight"],
            "identification": seed.get("empirical_data", seed.get("identification", "")),
            "taste_probability": probability,
            "taste_reasoning": reasoning,
            "taste_risk": risk,
        }
        scored_seeds.append(scored_seed)

    scored_seeds.sort(key=lambda x: x["taste_probability"], reverse=True)

    if verbose:
        n_pass = sum(1 for s in scored_seeds if s["taste_probability"] >= score_threshold)
        print(f"  {n_pass}/{len(seeds)} seeds passed taste filter (probability >= {score_threshold})")

    return scored_seeds
