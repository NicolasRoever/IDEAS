"""Stage 1b: Free pre-filter using Gemini. Kills obvious trash before expensive Anthropic calls."""

import re
import google.generativeai as genai


CRITERIA = [
    "Does the idea have a stated theoretical result (not just a framing)?",
    "Does the idea make sense? Or is there a big logical error in there somewhere?",
    "Does the empirical design follow from the theory (vs. bolted on)?",
    "Is this NOT just 'apply framework X to setting Y'?",
    "Does it speak to a real market/policy (not just academia)?",
]

CRITERIA_BLOCK = "\n".join(f"{i+1}. {c}" for i, c in enumerate(CRITERIA))

PROMPT_TEMPLATE = """Evaluate this economics research idea against 5 binary criteria.

TITLE: {title}
QUESTION: {question}
THEORETICAL CONTRIBUTION: {insight}
EMPIRICAL DESIGN: {identification}

Criteria:
{criteria}

Respond with exactly 5 lines. Each line: the criterion number, then YES or NO. Nothing else.
Example format:
1. YES
2. NO
3. YES
4. YES
5. NO"""


def _parse_answers(text: str) -> tuple[list[bool], int]:
    """Extract YES/NO answers and return (answers, score)."""
    answers = []
    for i in range(1, 6):
        match = re.search(rf"^{i}\.\s*(YES|NO)", text, re.IGNORECASE | re.MULTILINE)
        answers.append(match is not None and match.group(1).upper() == "YES")
    return answers, sum(answers)


def filter_seeds(seeds: list[dict], config: dict, verbose: bool = True) -> list[dict]:
    """Score every seed on 5 binary criteria; attach scores; return all seeds annotated."""
    api_key_env = config.get("google_api_key_env", "GOOGLE_API_KEY")
    import os
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"{api_key_env} not set — needed for free filter (Gemini)")

    genai.configure(api_key=api_key)
    model_name = config["model_routing"]["free_filter"]
    model = genai.GenerativeModel(model_name)
    threshold = config["free_filter"]["score_threshold"]

    annotated = []
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"  Free-filter {seed['id']} ({i+1}/{len(seeds)}): {seed['title'][:60]}...")

        prompt = PROMPT_TEMPLATE.format(
            title=seed["title"],
            question=seed["question"],
            insight=seed.get("insight", ""),
            identification=seed.get("empirical_data", seed.get("identification", "")),
            criteria=CRITERIA_BLOCK,
        )

        response = model.generate_content(prompt)
        answers, score = _parse_answers(response.text)

        annotated.append({
            **seed,
            "filter_score": score,
            "filter_answers": answers,
        })

        if verbose:
            verdict = "PASS" if score > threshold else "KILL"
            print(f"    Score {score}/5 → {verdict}")

    n_pass = sum(1 for s in annotated if s["filter_score"] > threshold)
    if verbose:
        print(f"  {n_pass}/{len(seeds)} seeds passed free filter (score > {threshold})")

    return annotated
