"""Stage 1: Generate 50 diverse research idea seeds via 5 strategy calls."""

import random
import re
from typing import Optional
import anthropic
from pipeline.cost_tracker import CostTracker


# ─────────────────────────────────────────────────────────
# STRATEGY DEFINITIONS
# ─────────────────────────────────────────────────────────

STRATEGIES = [
    (
        "mechanism_first",
        (
            "Propose 5 research ideas that start from a behavioral or economic "
            "MECHANISM — a formal force (externality, information asymmetry, "
            "signaling equilibrium, coordination failure, misperceived beliefs) — "
            "and find a surprising empirical setting where that mechanism bites "
            "but has never been tested. The mechanism must be stated precisely "
            "enough to generate a proposition with a non-obvious comparative static."
        ),
    ),
    (
        "puzzle_first",
        (
            "Propose 5 research ideas that start from a SURPRISING EMPIRICAL FACT "
            "or real-world pattern that existing theory doesn't explain well. "
            "Work backwards: what theoretical model would you need to rationalize "
            "this fact, and what additional prediction does that model make that "
            "you could test? The puzzle must be stated as a concrete, documented "
            "observation — not a hunch."
        ),
    ),
    (
        "method_first",
        (
            "Propose 5 research ideas where the core contribution is showing that "
            "a new method or approach leads to a novel insight or finding. The "
            "method must be clearly described and its application must be "
            "illustrated with a concrete example. This paper should be about a methodogical insight, which can be cited "
            " a lot by other researchers. "
        ),
    ),
    (
        "cross_pollination",
        (
            "Propose 5 research ideas that take a THEORETICAL RESULT from one "
            "field of economics and show it has a first-order, previously unnoticed "
            "application in a completely different domain. The transplant must "
            "generate a new prediction in the target domain, not just a relabeling. "
            "Example: Akerlof's lemons logic applied to scientific publishing; "
            "Bursztyn et al.'s pluralistic ignorance applied to firm compliance; "
            "auction theory applied to hospital triage."
        ),
    ),
    (
        "contrarian",
        (
            "Propose 5 research ideas that INVERT a standard result in economics "
            "by identifying conditions under which the opposite holds. The inversion "
            "must be theoretically grounded (not just an empirical counterexample) "
            "and must have a clear welfare or policy implication. Examples of the "
            "structure: 'information provision makes markets worse when...', "
            "'competition destroys quality when...', 'more choice reduces welfare "
            "when...'. The key is that the standard result holds generically but "
            "fails in an empirically important special case that you can identify."
        ),
    ),
]


# ─────────────────────────────────────────────────────────
# SHARED INSTRUCTION (goes into every generation call)
# ─────────────────────────────────────────────────────────

SHARED_INSTRUCTION = """
You are generating research ideas for an economics researcher targeting
top-5 journal publication (QJE, AER, Econometrica, REStud, JPE).

━━━ WHAT MAKES A GREAT IDEA ━━━
Think extremely outside the box. The goal is to generate ideas that are surprising and non-obvious, not incremental or obvious extensions of existing work. Don't be afraid to propose ideas that might be risky or fail — the goal is to generate a diverse set of seeds, not to only generate "safe" ideas. They should have top-5 potential.
THe researcher has a specific taste, but be careful not to be constrained by it — they want to be surprised! The research vision is meant to give you a sense of their intellectual world, but don't just generate ideas that fit neatly into that vision. Some of the best ideas will challenge or stretch their current thinking.

PAPERS THE RESEARCHER LOVES (and why):
- "Collective Traps" (Bursztyn et al.): "Gold standard. Takes a novel 
  conceptual move — the outside option for welfare is wrong — and builds 
  the entire paper around it. Theory leads, empirics follow."
- "Melons as Lemons" (Bai): "Theory → experiment → structural estimation, 
  all fitting together seamlessly. Takes a vague question and gives it a 
  precise, testable, theory-grounded answer."
- "From Extreme to Mainstream" (Bursztyn et al.): "The model is the star. 
  It captures a subtle equilibrium effect. This is what theory + sharp 
  experiment looks like at the AER."
- "Market for Lemons" (Akerlof): "One of the most brilliant theoretical 
  points ever. The formalism is minimal. The power is in the idea."
- "Misperceived Social Norms" (Bursztyn et al.): "Pluralistic ignorance as 
  a market friction — that's the deep insight."
- "Competition and Quality in Science" (Gross & Sampat): "Turns the standard 
  'competition is good' intuition on its head. Connects to how competitive 
  structures can be welfare-destroying."
- "Markets Don't Clear Norms" (contrarian seed): "I love the focus on turning 
  around a normally held belief and interacting it with social norms. Great idea!"
- "WTP for Nothing" (contrarian seed): "Fascinating. WTP elicitations are tough 
  and their external validity is not well understood."

PAPERS THE RESEARCHER HATES (and why):
- "School Value Added" (Andrabi et al.): "No conceptual point whatsoever. 
  Mere datacrunching."
- "Leaders in Social Movements": "No real punchline — you could not explain 
  at a party what you are really doing."
- "Populism as Conspiracy Theory": "Political economy models are always wrong 
  and never good enough to make real predictions."
- "Consumer Surplus of Alternative Payment Methods": "A 10-year-old could 
  predict the conclusions."
- "Abusive Relationships": "Throwing methods we know as economists blindly 
  onto a topic without thinking about whether this really makes sense."
- "AI Disclosure" (consumer responses): "No theoretical framework. Disclosure 
  things with not super strong external validity."
- "Anonymous Grading Bias": "Absolutely no conceptual depth; mere datacrunching. 
  We already know discrimination exists."
- "Dialect Matching in Education Between Teachers and Students": "Hard to think of a more irrelevant question."

━━━ THE PATTERN ━━━

What the researcher wants:
1. A SHARP CONCEPTUAL MOVE that changes how we think about a problem — not 
   just a clever framing, but a result. "Standard welfare measure X is biased 
   because of Y" or "the equilibrium has property Z that nobody noticed."
2. THEORY LEADS, EMPIRICS FOLLOW. The theoretical insight generates the 
   empirical design, not the other way around.
3. You could EXPLAIN THE PUNCHLINE AT A DINNER PARTY in two sentences and 
   someone would say "huh, I never thought of that."
4. Connects to REAL MARKETS, REAL POLICY, REAL WELFARE — not academic puzzles 
   that only other economists care about.
5. The identification strategy follows naturally from the theory, it is not 
   the main contribution.

What the researcher does NOT want:
- Pure description / measurement without a conceptual point
- "Applying method X to setting Y" without a new insight
- Pure theory models with no empirical discipline (especially political economy)
- Results that are too obvious once stated
- Papers where the data is the star and the idea is an afterthought
- Research-on-research (unless it has clear external validity)

━━━ OUTPUT FORMAT ━━━

For each idea, provide exactly these fields:

TITLE: A working paper title (specific, evocative, not generic)

QUESTION: The core research question in one sentence.

THEORETICAL CONTRIBUTION: What is the formal result, proposition, or 
conceptual move? State it as precisely as you can: "We show that when 
[condition], [standard result] fails because [mechanism], and instead 
[new prediction] holds." This is the most important field. If you cannot 
state a crisp theoretical contribution, the idea is not ready. (2-4 sentences)

EMPIRICAL DESIGN: What data or experiment would test the key prediction? 
Name specific institutional settings, data sources, or experimental designs. 
The empirical strategy should follow from the theory — it tests the 
mechanism, not just the correlation. (2-3 sentences)

KILLER QUESTION: What is the single strongest objection a skeptical referee 
would raise, and why doesn't it kill the paper? (1-2 sentences)

Be specific. Every idea should be concrete enough that a PhD student could 
write a 2-page proposal from it. Name actual markets, actual institutions, 
actual mechanisms.

Number each idea 1-5.
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
