# Research Idea Generator — Pipeline Spec

## Overview

A three-stage pipeline that generates novel economics research proposal seeds, filters them through a personalized taste profile, and verifies novelty against the existing literature. Output: 10 ranked idea seeds ready for manual selection and full proposal development.

## Directory Structure

```
IDEAS/
├── inputs/
│   ├── taste_profile.json       # 30 papers: title, abstract, reaction
│   └── research_vision.md             # 1-page research identity doc (what you care about, what bores you, your style)
├── pipeline/
│   ├── generator.py             # Stage 1: idea generation
│   ├── taste_judge.py           # Stage 2: taste-based scoring
│   ├── novelty_checker.py       # Stage 3: literature novelty check
│   └── run.py                   # Orchestrator: runs all stages in sequence
├── output/
│   └── batch_YYYY-MM-DD/
│       ├── seeds_raw.json       # Stage 1 output
│       ├── seeds_scored.json    # Stage 2 output
│       ├── seeds_final.json     # Stage 3 output (top 10)
│       └── report.md            # Human-readable summary of top 10
└── config.yaml                  # Model name, API keys, number of seeds, thresholds
```

## Input Formats

### taste_profile.json

```json
[
  {
    "id": "paper id",
    "title": "Paper Title",
    "abstract": "abstract text...",
    "reaction": "Free-form 3-8 sentences. What excites or bores you about this. Be honest and specific.",
  }
]
```

### research_vision.md

~1 page, freeform. Should cover:
- What makes a research question important vs. merely answerable
- What identification strategies you find elegant vs. hacky
- What types of contributions you value (mechanism tests, new facts, methodological, conceptual)
- What bores you
- Your research identity in 2-3 sentences

### config.yaml

```yaml
model: "claude-sonnet-4-20250514"  # swap as models improve
api_key_env: "ANTHROPIC_API_KEY"
semantic_scholar_api_key_env: "S2_API_KEY"  # optional, higher rate limits

generation:
  num_seeds: 50                    # raw seeds to generate
  seeds_per_call: 10               # seeds per LLM call (5 calls total)
  temperature: 1.0                 # high temperature = more creative variance

judging:
  score_threshold: 3               # kill everything below this
  max_survivors: 20                # pass at most this many to novelty check

novelty:
  papers_per_query: 5              # similar papers to retrieve per seed
  api: "semantic_scholar"          # or "openalex"

output:
  top_n: 10                        # final number of ideas to present
```

---

## Stage 1: Idea Generation (`generator.py`)

### Purpose
Generate 50 diverse research idea seeds. Maximize creative breadth.

### Context window contents
1. `research_vision.md` (full text)
2. Random draw of 20 papers from `taste_profile.json` 
3. Strategy-specific instruction (see below)

### Why only top 20 papers, not all 30?
Generation should be steered lightly. The full 30 papers are for judging. Here we just want the LLM to understand your intellectual identity, not overfit to your ratings.

### Generation strategies
Run 5 calls, 10 seeds each. Each call uses a different strategy prompt to ensure diversity:

**Call 1 — Mechanism-first:**
"Propose 10 research ideas that start from a behavioral or economic mechanism and find a surprising empirical setting to test it. The setting should be one where standard theory and behavioral theory make different predictions."

**Call 2 — Puzzle-first:**
"Propose 10 research ideas that start from a surprising empirical fact, pattern, or real-world observation that existing theory doesn't explain well. Work backwards to a research question and identification approach."

**Call 3 — Method-first:**
"Propose 10 research ideas built around an elegant identification strategy — a natural experiment, institutional feature, or data structure that enables unusually clean causal inference on an important question."

**Call 4 — Cross-pollination:**
"Propose 10 research ideas that take an insight or method from one field of economics (or adjacent field: psychology, CS, political science) and apply it to a different domain where it hasn't been used."

**Call 5 — Contrarian:**
"Propose 10 research ideas that challenge a widely held assumption in economics, test a 'sacred cow,' or investigate something the field treats as settled but probably shouldn't."

### Shared instruction appended to all calls

```
For each idea, provide exactly:
- TITLE: A working paper title (specific, not generic)
- QUESTION: The core research question in one sentence
- INSIGHT: What is the key intellectual move or contribution? Why is this not obvious? (2-3 sentences)
- EMPIRICAL DATA: What kind of empirical data/strategy could support the argument (1-2 sentences)

Be specific. "Does X affect Y" is too vague. Name actual institutional contexts, actual data sources where plausible, actual mechanisms. Every idea should be concrete enough that a PhD student could evaluate its feasibility.
```

### Output format (seeds_raw.json)

```json
[
  {
    "id": "seed_001",
    "strategy": "mechanism_first",
    "title": "...",
    "question": "...",
    "insight": "...",
    "empirical_data": "..."
  }
]
```

---

## Stage 2: Taste-Based Judging (`taste_judge.py`)

### Purpose
Score each seed against your full taste profile. Kill ideas you wouldn't find exciting.

### Context window contents
1. `research_vision.md` (full text)
2. ALL 30 papers from `taste_profile.json`, each as: title + abstract + idea_score + reaction + spark_or_killer
3. The seed to evaluate

### Important: evaluate seeds individually
Do NOT batch seeds. Each seed gets its own LLM call. Batching causes relative ranking within the batch rather than evaluation against your absolute taste.

### Prompt

```
You are evaluating a research idea seed for a specific economist. Your job is to predict whether they would find this idea exciting enough to pursue.

Here is their research vision:
{research_vision}

Here are 30 papers they have rated, with their unfiltered reactions:
{taste_profile_formatted}

Now evaluate this idea seed:
{seed}

Respond with:
- SCORE: 1-5 (1 = they would find this boring/generic, 5 = they would be genuinely excited)
- REASONING: 2-3 sentences explaining why this researcher specifically would or wouldn't like this idea. Reference their taste profile where relevant.
- RISK: The single biggest reason this idea might not work or might not be interesting.
```

### Processing
- Parse score, reasoning, risk for each seed
- Sort by score descending
- Kill everything below `score_threshold` (default: 3)
- Keep top `max_survivors` (default: 20)
- Write to `seeds_scored.json`

### Output format (seeds_scored.json)

```json
[
  {
    "id": "seed_001",
    "strategy": "mechanism_first",
    "title": "...",
    "question": "...",
    "insight": "...",
    "taste_score": 4,
    "taste_reasoning": "...",
    "taste_risk": "..."
  }
]
```

---

## Stage 3: Novelty Check (`novelty_checker.py`)

### Purpose
Verify that surviving ideas haven't already been done. Kill duplicates, flag partial overlaps.

### Process per seed

1. **Extract search queries.** Use LLM to generate 2-3 short keyword queries from the seed (e.g., "AI access timing student learning" and "generation effect educational technology"). This is better than hand-crafting queries because the LLM understands the conceptual core.

2. **Query Semantic Scholar / OpenAlex.** For each query, retrieve top 5 papers (title + abstract). Deduplicate across queries. You end up with 5-10 unique papers per seed.

3. **LLM novelty assessment.** Feed the seed + retrieved papers to the LLM:

```
Here is a research idea:
{seed}

Here are the most similar existing papers found in the literature:
{retrieved_papers}

Is this idea substantively novel?
Respond with:
- VERDICT: "novel" / "partially_novel" / "already_done"
- NEAREST_PAPER: Title and authors of the closest existing paper
- DISTINCTION: If partially novel or novel, what is the key difference? (1-2 sentences)
- IF ALREADY_DONE: Which paper does this duplicate and why?
```

### Processing
- Kill all "already_done" seeds
- Keep "novel" and "partially_novel"
- Sort by taste_score from Stage 2
- Take top 10
- Write to `seeds_final.json`

### Output format (seeds_final.json)

```json
[
  {
    "id": "seed_001",
    "strategy": "mechanism_first",
    "title": "...",
    "question": "...",
    "insight": "...",
    "identification": "...",
    "taste_score": 4,
    "taste_reasoning": "...",
    "taste_risk": "...",
    "novelty_verdict": "novel",
    "nearest_paper": "Author (2024), Title",
    "novelty_distinction": "..."
  }
]
```

---

## Stage 4: Report Generation

### Purpose
Produce a human-readable markdown summary of the top 10 ideas.

### Format (report.md)

For each idea, ordered by taste score:

```
### #1: [Title]
**Question:** ...
**Key insight:** ...
**Identification:** ...
**Why this fits you:** [taste_reasoning]
**Biggest risk:** [taste_risk]
**Novelty:** [verdict] — nearest paper: [nearest_paper]. Distinction: [novelty_distinction]
---
```

---

## Orchestrator (`run.py`)

```
1. Load config, taste profile, manifesto
2. Run generator (5 calls) → seeds_raw.json
3. Run taste judge (1 call per seed, ~50 calls) → seeds_scored.json
4. Run novelty checker (1 LLM call + 2-3 API calls per surviving seed) → seeds_final.json
5. Generate report.md
6. Print summary: "Generated 50 seeds → 20 passed taste filter → N passed novelty → top 10 in report.md"
```

### Cost estimate per run
- Stage 1: 5 LLM calls (~moderate context)
- Stage 2: ~50 LLM calls (large context due to full taste profile, but short output)
- Stage 3: ~20 LLM calls (moderate context) + ~60 Semantic Scholar queries
- Report: 1 call
- Total: ~76 LLM calls. With Claude Sonnet: roughly $1-3 per run depending on context sizes.

---

## Scaling Up (Future Iterations)

When you want higher quality, add these in order of impact:

1. **Expand taste profile.** Add papers over time, especially ideas you generated and rejected (with reasons). Your Stage 4 selections become training data.
2. **Add adversarial elaboration.** After novelty check, a separate LLM call tries to kill each surviving idea ("steelman why this won't work"). Cheap to add, meaningful quality boost.
3. **Multi-model generation.** Run Stage 1 with Claude AND GPT AND Gemini. Different models have different creative biases. Pool the seeds before Stage 2.
4. **Literature-aware generation.** Once the pipeline works, you can optionally feed recent abstracts into specific generation calls — not all of them, just one "gap detection" strategy that builds on recent work.
5. **Full proposal writer.** After you select ideas from the top 10, a final stage writes the full 2-page proposal.
6. **Feedback loop.** Track which seeds you select, which you reject and why. Periodically retrain the taste judge on this accumulated data.