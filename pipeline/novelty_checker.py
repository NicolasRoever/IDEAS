"""Stage 3: Check surviving seeds against existing literature via OpenAlex."""

import re
import time
import anthropic
import requests


OPENALEX_BASE = "https://api.openalex.org/works"


def _generate_queries(seed: dict, client: anthropic.Anthropic, model: str) -> list[str]:
    """Use LLM to generate 2-3 short keyword search queries for a seed."""
    prompt = f"""Given this research idea, generate 2-3 short keyword search queries to find related existing papers in economics.

TITLE: {seed['title']}
QUESTION: {seed['question']}
INSIGHT: {seed['insight']}

Return only the queries, one per line. Each query should be 3-6 words. No numbering, no explanation."""

    response = client.messages.create(
        model=model,
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    queries = [line.strip() for line in raw.splitlines() if line.strip()]
    return queries[:3]


def _search_openalex(query: str, per_page: int = 5, mailto: str = "research@example.com") -> list[dict]:
    """Search OpenAlex and return list of {title, abstract} dicts."""
    params = {
        "search": query,
        "per-page": per_page,
        "select": "title,abstract_inverted_index",
        "mailto": mailto,
    }

    for attempt in range(3):
        try:
            resp = requests.get(OPENALEX_BASE, params=params, timeout=15)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException:
            if attempt == 2:
                return []
            time.sleep(1)

    papers = []
    for work in data.get("results", []):
        title = work.get("title", "")
        # Reconstruct abstract from inverted index
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index") or {})
        if title:
            papers.append({"title": title, "abstract": abstract})

    return papers


def _reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(word for _, word in word_positions)


def _deduplicate_papers(paper_lists: list[list[dict]]) -> list[dict]:
    """Flatten and deduplicate papers by title."""
    seen = set()
    unique = []
    for papers in paper_lists:
        for p in papers:
            key = p["title"].lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(p)
    return unique


def _format_retrieved_papers(papers: list[dict]) -> str:
    lines = []
    for i, p in enumerate(papers, 1):
        lines.append(f"{i}. {p['title']}")
        if p["abstract"]:
            lines.append(f"   Abstract: {p['abstract'][:300]}...")
        lines.append("")
    return "\n".join(lines)


def _assess_novelty(
    seed: dict,
    retrieved_papers: list[dict],
    client: anthropic.Anthropic,
    model: str,
) -> tuple[str, str, str]:
    """LLM novelty assessment. Returns (verdict, nearest_paper, distinction)."""
    seed_text = (
        f"TITLE: {seed['title']}\n"
        f"QUESTION: {seed['question']}\n"
        f"INSIGHT: {seed['insight']}\n"
        f"IDENTIFICATION: {seed.get('identification', '')}"
    )

    papers_text = _format_retrieved_papers(retrieved_papers)

    prompt = f"""Here is a research idea:
{seed_text}

Here are the most similar existing papers found in the literature:
{papers_text}

Is this idea substantively novel?
Respond with:
- VERDICT: "novel" / "partially_novel" / "already_done"
- NEAREST_PAPER: Title and authors of the closest existing paper
- DISTINCTION: If partially novel or novel, what is the key difference? (1-2 sentences)
- IF_ALREADY_DONE: Which paper does this duplicate and why? (only if verdict is already_done)"""

    response = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text

    verdict_match = re.search(r'VERDICT:\s*["\']?(novel|partially_novel|already_done)["\']?', raw, re.IGNORECASE)
    verdict = verdict_match.group(1).lower() if verdict_match else "novel"

    nearest_match = re.search(r'NEAREST_PAPER:\s*(.+?)(?=\n(?:DISTINCTION|IF_ALREADY_DONE|VERDICT)|$)', raw, re.IGNORECASE | re.DOTALL)
    nearest_paper = nearest_match.group(1).strip() if nearest_match else ""

    if verdict == "already_done":
        done_match = re.search(r'IF_ALREADY_DONE:\s*(.+?)(?=\n(?:VERDICT|NEAREST_PAPER|DISTINCTION)|$)', raw, re.IGNORECASE | re.DOTALL)
        distinction = done_match.group(1).strip() if done_match else ""
    else:
        dist_match = re.search(r'DISTINCTION:\s*(.+?)(?=\n(?:VERDICT|NEAREST_PAPER|IF_ALREADY_DONE)|$)', raw, re.IGNORECASE | re.DOTALL)
        distinction = dist_match.group(1).strip() if dist_match else ""

    return verdict, nearest_paper, distinction


def check_novelty(
    seeds: list[dict],
    config: dict,
    verbose: bool = True,
) -> list[dict]:
    """Check novelty of each seed; return filtered, sorted final list."""
    client = anthropic.Anthropic()
    model = config["model"]
    papers_per_query = config["novelty"]["papers_per_query"]
    top_n = config["output"]["top_n"]

    final_seeds = []

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"  Checking novelty {seed['id']} ({i+1}/{len(seeds)}): {seed['title'][:60]}...")

        # Step 1: Generate search queries
        queries = _generate_queries(seed, client, model)
        if verbose:
            print(f"    Queries: {queries}")

        # Step 2: Search OpenAlex for each query
        paper_lists = []
        for q in queries:
            papers = _search_openalex(q, per_page=papers_per_query)
            paper_lists.append(papers)
            time.sleep(0.1)  # polite rate limiting

        retrieved = _deduplicate_papers(paper_lists)
        if verbose:
            print(f"    Retrieved {len(retrieved)} unique papers")

        # Step 3: LLM novelty assessment
        verdict, nearest_paper, distinction = _assess_novelty(seed, retrieved, client, model)

        if verbose:
            print(f"    Verdict: {verdict}")

        if verdict == "already_done":
            continue

        final_seed = {
            **seed,
            "novelty_verdict": verdict,
            "nearest_paper": nearest_paper,
            "novelty_distinction": distinction,
        }
        final_seeds.append(final_seed)

    # Sort by taste score, keep top N
    final_seeds.sort(key=lambda x: x["taste_score"], reverse=True)
    final_seeds = final_seeds[:top_n]

    if verbose:
        print(f"  {len(final_seeds)} seeds passed novelty check (top {top_n} selected)")

    return final_seeds
