"""Tests for novelty_checker.py — unit tests + end-to-end OpenAlex integration test."""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.novelty_checker import (
    _reconstruct_abstract,
    _deduplicate_papers,
    _format_retrieved_papers,
    _search_openalex,
)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestReconstructAbstract:
    def test_empty_dict_returns_empty_string(self):
        assert _reconstruct_abstract({}) == ""

    def test_none_returns_empty_string(self):
        assert _reconstruct_abstract(None) == ""

    def test_single_word(self):
        assert _reconstruct_abstract({"hello": [0]}) == "hello"

    def test_multiple_words_sorted_by_position(self):
        index = {"world": [1], "hello": [0]}
        assert _reconstruct_abstract(index) == "hello world"

    def test_word_appearing_at_multiple_positions(self):
        index = {"the": [0, 3], "cat": [1], "sat": [2]}
        result = _reconstruct_abstract(index)
        assert result == "the cat sat the"

    def test_realistic_snippet(self):
        index = {"We": [0], "study": [1], "incentives": [2]}
        assert _reconstruct_abstract(index) == "We study incentives"


class TestDeduplicatePapers:
    def test_empty_input(self):
        assert _deduplicate_papers([]) == []

    def test_single_list_no_duplicates(self):
        papers = [{"title": "Paper A", "abstract": ""}, {"title": "Paper B", "abstract": ""}]
        result = _deduplicate_papers([papers])
        assert len(result) == 2

    def test_exact_duplicates_across_lists(self):
        p = {"title": "Same Paper", "abstract": "text"}
        result = _deduplicate_papers([[p], [p]])
        assert len(result) == 1

    def test_case_insensitive_dedup(self):
        a = {"title": "Behavioural Economics", "abstract": ""}
        b = {"title": "behavioural economics", "abstract": ""}
        result = _deduplicate_papers([[a], [b]])
        assert len(result) == 1

    def test_preserves_order_first_occurrence(self):
        p1 = {"title": "First", "abstract": ""}
        p2 = {"title": "Second", "abstract": ""}
        p3 = {"title": "First", "abstract": "different"}
        result = _deduplicate_papers([[p1, p2], [p3]])
        assert result[0]["title"] == "First"
        assert result[1]["title"] == "Second"

    def test_whitespace_stripped_for_dedup(self):
        a = {"title": "  Paper A  ", "abstract": ""}
        b = {"title": "Paper A", "abstract": ""}
        result = _deduplicate_papers([[a], [b]])
        assert len(result) == 1


class TestFormatRetrievedPapers:
    def test_empty_list(self):
        assert _format_retrieved_papers([]) == ""

    def test_paper_without_abstract(self):
        papers = [{"title": "A Study", "abstract": ""}]
        result = _format_retrieved_papers(papers)
        assert "1. A Study" in result
        assert "Abstract:" not in result

    def test_paper_with_abstract(self):
        papers = [{"title": "A Study", "abstract": "Some text here"}]
        result = _format_retrieved_papers(papers)
        assert "Abstract:" in result

    def test_abstract_truncated_at_300_chars(self):
        long_abstract = "x" * 400
        papers = [{"title": "T", "abstract": long_abstract}]
        result = _format_retrieved_papers(papers)
        # Should contain truncation marker
        assert "..." in result
        # The abstract portion should not exceed 300 chars + ellipsis
        abstract_line = [l for l in result.splitlines() if "Abstract:" in l][0]
        assert len(abstract_line) <= len("   Abstract: ") + 300 + 3 + 10  # some slack

    def test_numbering_increments(self):
        papers = [
            {"title": "First", "abstract": ""},
            {"title": "Second", "abstract": ""},
        ]
        result = _format_retrieved_papers(papers)
        assert "1. First" in result
        assert "2. Second" in result


# ---------------------------------------------------------------------------
# End-to-end OpenAlex API test (makes a real HTTP request)
# ---------------------------------------------------------------------------

class TestSearchOpenAlexEndToEnd:
    """Live integration tests against the OpenAlex API.

    These tests hit the real API. They verify that:
    - The API responds successfully
    - The response structure matches what our code expects
    - Abstract reconstruction works on real data
    """

    def test_basic_query_returns_results(self):
        """A well-known topic should return at least one result."""
        results = _search_openalex("behavioral economics nudge", per_page=3)
        assert isinstance(results, list)
        assert len(results) > 0, "Expected results for 'behavioral economics nudge'"

    def test_result_structure(self):
        """Each result must have 'title' and 'abstract' keys."""
        results = _search_openalex("minimum wage employment", per_page=3)
        assert len(results) > 0
        for paper in results:
            assert "title" in paper, "Missing 'title' key"
            assert "abstract" in paper, "Missing 'abstract' key"
            assert isinstance(paper["title"], str)
            assert isinstance(paper["abstract"], str)

    def test_title_is_non_empty(self):
        """Titles must be non-empty strings (our code filters on this)."""
        results = _search_openalex("social media welfare externalities", per_page=3)
        for paper in results:
            assert paper["title"].strip() != "", "Got an empty title"

    def test_per_page_limit_respected(self):
        """Response should not exceed requested per_page count."""
        results = _search_openalex("price elasticity demand", per_page=3)
        assert len(results) <= 3

    def test_abstract_reconstruction_produces_readable_text(self):
        """At least some papers should have non-empty abstracts we can reconstruct."""
        results = _search_openalex("moral hazard insurance", per_page=5)
        abstracts = [p["abstract"] for p in results if p["abstract"]]
        assert len(abstracts) > 0, "Expected at least one paper with a non-empty abstract"
        # Reconstructed abstract should look like words, not garbage
        sample = abstracts[0]
        words = sample.split()
        assert len(words) >= 5, f"Abstract too short to be real text: {sample!r}"

    def test_obscure_query_returns_list(self):
        """Even a narrow query should return a list (possibly empty), not raise."""
        results = _search_openalex("xyzzy quux frobnicate economics 12345", per_page=3)
        assert isinstance(results, list)

    def test_known_paper_appears_for_specific_query(self):
        """Searching for a famous paper's key terms should surface related work."""
        results = _search_openalex("Card Krueger minimum wage New Jersey", per_page=5)
        assert len(results) > 0
        titles = " ".join(p["title"].lower() for p in results)
        # Should find something about minimum wage
        assert "minimum wage" in titles or "employment" in titles, (
            f"Expected minimum-wage papers, got titles: {[p['title'] for p in results]}"
        )
