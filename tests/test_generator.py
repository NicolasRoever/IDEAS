"""Tests for generator.py — parsing logic (no LLM calls)."""

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.generator import _parse_seeds, _parse_single_seed, load_directions


class TestParseSingleSeed:
    BLOCK = textwrap.dedent("""
        TITLE: Peer Effects in Debt Repayment
        QUESTION: Does observing a neighbor's debt repayment change own behavior?
        THEORETICAL CONTRIBUTION: Social norms literature predicts contagion; this tests causal direction using staggered rollout.
        EMPIRICAL DESIGN: Administrative microfinance data with village-level variation.
        KILLER QUESTION: Why wouldn't lenders already exploit this?
    """).strip()

    def test_extracts_title(self):
        seed = _parse_single_seed(self.BLOCK, "mechanism_first", 0)
        assert seed["title"] == "Peer Effects in Debt Repayment"

    def test_extracts_question(self):
        seed = _parse_single_seed(self.BLOCK, "mechanism_first", 0)
        assert "neighbor" in seed["question"]

    def test_extracts_insight(self):
        seed = _parse_single_seed(self.BLOCK, "mechanism_first", 0)
        assert "Social norms" in seed["insight"]

    def test_extracts_empirical_data(self):
        seed = _parse_single_seed(self.BLOCK, "mechanism_first", 0)
        assert "microfinance" in seed["empirical_data"]

    def test_extracts_killer_question(self):
        seed = _parse_single_seed(self.BLOCK, "mechanism_first", 0)
        assert "lenders" in seed["killer_question"]

    def test_assigns_strategy(self):
        seed = _parse_single_seed(self.BLOCK, "puzzle_first", 0)
        assert seed["strategy"] == "puzzle_first"

    def test_assigns_id_from_idx(self):
        seed = _parse_single_seed(self.BLOCK, "mechanism_first", 4)
        assert seed["id"] == "seed_005"

    def test_returns_none_when_no_title(self):
        block = "QUESTION: Something?\nTHEORETICAL CONTRIBUTION: Unclear."
        seed = _parse_single_seed(block, "mechanism_first", 0)
        assert seed is None

    def test_missing_killer_question_returns_empty_string(self):
        block = "TITLE: T\nQUESTION: Q\nTHEORETICAL CONTRIBUTION: I\nEMPIRICAL DESIGN: E"
        seed = _parse_single_seed(block, "x", 0)
        assert seed["killer_question"] == ""


class TestParseSeeds:
    def test_parses_multiple_numbered_ideas(self):
        text = textwrap.dedent("""
            1. TITLE: First Idea
            QUESTION: Q1
            THEORETICAL CONTRIBUTION: I1
            EMPIRICAL DESIGN: E1
            KILLER QUESTION: K1

            2. TITLE: Second Idea
            QUESTION: Q2
            THEORETICAL CONTRIBUTION: I2
            EMPIRICAL DESIGN: E2
            KILLER QUESTION: K2
        """).strip()
        seeds = _parse_seeds(text, "mechanism_first", 0)
        assert len(seeds) == 2
        assert seeds[0]["title"] == "First Idea"
        assert seeds[1]["title"] == "Second Idea"

    def test_id_offset_applied(self):
        text = "TITLE: Only One\nQUESTION: Q\nTHEORETICAL CONTRIBUTION: I\nEMPIRICAL DESIGN: E\nKILLER QUESTION: K"
        seeds = _parse_seeds(text, "x", 9)
        assert seeds[0]["id"] == "seed_010"

    def test_skips_blocks_without_title(self):
        text = textwrap.dedent("""
            1. TITLE: Valid Idea
            QUESTION: Q
            THEORETICAL CONTRIBUTION: I
            EMPIRICAL DESIGN: E
            KILLER QUESTION: K

            2. QUESTION: No title here
            THEORETICAL CONTRIBUTION: I2
        """).strip()
        seeds = _parse_seeds(text, "x", 0)
        assert len(seeds) == 1
        assert seeds[0]["title"] == "Valid Idea"

    def test_empty_text_returns_empty_list(self):
        seeds = _parse_seeds("", "x", 0)
        assert seeds == []


class TestLoadDirections:
    def test_parses_multiple_sections(self, tmp_path):
        md = tmp_path / "directions.md"
        md.write_text(
            "## Mechanism-first\nPropose ideas from mechanisms.\n\n"
            "## Puzzle-first\nPropose ideas from puzzles.\n"
        )
        directions = load_directions(md)
        assert len(directions) == 2
        assert directions[0][0] == "Mechanism-first"
        assert "mechanisms" in directions[0][1]
        assert directions[1][0] == "Puzzle-first"

    def test_ignores_empty_sections(self, tmp_path):
        md = tmp_path / "directions.md"
        md.write_text("## EmptySection\n\n## RealSection\nActual content here.\n")
        directions = load_directions(md)
        assert len(directions) == 1
        assert directions[0][0] == "RealSection"

    def test_single_section(self, tmp_path):
        md = tmp_path / "directions.md"
        md.write_text("## Only\nDo this.\n")
        directions = load_directions(md)
        assert len(directions) == 1
        assert directions[0] == ("Only", "Do this.")

    def test_no_sections_returns_empty(self, tmp_path):
        md = tmp_path / "directions.md"
        md.write_text("No headers here at all.\n")
        directions = load_directions(md)
        assert directions == []
