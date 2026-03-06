"""Tests for taste_judge.py."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.taste_judge import _parse_judgment, _format_seed


class TestParseJudgment:
    def test_parses_probability_reasoning_risk(self):
        text = "PROBABILITY: 72\nREASONING: Fits well.\nRISK: Hard to identify."
        prob, reasoning, risk = _parse_judgment(text)
        assert prob == 72
        assert "Fits well" in reasoning
        assert "Hard to identify" in risk

    def test_probability_clamped_to_100(self):
        text = "PROBABILITY: 150\nREASONING: Great.\nRISK: None."
        prob, _, _ = _parse_judgment(text)
        assert prob == 100

    def test_probability_clamped_to_0(self):
        text = "PROBABILITY: -5\nREASONING: Boring.\nRISK: Everything."
        prob, _, _ = _parse_judgment(text)
        assert prob == 0

    def test_missing_probability_returns_0(self):
        text = "REASONING: Interesting.\nRISK: Data."
        prob, _, _ = _parse_judgment(text)
        assert prob == 0

    def test_probability_case_insensitive(self):
        text = "probability: 55\nREASONING: ok\nRISK: none"
        prob, _, _ = _parse_judgment(text)
        assert prob == 55

    def test_multiline_reasoning_captured(self):
        text = "PROBABILITY: 60\nREASONING: This is line one.\nThis is line two.\nRISK: Identification."
        _, reasoning, _ = _parse_judgment(text)
        assert "line one" in reasoning
        assert "line two" in reasoning

    def test_empty_string_returns_defaults(self):
        prob, reasoning, risk = _parse_judgment("")
        assert prob == 0
        assert reasoning == ""
        assert risk == ""

    def test_probability_boundary_0(self):
        text = "PROBABILITY: 0\nREASONING: Boring.\nRISK: Boring."
        prob, _, _ = _parse_judgment(text)
        assert prob == 0

    def test_probability_boundary_100(self):
        text = "PROBABILITY: 100\nREASONING: Perfect.\nRISK: None."
        prob, _, _ = _parse_judgment(text)
        assert prob == 100


class TestFormatSeed:
    BASE_SEED = {
        "id": "seed_001",
        "strategy": "mechanism_first",
        "title": "Do Nudges Persist?",
        "question": "Does a one-time nudge have lasting effects?",
        "insight": "Standard models predict decay; this tests whether salience locks in behavior.",
        "empirical_data": "Admin data from a pension reform natural experiment.",
    }

    def test_contains_title(self):
        result = _format_seed(self.BASE_SEED)
        assert "Do Nudges Persist?" in result

    def test_contains_question(self):
        result = _format_seed(self.BASE_SEED)
        assert "Does a one-time nudge" in result

    def test_uses_empirical_design_label(self):
        result = _format_seed(self.BASE_SEED)
        assert "EMPIRICAL DESIGN:" in result

    def test_no_killer_question_field_when_absent(self):
        result = _format_seed(self.BASE_SEED)
        assert "KILLER QUESTION" not in result

    def test_includes_killer_question_when_present(self):
        seed = {**self.BASE_SEED, "killer_question": "Why wouldn't this decay?"}
        result = _format_seed(seed)
        assert "KILLER QUESTION:" in result
        assert "Why wouldn't this decay?" in result

    def test_falls_back_to_identification_field(self):
        seed = {
            "id": "seed_002",
            "strategy": "puzzle_first",
            "title": "Title",
            "question": "Question",
            "insight": "Insight",
            "identification": "Regression discontinuity around cutoff.",
        }
        result = _format_seed(seed)
        assert "Regression discontinuity" in result

    def test_prefers_empirical_data_over_identification(self):
        seed = {
            **self.BASE_SEED,
            "identification": "Should not appear",
        }
        result = _format_seed(seed)
        assert "Admin data" in result
        assert "Should not appear" not in result
