"""Tests for cost_tracker.py."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.cost_tracker import CostTracker


class TestCostTracker:
    def test_empty_tracker_total_cost_is_zero(self):
        tracker = CostTracker()
        assert tracker.total_cost() == 0.0

    def test_record_known_model_computes_cost(self):
        tracker = CostTracker()
        # claude-sonnet-4-6: $3/M input, $15/M output
        tracker.record("judging", "claude-sonnet-4-6", 1_000_000, 1_000_000)
        assert abs(tracker.total_cost() - 18.0) < 0.001

    def test_record_unknown_model_zero_cost(self):
        tracker = CostTracker()
        tracker.record("judging", "unknown-model-xyz", 1_000_000, 1_000_000)
        assert tracker.total_cost() == 0.0

    def test_multiple_records_sum_correctly(self):
        tracker = CostTracker()
        tracker.record("generation", "claude-opus-4-6", 500_000, 100_000)
        tracker.record("judging", "claude-sonnet-4-6", 200_000, 50_000)
        # opus: 500k * 5/M + 100k * 25/M = 2.5 + 2.5 = 5.0
        # sonnet: 200k * 3/M + 50k * 15/M = 0.6 + 0.75 = 1.35
        expected = 5.0 + 1.35
        assert abs(tracker.total_cost() - expected) < 0.001

    def test_by_stage_groups_correctly(self):
        tracker = CostTracker()
        tracker.record("generation", "claude-opus-4-6", 1000, 500)
        tracker.record("generation", "claude-opus-4-6", 1000, 500)
        tracker.record("judging", "claude-sonnet-4-6", 2000, 100)
        stages = tracker.by_stage()
        assert "generation" in stages
        assert "judging" in stages
        assert stages["generation"]["calls"] == 2
        assert stages["judging"]["calls"] == 1

    def test_by_stage_token_totals(self):
        tracker = CostTracker()
        tracker.record("novelty", "claude-sonnet-4-6", 300, 200)
        tracker.record("novelty", "claude-sonnet-4-6", 100, 50)
        stages = tracker.by_stage()
        assert stages["novelty"]["input_tokens"] == 400
        assert stages["novelty"]["output_tokens"] == 250

    def test_summary_contains_total(self):
        tracker = CostTracker()
        tracker.record("judging", "claude-sonnet-4-6", 1000, 500)
        summary = tracker.summary()
        assert "TOTAL" in summary

    def test_summary_contains_stage_name(self):
        tracker = CostTracker()
        tracker.record("generation", "claude-opus-4-6", 1000, 500)
        summary = tracker.summary()
        assert "generation" in summary

    def test_zero_tokens_zero_cost(self):
        tracker = CostTracker()
        tracker.record("judging", "claude-sonnet-4-6", 0, 0)
        assert tracker.total_cost() == 0.0
