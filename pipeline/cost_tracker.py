"""Tracks token usage and estimated cost across all pipeline stages."""

from dataclasses import dataclass, field

# Prices in USD per million tokens. Update here when Anthropic changes pricing.
PRICE_PER_MILLION: dict[str, dict[str, float]] = {
    "claude-opus-4-6":   {"input": 5, "output": 25.0},
    "claude-sonnet-4-6": {"input":  3.0, "output": 15.0},
}


@dataclass
class CostTracker:
    records: list = field(default_factory=list)

    def record(self, stage: str, model: str, input_tokens: int, output_tokens: int) -> None:
        prices = PRICE_PER_MILLION.get(model, {"input": 0.0, "output": 0.0})
        cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
        self.records.append({
            "stage": stage,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
        })

    def total_cost(self) -> float:
        return sum(r["cost_usd"] for r in self.records)

    def by_stage(self) -> dict:
        stages: dict = {}
        for r in self.records:
            s = r["stage"]
            if s not in stages:
                stages[s] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "model": r["model"]}
            stages[s]["calls"] += 1
            stages[s]["input_tokens"] += r["input_tokens"]
            stages[s]["output_tokens"] += r["output_tokens"]
            stages[s]["cost_usd"] += r["cost_usd"]
        return stages

    def summary(self) -> str:
        lines = ["Cost breakdown:"]
        for stage, data in self.by_stage().items():
            lines.append(
                f"  {stage} ({data['model']}): "
                f"{data['calls']} calls, "
                f"{data['input_tokens']:,} in / {data['output_tokens']:,} out tokens "
                f"— ${data['cost_usd']:.4f}"
            )
        lines.append(f"  TOTAL: ${self.total_cost():.4f}")
        return "\n".join(lines)
