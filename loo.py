"""Nested leave-one-out evaluation; no pseudo-target enters its training pool."""

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import main as model


PRACTICE_INPUT = Path(
    "/home/hatch/workspace/makefun/projects/gta6-twitch-ccu-forecast-01/kit/"
    "gta6-twitch-ccu-forecast-01-kit/input"
)


@dataclass
class Scores:
    losses: list[float] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)
    covered: int = 0

    def add(self, rows: list[dict], actuals: dict[int, float]) -> None:
        for row in rows:
            hour = row["hour"]
            if hour not in actuals:
                continue
            actual = math.log(actuals[hour])
            logs = [math.log(value) for value in row["q"]]
            for tau, prediction in zip(model.LEVELS, logs):
                d = actual - prediction
                self.losses.append(2 * max(tau * d, (tau - 1) * d))
            self.covered += logs[0] <= actual <= logs[-1]
            self.errors.append(abs(actual - logs[2]))

    def merge(self, other: "Scores") -> None:
        self.losses.extend(other.losses)
        self.errors.extend(other.errors)
        self.covered += other.covered

    def report(self, label: str, targets: int) -> str:
        if not self.errors:
            return f"{label}: pseudo-targets={targets}, scored-hours=0, no valid observations"
        return (
            f"{label}: pseudo-targets={targets}, scored-hours={len(self.errors)}, "
            f"loss={math.fsum(self.losses) / len(self.losses):.6f}, "
            f"coverage90={self.covered / len(self.errors):.2%}, "
            f"median_abs_log_error={median(self.errors):.6f}"
        )


def pseudo_case(curve: model.ComparableCurve, pool: list[dict]) -> dict:
    """Use the real target's notes parser, including both metadata text fields."""
    return {
        "day_start": curve.day_start.isoformat(),
        "known": [{"hour": 1, "viewers": curve.viewers[1]}],
        "notes": model.comparable_notes(curve),
        "comparables": pool,
    }


def house_baseline(log_first: float) -> list[dict]:
    return [
        {"hour": hour, "q": model.quantiles(log_first, 0.15 * math.sqrt(hour - 1))}
        for hour in model.HOURS
    ]


def diagnostics(name: str, case: dict) -> str:
    result = model.forecast_details(case)
    first = case["known"][0]["viewers"]
    ratios = {row["hour"]: row["q"][2] / first for row in result.hours}
    peak = max(model.HOURS, key=ratios.__getitem__)
    selected_ratios = ", ".join(
        f"h{hour}={ratios[hour]:.6f}" for hour in (6, 9, 24)
    )
    selected_sigma = ", ".join(
        f"h{hour}={result.sigma[hour]:.6f}" for hour in (2, 6, 12, 24)
    )
    return (
        f"{name}: peak_hour={peak}, peak_ratio={ratios[peak]:.6f}; "
        f"median_ratios[{selected_ratios}]; sigma[{selected_sigma}]; k={result.k:.6f}"
    )


def evaluate(input_dir: Path) -> None:
    paths = sorted(path for path in input_dir.glob("*.in") if path.is_file())
    if not paths:
        raise ValueError(f"No .in cases found in {input_dir}")
    pooled = Scores()
    baseline = Scores()
    total_targets = 0
    cases = []
    print("Nested LOO: each pseudo-target is excluded from shape fitting and spread calibration.")
    print("Loss is mean log pinball loss over every valid hour and quantile.")
    for path in paths:
        case = json.loads(path.read_text(encoding="utf-8"))
        cases.append((path.stem, case))
        comparables = case.get("comparables", [])
        scores = Scores()
        targets = 0
        for index, comparable in enumerate(comparables):
            curve = model.extract_curve(comparable)
            if curve is None or not curve.ratios:
                continue
            # Calibration is refitted on this reduced pool, so the held-out
            # launch's outcomes cannot influence either the shape or spread.
            pool = comparables[:index] + comparables[index + 1:]
            rows = model.forecast(pseudo_case(curve, pool))
            scores.add(rows, curve.viewers)
            baseline.add(house_baseline(curve.log_first), curve.viewers)
            targets += 1
        print(scores.report(path.stem, targets))
        pooled.merge(scores)
        total_targets += targets
    print(pooled.report("Forecaster pooled", total_targets))
    print(baseline.report("House baseline pooled", total_targets))
    print("Real practice forecasts (peak over hours 2-24; sigma is smoothed, before k):")
    for name, case in cases:
        print(diagnostics(name, case))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PRACTICE_INPUT)
    args = parser.parse_args()
    evaluate(args.input_dir)


if __name__ == "__main__":
    main()
