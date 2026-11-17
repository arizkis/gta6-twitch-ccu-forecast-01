"""Nested leave-one-out evaluation; no pseudo-target enters its training pool."""

import argparse
import json
import math
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from statistics import median

import main as model


PRACTICE_INPUT = Path(
    "/home/hatch/workspace/makefun/projects/gta6-twitch-ccu-forecast-01/kit/"
    "gta6-twitch-ccu-forecast-01-kit/input"
)
SHRINKAGES = (0.10, 0.25, 0.40)
SIZE_SCALES = (0.5, 1.0, 2.0)
RECENCIES = (3.0, 6.0, 12.0)
LOSS_TOLERANCE = 0.002


@dataclass
class Scores:
    losses: list[float] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)
    covered: int = 0

    @property
    def loss(self) -> float:
        return math.fsum(self.losses) / len(self.losses) if self.losses else math.inf

    @property
    def coverage(self) -> float:
        return self.covered / len(self.errors) if self.errors else 0.0

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
            f"loss={self.loss:.6f}, "
            f"coverage90={self.coverage:.2%}, "
            f"median_abs_log_error={median(self.errors):.6f}"
        )


@dataclass
class PoolEvaluation:
    threshold: float | None
    targets: int = 0
    pooled: Scores = field(default_factory=Scores)
    baseline: Scores = field(default_factory=Scores)
    large: Scores = field(default_factory=Scores)
    rest: Scores = field(default_factory=Scores)


@dataclass
class Evaluation:
    cases: list[tuple[str, PoolEvaluation]] = field(default_factory=list)
    pooled: Scores = field(default_factory=Scores)
    baseline: Scores = field(default_factory=Scores)
    large: Scores = field(default_factory=Scores)
    rest: Scores = field(default_factory=Scores)

    @property
    def targets(self) -> int:
        return sum(result.targets for _, result in self.cases)


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


def diagnostics(name: str, case: dict, params: model.Params | None = None) -> str:
    result = model.forecast_details(case, params)
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


def load_cases(input_dir: Path) -> list[tuple[str, dict]]:
    paths = sorted(path for path in input_dir.glob("*.in") if path.is_file())
    if not paths:
        raise ValueError(f"No .in cases found in {input_dir}")
    return [
        (path.stem, json.loads(path.read_text(encoding="utf-8"))) for path in paths
    ]


def evaluate_pool(comparables: list[dict], params: model.Params) -> PoolEvaluation:
    eligible = []
    for index, comparable in enumerate(comparables):
        curve = model.extract_curve(comparable)
        if curve is not None and curve.ratios:
            eligible.append((index, curve))
    sizes = sorted(curve.viewers[1] for _, curve in eligible)
    # The largest ceil(n/3) launches form the top tercile; include threshold ties.
    threshold = sizes[-math.ceil(len(sizes) / 3)] if sizes else None
    result = PoolEvaluation(threshold)
    for index, curve in eligible:
        # Refit calibration on the reduced pool for every pseudo-target.
        pool = comparables[:index] + comparables[index + 1:]
        rows = model.forecast(pseudo_case(curve, pool), params)
        result.pooled.add(rows, curve.viewers)
        result.baseline.add(house_baseline(curve.log_first), curve.viewers)
        stratum = result.large if curve.viewers[1] >= threshold else result.rest
        stratum.add(rows, curve.viewers)
        result.targets += 1
    return result


def run_evaluation(
    cases: list[tuple[str, dict]], params: model.Params
) -> Evaluation:
    result = Evaluation()
    cache = {}
    for name, case in cases:
        comparables = case.get("comparables", [])
        # Reuse exact duplicate pools within this variant only. Every case
        # still contributes to pooled scores; no fits cross parameter settings.
        key = json.dumps(comparables, sort_keys=True, separators=(",", ":"))
        if key not in cache:
            cache[key] = evaluate_pool(comparables, params)
        pool = cache[key]
        result.cases.append((name, pool))
        for attribute in ("pooled", "baseline", "large", "rest"):
            getattr(result, attribute).merge(getattr(pool, attribute))
    return result


def parameter_text(params: model.Params) -> str:
    return (
        f"LAMBDA={params.shrinkage:.2f}, SIZE_SCALE={params.size_scale:.1f}, "
        f"RECENCY_YEARS={params.recency_years:g}"
    )


def evaluate(input_dir: Path, params: model.Params | None = None) -> None:
    params = model.Params() if params is None else params
    cases = load_cases(input_dir)
    result = run_evaluation(cases, params)
    print(f"Parameters: {parameter_text(params)}")
    print("Nested LOO: each pseudo-target is excluded from shape fitting and spread calibration.")
    print("Loss is mean log pinball loss over every valid hour and quantile.")
    for name, pool in result.cases:
        print(pool.pooled.report(name, pool.targets))
    print(result.pooled.report("Forecaster pooled", result.targets))
    print(result.baseline.report("House baseline pooled", result.targets))
    print("Real practice forecasts (peak over hours 2-24; sigma is smoothed, before k):")
    for name, case in cases:
        print(diagnostics(name, case, params))


def centrality(params: model.Params) -> int:
    """Count grid steps from the middle setting on each of the three axes."""
    return sum(
        abs(grid.index(value) - len(grid) // 2)
        for grid, value in (
            (SHRINKAGES, params.shrinkage),
            (SIZE_SCALES, params.size_scale),
            (RECENCIES, params.recency_years),
        )
    )


def select_parameters(
    results: list[tuple[model.Params, Evaluation]]
) -> tuple[model.Params, Evaluation]:
    eligible = [
        (params, result) for params, result in results
        if result.pooled.errors and 0.87 <= result.pooled.coverage <= 0.94
    ]
    if not eligible:
        raise ValueError("No parameter combination meets the 87%-94% coverage gate")
    best_loss = min(result.pooled.loss for _, result in eligible)
    near_best = [
        (params, result) for params, result in eligible
        if result.pooled.loss <= best_loss + LOSS_TOLERANCE
    ]

    def tie_break(item: tuple[model.Params, Evaluation]) -> tuple:
        params, result = item
        scores = result.pooled
        # Integer arithmetic makes equal distances above/below 90% tie exactly.
        coverage_distance = (
            abs(10 * scores.covered - 9 * len(scores.errors)) / len(scores.errors)
        )
        return (
            coverage_distance, centrality(params), scores.loss,
            params.shrinkage, params.size_scale, params.recency_years,
        )

    return min(near_best, key=tie_break)


def print_strata(label: str, result: Evaluation) -> None:
    thresholds = {pool.threshold for _, pool in result.cases}
    if len(thresholds) == 1:
        threshold = next(iter(thresholds))
        text = f"{threshold:g}" if threshold is not None else "unavailable"
    else:
        text = ", ".join(
            f"{name}={pool.threshold:g}"
            if pool.threshold is not None else f"{name}=unavailable"
            for name, pool in result.cases
        )
    print(
        f"{label} size strata: top_tercile_threshold={text} "
        "hour-1 viewers (>=; ties included)"
    )
    print(
        f"  top_tercile_loss={result.large.loss:.6f}, rest_loss={result.rest.loss:.6f}; "
        f"scored_hours={len(result.large.errors)}/{len(result.rest.errors)}"
    )


def sweep(input_dir: Path) -> None:
    cases = load_cases(input_dir)
    current = model.Params()
    print(f"Current defaults: {parameter_text(current)}", flush=True)
    print(
        "Sweep: 27 combinations; identical pools are reused within each combination.",
        flush=True,
    )
    results = [
        (params, run_evaluation(cases, params))
        for params in (
            model.Params(shrinkage, size, recency)
            for shrinkage, size, recency in product(SHRINKAGES, SIZE_SCALES, RECENCIES)
        )
    ]
    results.sort(key=lambda item: (
        item[1].pooled.loss,
        item[0].shrinkage, item[0].size_scale, item[0].recency_years,
    ))
    print("LAMBDA SIZE_SCALE RECENCY_YEARS POOLED_LOSS COVERAGE90")
    for params, result in results:
        print(
            f"{params.shrinkage:6.2f} {params.size_scale:10.1f} "
            f"{params.recency_years:13g} {result.pooled.loss:11.6f} "
            f"{result.pooled.coverage:9.2%}"
        )
    selected, result = select_parameters(results)
    print(
        "Selection: coverage in [87%, 94%]; within 0.002 of the best eligible loss, "
        "prefer coverage nearest 90%, then the fewest steps from grid midpoints."
    )
    print(
        f"Selected: {parameter_text(selected)}; loss={result.pooled.loss:.6f}, "
        f"coverage90={result.pooled.coverage:.2%}"
    )
    current_result = next((r for p, r in results if p == current), None)
    if current_result is None:
        current_result = run_evaluation(cases, current)
    print_strata(f"Current defaults ({parameter_text(current)})", current_result)
    print_strata(f"Selected ({parameter_text(selected)})", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PRACTICE_INPUT)
    parser.add_argument(
        "--sweep", action="store_true", help="Evaluate the 27-point parameter grid"
    )
    args = parser.parse_args()
    if args.sweep:
        sweep(args.input_dir)
    else:
        evaluate(args.input_dir)


if __name__ == "__main__":
    main()
