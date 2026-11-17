"""Offline launch-day Twitch viewer quantiles from comparable log-ratio curves."""

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


HOURS = range(2, 25)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_SCORES = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)
LAMBDA = 0.20  # Unweighted share for hours 2-11; hours 12-24 use zero.
SIZE_SCALE = 1.0
SIGMA_FLOOR = 0.05
SMOOTHING_WINDOW = 5
K_MIN = 0.5
K_MAX = 3.0
LOG_MATCH_BONUS = math.log(2.0)
RECENCY_YEARS = 12.0
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
MIN_VIEWERS = 0.01  # Stays positive after rounding to two decimals.
MAX_VIEWERS = 1_000_000_000.0
LOG_MIN = math.log(MIN_VIEWERS)
LOG_MAX = math.log(MAX_VIEWERS)

ROLLING_UNLOCK = re.compile(
    r"\brolling\b|\bmidnight[\s-]+local\b|\blocal[\s-]+midnight\b"
    r"|\b12\s*a\.?m\.?\s+local\b",
    re.IGNORECASE,
)
CONSOLE = re.compile(
    r"\b(?:consoles?|playstation\s*[1-5]?|ps[1-5]?|xbox|xb1|nintendo|switch|wii)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Params:
    shrinkage: float = LAMBDA
    size_scale: float = SIZE_SCALE
    recency_years: float = RECENCY_YEARS

    def __post_init__(self) -> None:
        values = (self.shrinkage, self.size_scale, self.recency_years)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Forecast parameters must be finite")
        if not 0 <= self.shrinkage <= 1:
            raise ValueError("Shrinkage must lie in [0, 1]")
        if self.size_scale <= 0 or self.recency_years <= 0:
            raise ValueError("Size scale and recency years must be positive")


@dataclass(frozen=True)
class Target:
    day_start: datetime
    log_first: float
    rolling: bool
    console: bool


@dataclass(frozen=True)
class ComparableCurve:
    day_start: datetime
    viewers: dict[int, float]
    log_first: float
    ratios: dict[int, float]
    unlock: str
    platforms: str


@dataclass(frozen=True)
class ForecastResult:
    hours: list[dict]
    shape: dict[int, float]
    sigma: dict[int, float]  # Smoothed and floored, before applying k.
    k: float


@dataclass(frozen=True)
class SpreadCalibration:
    sigma: dict[int, float]
    log_size_threshold: float | None
    large_k: float
    rest_k: float


def utc_time(value: str) -> datetime:
    """Parse an ISO timestamp without depending on the machine's time zone."""
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def positive_log(value: object) -> float | None:
    """Missing, nonfinite, and nonpositive observations are not evidence."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return math.log(value)


def extract_curve(comparable: dict) -> ComparableCurve | None:
    """Use exact hourly timestamp offsets, never the point's list position."""
    day_start = utc_time(comparable["day_start"])
    viewers = {}
    for point in comparable.get("hourly", []):
        offset = (utc_time(point["from"]) - day_start).total_seconds()
        if not (0 <= offset < 24 * 3600) or offset % 3600 != 0:
            continue
        if positive_log(point.get("viewers")) is not None:
            viewers[int(offset // 3600) + 1] = point["viewers"]
    if 1 not in viewers:
        return None
    log_first = math.log(viewers[1])
    return ComparableCurve(
        day_start=day_start,
        viewers=viewers,
        log_first=log_first,
        ratios={
            hour: math.log(value) - log_first
            for hour, value in viewers.items() if hour in HOURS
        },
        unlock=comparable.get("unlock", ""),
        platforms=comparable.get("platforms", ""),
    )


def extract_comparables(comparables: list[dict]) -> list[ComparableCurve]:
    """Parse each curve once and discard launches with no positive hour 1."""
    curves = []
    for comparable in comparables:
        curve = extract_curve(comparable)
        if curve is not None:
            curves.append(curve)
    return curves


def target_attributes(day_start: datetime, log_first: float, notes: str) -> Target:
    """The single notes-text attribute path for real and pseudo targets."""
    return Target(
        day_start, log_first,
        bool(ROLLING_UNLOCK.search(notes)), bool(CONSOLE.search(notes)),
    )


def comparable_notes(curve: ComparableCurve) -> str:
    return f"{curve.unlock}\nPlatforms: {curve.platforms}"


def target_from_case(case: dict) -> Target:
    log_first = positive_log(case["known"][0]["viewers"])
    if log_first is None:
        raise ValueError("The target hour-1 viewer count must be finite and positive")
    return target_attributes(
        utc_time(case["day_start"]), log_first, case.get("notes", "")
    )


def similarity_log_weight(
    target: Target, curve: ComparableCurve, params: Params | None = None
) -> float:
    """Log of the product of size, unlock, console, and recency factors."""
    params = Params() if params is None else params
    years = abs((curve.day_start - target.day_start).total_seconds()) / SECONDS_PER_YEAR
    weight = (
        -abs(curve.log_first - target.log_first) / params.size_scale
        - years / params.recency_years
    )
    if bool(ROLLING_UNLOCK.search(curve.unlock)) == target.rolling:
        weight += LOG_MATCH_BONUS
    if target.console and CONSOLE.search(curve.platforms):
        weight += LOG_MATCH_BONUS
    return weight


def weighted_median(samples: list[tuple[float, float]]) -> float:
    """Take the first ratio reaching half the total weight in ascending order."""
    ordered = sorted(samples, key=lambda sample: sample[1])
    midpoint = math.fsum(weight for weight, _ in ordered) / 2
    cumulative = 0.0
    for weight, ratio in ordered:
        cumulative += weight
        if cumulative >= midpoint:
            return ratio
    return ordered[-1][1]


def blended_shape(
    samples: list[tuple[float, float]], hour: int, params: Params | None = None
) -> float:
    """Blend weighted and unweighted medians with hour-dependent shrinkage."""
    params = Params() if params is None else params
    if not samples:
        return 0.0  # No evidence for this hour: retain the hour-1 anchor.

    # Rescaling preserves relative product weights and avoids total underflow.
    maximum = max(log_weight for log_weight, _ in samples)
    weighted = [
        (math.exp(log_weight - maximum), ratio) for log_weight, ratio in samples
    ]
    weighted_middle = weighted_median(weighted)
    unweighted_middle = weighted_median([(1.0, ratio) for _, ratio in samples])
    shrinkage = params.shrinkage if hour < 12 else 0.0
    return (1 - shrinkage) * weighted_middle + shrinkage * unweighted_middle


def compute_shape(
    target: Target, curves: list[ComparableCurve], params: Params | None = None
) -> dict[int, float]:
    params = Params() if params is None else params
    samples = {hour: [] for hour in HOURS}
    for curve in curves:
        weight = similarity_log_weight(target, curve, params)
        for hour, ratio in curve.ratios.items():
            samples[hour].append((weight, ratio))
    return {hour: blended_shape(samples[hour], hour, params) for hour in HOURS}


def loo_residuals(
    curves: list[ComparableCurve], params: Params | None = None
) -> dict[int, list[tuple[float, float]]]:
    """Pair each LOO residual with its launch's log hour-1 size."""
    params = Params() if params is None else params
    residuals = {hour: [] for hour in HOURS}
    for index, curve in enumerate(curves):
        target = target_attributes(
            curve.day_start, curve.log_first, comparable_notes(curve)
        )
        shape = compute_shape(target, curves[:index] + curves[index + 1:], params)
        for hour, actual in curve.ratios.items():
            residuals[hour].append((actual - shape[hour], curve.log_first))
    return residuals


def calibration_multiplier(standardized: list[float]) -> float:
    """Nearest-rank 90th percentile, converted to a clamped normal multiplier."""
    if not standardized:
        return 1.0
    ordered = sorted(standardized)
    p90 = ordered[math.ceil(0.90 * len(ordered)) - 1]
    return min(K_MAX, max(K_MIN, p90 / Z_SCORES[-1]))


def calibrate_spread(
    curves: list[ComparableCurve], params: Params | None = None
) -> SpreadCalibration:
    """Fit per-hour MAD spread and size-stratified 90% coverage multipliers."""
    residuals = loo_residuals(curves, params)
    raw_sigma = {}
    for hour, samples in residuals.items():
        values = sorted(value for value, _ in samples)
        if values:
            middle = values[len(values) // 2]
            deviations = sorted(abs(value - middle) for value in values)
            raw_sigma[hour] = 1.4826 * deviations[len(deviations) // 2]
        else:
            raw_sigma[hour] = SIGMA_FLOOR
    radius = SMOOTHING_WINDOW // 2
    sigma = {}
    for hour in HOURS:
        window = range(
            max(HOURS.start, hour - radius), min(HOURS.stop, hour + radius + 1)
        )
        sigma[hour] = max(
            SIGMA_FLOOR, math.fsum(raw_sigma[neighbor] for neighbor in window) / len(window)
        )

    sizes = sorted(curve.log_first for curve in curves)
    log_threshold = sizes[-math.ceil(len(sizes) / 3)] if sizes else None
    standardized = []
    large = []
    rest = []
    for hour, samples in residuals.items():
        for value, log_first in samples:
            scaled = abs(value) / sigma[hour]
            standardized.append(scaled)
            stratum = large if log_first >= log_threshold else rest
            stratum.append(scaled)
    pooled_k = calibration_multiplier(standardized)
    return SpreadCalibration(
        sigma, log_threshold,
        calibration_multiplier(large) if large else pooled_k,
        calibration_multiplier(rest) if rest else pooled_k,
    )


def quantiles(log_median: float, sigma: float) -> list[float]:
    """Exponentiate only at output, then clamp, round, and enforce ordering."""
    result = []
    previous = MIN_VIEWERS
    for z in Z_SCORES:
        log_value = min(LOG_MAX, max(LOG_MIN, log_median + z * sigma))
        value = round(math.exp(log_value), 2)
        previous = max(previous, min(MAX_VIEWERS, value))
        result.append(previous)
    return result


def forecast_details(case: dict, params: Params | None = None) -> ForecastResult:
    """Fit once, returning rows plus reusable shape and calibration diagnostics."""
    params = Params() if params is None else params
    target = target_from_case(case)
    curves = extract_comparables(case.get("comparables", []))
    shape = compute_shape(target, curves, params)
    calibration = calibrate_spread(curves, params)
    sigma = calibration.sigma
    k = (
        calibration.large_k
        if calibration.log_size_threshold is not None
        and target.log_first >= calibration.log_size_threshold
        else calibration.rest_k
    )
    hours = [
        {"hour": hour, "q": quantiles(target.log_first + shape[hour], sigma[hour] * k)}
        for hour in HOURS
    ]
    return ForecastResult(hours, shape, sigma, k)


def forecast(case: dict, params: Params | None = None) -> list[dict]:
    """Return the 23 forecast rows without performing any file I/O."""
    return forecast_details(case, params).hours


def main() -> None:
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in sorted(input_dir.glob("*.in")):
        if not input_path.is_file():
            continue
        with input_path.open(encoding="utf-8") as source:
            result = {"hours": forecast(json.load(source))}
        output_path = output_dir / f"{input_path.stem}.out"
        output_path.write_text(
            json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
