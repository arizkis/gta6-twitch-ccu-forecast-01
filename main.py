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
LAMBDA = 0.25
SIGMA_FLOOR = 0.05
SMOOTHING_WINDOW = 5
K_MIN = 0.5
K_MAX = 3.0
LOG_MATCH_BONUS = math.log(2.0)
RECENCY_YEARS = 6.0
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


def similarity_log_weight(target: Target, curve: ComparableCurve) -> float:
    """Log of the product of size, unlock, console, and recency factors."""
    years = abs((curve.day_start - target.day_start).total_seconds()) / SECONDS_PER_YEAR
    weight = -abs(curve.log_first - target.log_first) - years / RECENCY_YEARS
    if bool(ROLLING_UNLOCK.search(curve.unlock)) == target.rolling:
        weight += LOG_MATCH_BONUS
    if target.console and CONSOLE.search(curve.platforms):
        weight += LOG_MATCH_BONUS
    return weight


def blended_shape(samples: list[tuple[float, float]]) -> float:
    """Partially pool weighted and unweighted means over identical samples."""
    if not samples:
        return 0.0  # No evidence for this hour: retain the hour-1 anchor.

    # Rescaling preserves relative product weights and avoids total underflow.
    maximum = max(log_weight for log_weight, _ in samples)
    weighted = [
        (math.exp(log_weight - maximum), ratio) for log_weight, ratio in samples
    ]
    total = math.fsum(weight for weight, _ in weighted)
    weighted_mean = math.fsum(weight * ratio for weight, ratio in weighted) / total
    unweighted_mean = math.fsum(ratio for _, ratio in samples) / len(samples)
    return (1 - LAMBDA) * weighted_mean + LAMBDA * unweighted_mean


def compute_shape(target: Target, curves: list[ComparableCurve]) -> dict[int, float]:
    samples = {hour: [] for hour in HOURS}
    for curve in curves:
        weight = similarity_log_weight(target, curve)
        for hour, ratio in curve.ratios.items():
            samples[hour].append((weight, ratio))
    return {hour: blended_shape(samples[hour]) for hour in HOURS}


def loo_residuals(curves: list[ComparableCurve]) -> dict[int, list[float]]:
    """Predict each launch using only the other launches, with the same shrinkage."""
    residuals = {hour: [] for hour in HOURS}
    for index, curve in enumerate(curves):
        target = target_attributes(
            curve.day_start, curve.log_first, comparable_notes(curve)
        )
        shape = compute_shape(target, curves[:index] + curves[index + 1:])
        for hour, actual in curve.ratios.items():
            residuals[hour].append(actual - shape[hour])
    return residuals


def calibrate_spread(curves: list[ComparableCurve]) -> tuple[dict[int, float], float]:
    """Fit per-hour RMS spread and one pooled 90% coverage multiplier once."""
    residuals = loo_residuals(curves)
    raw_sigma = {
        hour: math.sqrt(math.fsum(value * value for value in values) / len(values))
        if values else SIGMA_FLOOR
        for hour, values in residuals.items()
    }
    radius = SMOOTHING_WINDOW // 2
    sigma = {}
    for hour in HOURS:
        window = range(
            max(HOURS.start, hour - radius), min(HOURS.stop, hour + radius + 1)
        )
        sigma[hour] = max(
            SIGMA_FLOOR, math.fsum(raw_sigma[neighbor] for neighbor in window) / len(window)
        )

    standardized = sorted(
        abs(value) / sigma[hour] for hour, values in residuals.items() for value in values
    )
    if not standardized:
        return sigma, 1.0
    # Nearest-rank empirical percentile: at least 90% are at or below p90.
    p90 = standardized[math.ceil(0.90 * len(standardized)) - 1]
    k = min(K_MAX, max(K_MIN, p90 / Z_SCORES[-1]))
    return sigma, k


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


def forecast_details(case: dict) -> ForecastResult:
    """Fit once, returning rows plus reusable shape and calibration diagnostics."""
    target = target_from_case(case)
    curves = extract_comparables(case.get("comparables", []))
    shape = compute_shape(target, curves)
    sigma, k = calibrate_spread(curves)
    hours = [
        {"hour": hour, "q": quantiles(target.log_first + shape[hour], sigma[hour] * k)}
        for hour in HOURS
    ]
    return ForecastResult(hours, shape, sigma, k)


def forecast(case: dict) -> list[dict]:
    """Return the 23 forecast rows without performing any file I/O."""
    return forecast_details(case).hours


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
