"""Offline launch-day Twitch viewer quantiles from comparable log-ratio curves."""

import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path


HOURS = range(2, 25)
Z_SCORES = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)
SIGMA_FLOOR = 0.10
MATCH_BONUS = math.log(2.0)
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


def launch_day_logs(comparable: dict, day_start: datetime) -> dict[int, float]:
    """Use exact hourly timestamp offsets, never the point's list position."""
    result = {}
    for point in comparable.get("hourly", []):
        offset = (utc_time(point["from"]) - day_start).total_seconds()
        if not (0 <= offset < 24 * 3600) or offset % 3600 != 0:
            continue
        log_viewers = positive_log(point.get("viewers"))
        if log_viewers is not None:
            result[int(offset // 3600) + 1] = log_viewers
    return result


def weighted_shape(samples: list[tuple[float, float]]) -> tuple[float, float]:
    """Return the weighted mean and population standard deviation of log ratios."""
    if not samples:
        # No valid evidence for this hour: retain the target's hour-1 level.
        return 0.0, SIGMA_FLOOR

    # Each sample is (log weight, log ratio). Rescaling preserves the product
    # weights while avoiding underflow when all comparables are distant.
    maximum = max(log_weight for log_weight, _ in samples)
    weighted = [
        (math.exp(log_weight - maximum), ratio) for log_weight, ratio in samples
    ]
    total = math.fsum(weight for weight, _ in weighted)
    shape = math.fsum(weight * ratio for weight, ratio in weighted) / total
    variance = math.fsum(
        weight * (ratio - shape) ** 2 for weight, ratio in weighted
    ) / total
    return shape, max(SIGMA_FLOOR, math.sqrt(variance))


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


def forecast(case: dict) -> dict:
    target_first = positive_log(case["known"][0]["viewers"])
    if target_first is None:
        raise ValueError("The target hour-1 viewer count must be finite and positive")
    target_start = utc_time(case["day_start"])
    notes = case.get("notes", "")
    target_rolling = bool(ROLLING_UNLOCK.search(notes))
    target_console = bool(CONSOLE.search(notes))
    samples = {hour: [] for hour in HOURS}

    for comparable in case.get("comparables", []):
        comp_start = utc_time(comparable["day_start"])
        curve = launch_day_logs(comparable, comp_start)
        if 1 not in curve:
            continue
        comp_first = curve[1]
        years = abs((comp_start - target_start).total_seconds()) / SECONDS_PER_YEAR

        # Add log factors instead of multiplying size, metadata, and recency
        # weights. All dates and launch metadata come from this case alone.
        log_weight = -abs(comp_first - target_first) - years / RECENCY_YEARS
        if bool(ROLLING_UNLOCK.search(comparable.get("unlock", ""))) == target_rolling:
            log_weight += MATCH_BONUS
        if target_console and CONSOLE.search(comparable.get("platforms", "")):
            log_weight += MATCH_BONUS

        for hour in HOURS:
            if hour in curve:
                samples[hour].append((log_weight, curve[hour] - comp_first))

    hours = []
    for hour in HOURS:
        shape, sigma = weighted_shape(samples[hour])
        hours.append({"hour": hour, "q": quantiles(target_first + shape, sigma)})
    return {"hours": hours}


def main() -> None:
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in sorted(input_dir.glob("*.in")):
        if not input_path.is_file():
            continue
        with input_path.open(encoding="utf-8") as source:
            result = forecast(json.load(source))
        output_path = output_dir / f"{input_path.stem}.out"
        output_path.write_text(
            json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
