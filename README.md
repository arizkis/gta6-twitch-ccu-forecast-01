# gta6-twitch-ccu-forecast-01

[![make.fun bounty](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![make.fun rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/arizkis/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard)
[![make.fun reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/arizkis/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

A comparables-based quantile forecaster for launch-day Twitch viewers. It anchors historical launch curves to the target's hour-1 print and weights them by audience size, unlock style, console overlap, and recency. Each log-ratio shape blends 90% similarity-weighted and 10% unweighted means, with size scale 1.0 and recency scale 12 years. Leave-one-out residual RMS spreads are smoothed over five hours, floored at 0.05, and calibrated to their pooled empirical 90% band. The program is deterministic, uses only Python 3.13's standard library, and runs without network access.

## Run it

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Every `*.in` file is read in filename order and produces a JSON file with the same stem and a `.out` extension. Input and output default to `/input` and `/output` when the environment variables are unset.

Run the nested leave-one-out evaluation on the kit's practice inputs:

```sh
python3 loo.py
```

`loo.py` also accepts `--input-dir`. Each pseudo-target is excluded from both shape fitting and spread calibration. The report includes per-case and pooled log pinball loss, 90% coverage, median absolute log error, the house baseline comparison, and real-case shape and spread diagnostics. Exact duplicate comparable pools are evaluated once per parameter setting; every practice case still contributes to pooled scores.

Run the 27-combination parameter sweep:

```sh
python3 loo.py --sweep
```

The sweep varies shrinkage (0.10, 0.25, 0.40), size scale (0.5, 1.0, 2.0), and recency years (3, 6, 12). Selection requires 87%–94% coverage. Among settings within 0.002 of the lowest eligible loss, it prefers coverage closest to 90%, then the fewest grid steps from the middle values. The report compares current and selected losses for the top size tercile and the rest. The top tercile contains the largest `ceil(n / 3)` eligible launches, including ties at its minimum hour-1 count.

`main.forecast(case, params=None)` returns the 23 rows without file I/O; `main.forecast_details(case, params=None)` also returns shape, smoothed sigma, and the calibration multiplier. Pass a `main.Params` object to override `shrinkage`, `size_scale`, and `recency_years` throughout the forecast and its inner LOO calibration. Defaults come from `LAMBDA`, `SIZE_SCALE`, and `RECENCY_YEARS` in `main.py`.

Entered on make.fun with referral link https://make.fun/?ref=8kbmacyg .
