# gta6-twitch-ccu-forecast-01

[![make.fun bounty](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![make.fun rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/arizkis/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard)
[![make.fun reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/arizkis/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

A comparables-based quantile forecaster for launch-day Twitch viewers. It anchors historical launch curves to the target's hour-1 print and weights them by audience size, unlock style, console overlap, and recency. Weighted log-ratio means and standard deviations produce five quantiles for each of hours 2–24, with a log-spread floor of 0.10. The program is deterministic, uses only Python 3.13's standard library, and runs without network access.

## Run it

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Every `*.in` file is read in filename order and produces a JSON file with the same stem and a `.out` extension. Input and output default to `/input` and `/output` when the environment variables are unset.

Entered on make.fun with referral link https://make.fun/?ref=8kbmacyg .
