#!/usr/bin/env python3
"""btc-cent-forecast-01 entry: damped multi-horizon momentum forecaster.

Forecasts BTC's price 240 minutes after each checkpoint from the 4,320
one-minute closes (72 hours) in the case file.

Method: take log-returns over 1h / 6h / 24h lookbacks, scale each linearly
to the 240-minute horizon, and combine them with weights that favour the
recent past (0.80 / 0.15 / 0.05). The combined drift is damped with a
shrinkage factor of 0.4 -- at a 4-hour horizon, short-horizon noise dominates
and the raw extrapolated drift overstates the move -- and the predicted
log-move is clamped to one realized 240-minute standard deviation (from
1-minute log-returns over the last 24 hours), so a wild recent swing cannot
produce an extreme guess. The forecast is the checkpoint close times
exp(damped, clamped move), rounded to the price grid's 5 decimals and kept
inside the scorer's bounds.

Settings were checked honestly against the kit's practice cases (5 of 6 have
a knowable target: each target equals the next case's checkpoint close).
The practice is no guide to the final, so the settings stay conservative:
heavy damping and a 1-sigma clamp.

Python 3.13 standard library only, deterministic, no network, no randomness.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999

# Lookbacks in minutes, with the weight each horizon's scaled drift gets.
HORIZONS = ((60, 0.80), (360, 0.15), (1440, 0.05))
SHRINK = 0.4           # damping applied to the combined extrapolated drift
CLAMP_SIGMAS = 1.0     # max |predicted log-move| in units of realized 240-min sigma
MAX_MOVE = 0.05        # hard cap on |predicted log-move| (~5% in 4h), a second
                       # safety net for wild markets where realized sigma is huge
VOL_WINDOW = 1440      # minutes of 1-minute returns used for realized volatility


def price5(x):
    """x kept on the grid's 5 decimals and inside the scorer's (0, 10,000,000)."""
    return round(min(MAX_PRICE, max(MIN_PRICE, x)) * 1e5) / 1e5


def log_return(prices, minutes):
    """Log-return of the last price vs the price `minutes` ago (or the first)."""
    if len(prices) < 2:
        return 0.0
    back = prices[0] if minutes >= len(prices) else prices[-1 - minutes]
    if back <= 0 or prices[-1] <= 0:
        return 0.0
    return math.log(prices[-1] / back)


def realized_sigma_240(prices):
    """Std-dev of 1-minute log-returns over the last VOL_WINDOW minutes,
    scaled to a 240-minute horizon. Falls back to a small floor."""
    n = min(VOL_WINDOW, len(prices) - 1)
    if n < 10:
        return 0.001
    rets = []
    for i in range(len(prices) - n, len(prices)):
        a, b = prices[i - 1], prices[i]
        if a > 0 and b > 0:
            rets.append(math.log(b / a))
    if len(rets) < 10:
        return 0.001
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    return max(math.sqrt(var * 240), 1e-6)


def forecast(prices):
    """One forecast: the checkpoint close moved by the damped, clamped drift."""
    # Robustness: a non-positive or non-finite close anywhere breaks log
    # math, so fall back to persistence on the last valid close.
    valid = [p for p in prices if isinstance(p, (int, float)) and math.isfinite(p) and p > 0]
    if len(valid) != len(prices):
        if not valid:
            raise ValueError("no valid prices")
        return price5(valid[-1])
    last = prices[-1]
    drift = 0.0
    for minutes, weight in HORIZONS:
        # Scale this horizon's log-return linearly to the 240-minute horizon.
        drift += weight * log_return(prices, minutes) * (240.0 / minutes)
    move = SHRINK * drift
    cap = CLAMP_SIGMAS * realized_sigma_240(prices)
    move = max(-cap, min(cap, move))
    move = max(-MAX_MOVE, min(MAX_MOVE, move))
    return price5(last * math.exp(move))


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(p.name for p in in_dir.glob("*.in")):
        case_id = name[: -len(".in")]
        with open(in_dir / name, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        markets = case["markets"]
        if len(markets) != 1:
            raise ValueError(f"{name} does not hold exactly one market")
        prices = markets[0]["prices"]
        if not prices:
            raise ValueError(f"{name} holds no prices")
        guess = forecast([float(p) for p in prices])
        out = {"forecasts": [{"case": case_id, "price": guess}]}
        with open(out_dir / f"{case_id}.out", "w", encoding="utf-8") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
