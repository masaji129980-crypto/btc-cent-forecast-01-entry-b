"""BTC 4-hour-ahead forecaster for make.fun btc-cent-forecast-01.

Reads every ``*.in`` case found in the input directory (``$MAKEFUN_INPUT_DIR``
or ``/input``) and writes one JSON guess per case into the output directory
(``$MAKEFUN_OUTPUT_DIR`` or ``/output``).

Method: damped multi-horizon momentum on log-returns.  Log-returns over the
last hour, the last six hours and the last day are each stretched linearly to
the 240-minute horizon, blended with weights that favour the recent past, and
then shrunk by a damping factor.  The resulting log-move is clamped to one
realised 240-minute volatility (the standard deviation of 1-minute log-returns
over the last day, scaled up to 240 minutes) and additionally capped at 5%,
so a wild print can never throw the guess far from the checkpoint close.
If the input is unusable, the checkpoint close itself is the guess.

Standard library only, no network, no randomness, fully deterministic.
"""

import json
import math
import os
from pathlib import Path

# --- tuning knobs -----------------------------------------------------------
# How far back each momentum window reaches, in minutes.
NEAR_MINUTES = 60
MID_MINUTES = 360
FAR_MINUTES = 1440

# Blend weights for the three windows; the recent past dominates.
NEAR_WEIGHT = 0.80
MID_WEIGHT = 0.15
FAR_WEIGHT = 0.05

# How much of the blended drift we actually trust.
DAMPING = 0.40

# Second safety net on the predicted log-move, besides the volatility clamp.
MOVE_LIMIT = 0.05

# How many minutes of 1-minute returns feed the volatility estimate.
VOL_MINUTES = 1440

# Target horizon of the competition, in minutes.
TARGET_MINUTES = 240

# Bounds the scorer accepts for a price guess.
MIN_PRICE = 0.0
MAX_PRICE = 10_000_000.0


def _valid_price(value):
    """Return the value as a finite positive float, or None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _clean_series(raw):
    """Keep only finite positive closes, in order; drop anything else."""
    series = []
    for value in raw or []:
        price = _valid_price(value)
        if price is not None:
            series.append(price)
    return series


def _log_change(series, span):
    """Log-return over the last ``span`` minutes of ``series``.

    Returns None when the series does not hold the full window, so a
    short series never lets a window silently stand in for a longer one
    (a 99-minute stretch must not be scaled as if it were a whole day).
    """
    if len(series) < span + 1:
        return None
    earlier = series[-span - 1]
    later = series[-1]
    return math.log(later / earlier)


def _minute_returns(series, count):
    """1-minute log-returns over the last ``count`` minutes of ``series``."""
    tail = series[-(count + 1):]
    out = []
    for prev, cur in zip(tail, tail[1:]):
        out.append(math.log(cur / prev))
    return out


def _spread(values):
    """Population standard deviation of a list of numbers (0 if degenerate)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)


def _volatility_240(series):
    """Realised volatility scaled to the 240-minute horizon.

    Standard deviation of 1-minute log-returns over the last day, scaled by
    sqrt(240) under the usual random-walk scaling.
    """
    returns = _minute_returns(series, VOL_MINUTES)
    if not returns:
        return 0.0
    return _spread(returns) * math.sqrt(TARGET_MINUTES)


def _projected_log_move(series):
    """Damped, clamped log-move expected over the next 240 minutes."""
    windows = (
        (NEAR_MINUTES, NEAR_WEIGHT),
        (MID_MINUTES, MID_WEIGHT),
        (FAR_MINUTES, FAR_WEIGHT),
    )
    blended = 0.0
    weight_used = 0.0
    for span, weight in windows:
        change = _log_change(series, span)
        if change is None:
            continue
        # Stretch the window's log-return linearly to the target horizon.
        blended += weight * change * (TARGET_MINUTES / span)
        weight_used += weight
    if weight_used == 0.0:
        return 0.0
    blended /= weight_used

    move = blended * DAMPING

    # Clamp to one realised 240-minute volatility, then to the hard cap.
    sigma = _volatility_240(series)
    if sigma > 0.0:
        move = max(-sigma, min(sigma, move))
    move = max(-MOVE_LIMIT, min(MOVE_LIMIT, move))
    return move


def guess(series):
    """Forecast price 240 minutes after the last close in ``series``."""
    if not series:
        return 1.0
    anchor = series[-1]
    move = _projected_log_move(series)
    if move == 0.0 or not math.isfinite(move):
        return round(anchor, 5)
    estimate = anchor * math.exp(move)
    estimate = round(estimate, 5)
    if not math.isfinite(estimate) or estimate <= MIN_PRICE:
        return round(anchor, 5)
    if estimate >= MAX_PRICE:
        return round(min(anchor, MAX_PRICE - 0.00001), 5)
    return estimate


def _case_files(input_dir):
    return sorted(Path(input_dir).glob("*.in"))


def _read_case(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_prices(case):
    """Pull the single market's close list out of a case document."""
    markets = case.get("markets") or []
    if not markets:
        return []
    return _clean_series(markets[0].get("prices") or [])


def _write_guess(output_dir, case_id, price):
    payload = {"forecasts": [{"case": case_id, "price": price}]}
    out_path = Path(output_dir) / (case_id + ".out")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for case_path in _case_files(input_dir):
        try:
            case = _read_case(case_path)
        except (OSError, ValueError):
            # Last resort: the file cannot be parsed, so no checkpoint close
            # is known. Still emit a shape-valid guess (price 1.0) so the
            # case stays valid rather than missing outright.
            _write_guess(output_dir, case_path.stem, 1.0)
            continue
        case_id = case.get("case") or case_path.stem
        series = _extract_prices(case)
        _write_guess(output_dir, case_id, guess(series))


if __name__ == "__main__":
    main()
