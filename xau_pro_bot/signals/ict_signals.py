"""ICT scoring contributions. Returns (bull_pts, bear_pts, reasons)."""

from __future__ import annotations

from xau_pro_bot import config
from xau_pro_bot.indicators.ict import (
    find_ote, find_fvg, find_order_blocks, get_killzone,
)
from xau_pro_bot.indicators.smc import detect_stop_hunt


def score_ict(h1_df, m15_df, h1_atr: float) -> tuple[float, float, list[str]]:
    bull = bear = 0.0
    reasons: list[str] = []

    ote = find_ote(h1_df, lookback=20)
    if ote["in_ote"]:
        if ote["direction"] == "bull":
            bull += 12
            reasons.append(f"ICT OTE bull ({ote['ote_low']:.2f}-{ote['ote_high']:.2f})")
        else:
            bear += 12
            reasons.append(f"ICT OTE bear ({ote['ote_low']:.2f}-{ote['ote_high']:.2f})")
    else:
        bull -= 5
        bear -= 5

    kz = get_killzone()
    if kz in config.PRIORITY_KILLZONES:
        bull += 10
        bear += 10
        reasons.append(f"ICT killzone {kz} (priority)")
    elif kz:
        bull += 6
        bear += 6
        reasons.append(f"ICT killzone {kz}")
    else:
        bull -= 12
        bear -= 12

    sweep = detect_stop_hunt(m15_df, atr=h1_atr)
    if sweep["bull_hunt"]:
        bull += 9
        reasons.append(f"Liquidity bull sweep @ {sweep['level_hunted']:.2f}")
    if sweep["bear_hunt"]:
        bear += 9
        reasons.append(f"Liquidity bear sweep @ {sweep['level_hunted']:.2f}")

    fvgs = find_fvg(h1_df, max_gaps=5)
    last_close = float(h1_df["Close"].iloc[-1])
    for fvg in fvgs[:3]:
        if fvg["type"] == "bull" and fvg["bottom"] <= last_close <= fvg["top"]:
            bull += 8
            reasons.append(f"H1 FVG bull mid {fvg['midpoint']:.2f}")
            break
        if fvg["type"] == "bear" and fvg["bottom"] <= last_close <= fvg["top"]:
            bear += 8
            reasons.append(f"H1 FVG bear mid {fvg['midpoint']:.2f}")
            break

    obs = find_order_blocks(h1_df, lookback=config.OB_LOOKBACK)
    for ob in obs[:5]:
        if not ob["tested"] and ob["low"] <= last_close <= ob["high"]:
            if ob["type"] == "bull":
                bull += 6
                reasons.append(f"H1 OB bull first-test {ob['mid']:.2f}")
            else:
                bear += 6
                reasons.append(f"H1 OB bear first-test {ob['mid']:.2f}")
            break

    return bull, bear, reasons
