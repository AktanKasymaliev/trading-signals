"""SMC scoring contributions."""

from __future__ import annotations

from xau_pro_bot.indicators.smc import detect_structure, premium_discount
from xau_pro_bot.indicators.ict import find_order_blocks, find_fvg


def score_smc(h4_df) -> tuple[float, float, list[str]]:
    bull = bear = 0.0
    reasons: list[str] = []

    struct = detect_structure(h4_df, swing_len=5)
    event = struct["last_event"]
    if event == "CHOCH_bull":
        bull += 15
        reasons.append("H4 CHOCH bull")
    elif event == "CHOCH_bear":
        bear += 15
        reasons.append("H4 CHOCH bear")
    elif event == "BOS_bull":
        bull += 10
        reasons.append("H4 BOS bull")
    elif event == "BOS_bear":
        bear += 10
        reasons.append("H4 BOS bear")

    pd_zone = premium_discount(h4_df, lookback=50)
    if pd_zone["zone"] == "discount":
        bull += 8
        bear -= 10
        reasons.append(f"H4 discount ({pd_zone['pct_of_range']:.0f}%)")
    elif pd_zone["zone"] == "premium":
        bear += 8
        bull -= 10
        reasons.append(f"H4 premium ({pd_zone['pct_of_range']:.0f}%)")

    obs = find_order_blocks(h4_df, lookback=50)
    last_close = float(h4_df["Close"].iloc[-1])
    for ob in obs[:5]:
        if not ob["tested"] and ob["low"] <= last_close <= ob["high"]:
            if ob["type"] == "bull":
                bull += 7
                reasons.append(f"H4 OB bull {ob['mid']:.2f}")
            else:
                bear += 7
                reasons.append(f"H4 OB bear {ob['mid']:.2f}")
            break

    fvgs = find_fvg(h4_df, max_gaps=5)
    for fvg in fvgs[:3]:
        if fvg["bottom"] <= last_close <= fvg["top"]:
            if fvg["type"] == "bull":
                bull += 5
                reasons.append(f"H4 FVG bull {fvg['midpoint']:.2f}")
            else:
                bear += 5
                reasons.append(f"H4 FVG bear {fvg['midpoint']:.2f}")
            break

    return bull, bear, reasons
