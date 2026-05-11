"""SMC scoring contributions, optionally augmented with S/R zones."""

from __future__ import annotations

from xau_pro_bot import config
from xau_pro_bot.indicators.smc import detect_structure, premium_discount
from xau_pro_bot.indicators.ict import find_order_blocks, find_fvg


def _zone_bonus_for_direction(zones: list[dict], price: float,
                              liquidity_levels: list[float],
                              direction: str) -> tuple[float, list[str]]:
    """Award zone bonus for direction (BUY = support, SELL = resistance).

    Anti-double-count: if zone level matches a liquidity level within 0.1%,
    a marker is added (bonus still applied but liquidity bonus must be skipped
    upstream).
    """
    pts = 0.0
    reasons: list[str] = []
    for z in zones[:3]:
        if not (z["zone_bot"] <= price <= z["zone_top"]):
            continue
        overlap = any(
            abs(z["level"] - lq) / max(abs(z["level"]), 1) < 0.001
            for lq in liquidity_levels
        )
        marker = " (+liq overlap)" if overlap else ""
        if z["type"] == "MAJOR":
            pts += 12
            reasons.append(f"{direction} MAJOR zone @ {z['level']:.2f}{marker}")
        elif z["type"] == "MINOR":
            pts += 8
            reasons.append(f"{direction} MINOR zone @ {z['level']:.2f}{marker}")
        else:
            pts += 6
            reasons.append(f"{direction} round level @ {z['level']:.2f}{marker}")
        if z["strength"] > 70:
            pts += 5
        break
    return pts, reasons


def _opposing_zone_penalty(zones: list[dict], price: float,
                           pip_value: float) -> float:
    """If signal is heading INTO a strong zone within 30 pips, penalize -8."""
    for z in zones[:3]:
        if z["strength"] > 70:
            dist_pips = abs(z["level"] - price) / pip_value
            if dist_pips <= 30:
                return 8.0
    return 0.0


def score_smc(h4_df, sr_zones: dict | None = None,
              liquidity: dict | None = None) -> tuple[float, float, list[str]]:
    bull = bear = 0.0
    reasons: list[str] = []
    sr_zones = sr_zones or {
        "resistance_zones": [], "support_zones": [],
        "at_resistance": False, "at_support": False,
    }
    liquidity = liquidity or {"buy_side": [], "sell_side": []}

    struct = detect_structure(h4_df, swing_len=5)
    event = struct["last_event"]
    if event == "CHOCH_bull":
        bull += 15; reasons.append("H4 CHOCH bull")
    elif event == "CHOCH_bear":
        bear += 15; reasons.append("H4 CHOCH bear")
    elif event == "BOS_bull":
        bull += 10; reasons.append("H4 BOS bull")
    elif event == "BOS_bear":
        bear += 10; reasons.append("H4 BOS bear")

    pd_zone = premium_discount(h4_df, lookback=50)
    if pd_zone["zone"] == "discount":
        bull += 8; bear -= 10
        reasons.append(f"H4 discount ({pd_zone['pct_of_range']:.0f}%)")
    elif pd_zone["zone"] == "premium":
        bear += 8; bull -= 10
        reasons.append(f"H4 premium ({pd_zone['pct_of_range']:.0f}%)")

    last_close = float(h4_df["Close"].iloc[-1])
    obs = find_order_blocks(h4_df, lookback=50)
    for ob in obs[:5]:
        if not ob["tested"] and ob["low"] <= last_close <= ob["high"]:
            if ob["type"] == "bull":
                bull += 7; reasons.append(f"H4 OB bull {ob['mid']:.2f}")
            else:
                bear += 7; reasons.append(f"H4 OB bear {ob['mid']:.2f}")
            break

    fvgs = find_fvg(h4_df, max_gaps=5)
    for fvg in fvgs[:3]:
        if fvg["bottom"] <= last_close <= fvg["top"]:
            if fvg["type"] == "bull":
                bull += 5; reasons.append(f"H4 FVG bull {fvg['midpoint']:.2f}")
            else:
                bear += 5; reasons.append(f"H4 FVG bear {fvg['midpoint']:.2f}")
            break

    # S/R zone bonuses
    buy_pts, buy_reasons = _zone_bonus_for_direction(
        sr_zones["support_zones"], last_close,
        liquidity["sell_side"], "BUY")
    bull += buy_pts; reasons.extend(buy_reasons)

    sell_pts, sell_reasons = _zone_bonus_for_direction(
        sr_zones["resistance_zones"], last_close,
        liquidity["buy_side"], "SELL")
    bear += sell_pts; reasons.extend(sell_reasons)

    bull -= _opposing_zone_penalty(sr_zones["resistance_zones"], last_close,
                                    config.XAU_PIP_VALUE)
    bear -= _opposing_zone_penalty(sr_zones["support_zones"], last_close,
                                    config.XAU_PIP_VALUE)

    return bull, bear, reasons
