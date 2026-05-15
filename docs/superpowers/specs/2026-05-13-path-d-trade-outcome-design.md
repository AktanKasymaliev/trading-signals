# Path D — Trade Outcome LightGBM (Design Spec)

**Date:** 2026-05-13
**Branch:** feature/hugging-face-ai-layer
**Status:** approved (awaiting plan)

## 1. Motivation

Path C (forward-return classifier, ±0.3% / 16 M15 bars) ships, but:

- holdout acc 65.2% / macro F1 0.53 — high accuracy hides poor trading edge
- backtest: WR 27.6% → 29.4%, Expectancy −0.12R → −0.09R, **PF 0.74 → 0.81** (still < 1)
- model never sees that a real trade can be stopped out *before* price moves the "right" way

**Path D goal:** learn from actual TP/SL outcomes, not abstract forward returns. Two heads:
- **Directional (Mode A):** BUY / SELL / NO_TRADE, label = direction *that actually paid* (TP hit before SL).
- **Filter (Mode B):** GOOD_TRADE / BAD_TRADE on baseline-emitted setups only — purpose: cut WEAK/NORMAL garbage.

Path C remains intact and runnable. Path D is opt-in via env / explicit ctor args.

## 2. Architectural Decisions (locked)

| Decision | Choice |
|---|---|
| Sampling source | Baseline setups only (Mode B); baseline + synthetic NO_TRADE for Mode A |
| Mode A variants | **A1** baseline-only, **A2** baseline + synthetic — both trained, compared in report |
| Outcome resolution TF | M15 future bars (finer than current H1 backtest); timeout = 192 M15 bars (≈ 48 H1) |
| Same-candle TP+SL | Conservative: SL first; **count and log same_candle_conflicts** |
| Unresolved policy | `FILTER_UNRESOLVED_POLICY=bad` for v1; dataset tracks TP/SL/UNRESOLVED/SAME_CANDLE_SL_FIRST separately |
| TP target | `LABEL_TP_TARGET=tp1` default; every row stores `tp_used` |
| Filter adapter | New class `TradeFilterModel` (separate from `HFTradingModel`) |
| Engine seam | Engine accepts optional `filter_model` + `hybrid_policy`; default behavior unchanged |
| Split | Time-based 70/15/15; threshold sweep on **validation only**; test touched once for final report |
| `is_synthetic` | Mandatory feature on Mode A |

## 3. File Map

### New
- `xau_pro_bot/models/trade_outcome.py` — sample harvester + M15 outcome labeler (TP/SL/UNRESOLVED/SAME_CANDLE_SL_FIRST, MFE/MAE/R/bars_to_outcome, tp_used)
- `xau_pro_bot/models/train_path_d.py` — train Directional A1, A2, and Filter; time-split 70/15/15; conservative LGBM params; save 3 joblibs to `models_cache/`
- `xau_pro_bot/models/trade_filter_model.py` — `TradeFilterModel` adapter, contract: `predict(features) → {good_prob, bad_prob, decision: KEEP|BLOCK, threshold_used}`
- `xau_pro_bot/signals/hybrid_policy.py` — pure function `apply_hybrid(tier, ai_directional, ai_filter, thresholds) → KEEP|BLOCK|DOWNGRADE`
- `scripts/train_path_d_model.py` — CLI: `--mode {directional_a1, directional_a2, filter}`, `--csv`, `--out`
- `scripts/eval_path_d.py` — backtest harness for 5 AI modes **+ 4 non-AI baselines**; threshold sweep; report writer
- `docs/reports/path_d_trade_outcome_results.md` — final report

### Modified (backwards-compatible)
- `xau_pro_bot/signals/engine.py` — expose `bull_score`/`bear_score` in return dict; accept optional `filter_model`, `hybrid_policy` in `__init__` (default None)
- `xau_pro_bot/config.py` — new env vars: `AI_PATH_D_FILTER_PATH`, `AI_PATH_D_DIRECTIONAL_PATH`, `AI_HYBRID_MODE`, `AI_FILTER_THRESHOLD_WEAK` (0.70), `AI_FILTER_THRESHOLD_NORMAL` (0.55), `AI_FILTER_THRESHOLD_STRONG_BLOCK` (0.80), `FILTER_UNRESOLVED_POLICY` (bad), `LABEL_TP_TARGET` (tp1)
- `xau_pro_bot/backtest.py` — plumb filter_model + hybrid_mode through `_build_analyzer`
- `.env.example` — document new vars

### Tests (new)
- `tests/test_trade_outcome_labeler.py` — TP-only / SL-only / both same-candle → SL first / unresolved at timeout / MFE / MAE / R / bars_to_outcome / tp_used recorded
- `tests/test_path_d_sample_harvest.py` — baseline-only harvest, synthetic NO_TRADE generation, `is_synthetic` flag, baseline_sample flag, same_candle counter
- `tests/test_trade_filter_model.py` — adapter contract, threshold sweep logic (`good_prob >= threshold → KEEP`), unresolved-policy mapping
- `tests/test_hybrid_policy.py` — all 5 modes (baseline / Path-C / Path-D-dir / Path-D-filter / hybrid), WEAK-block, STRONG override, directional/baseline-direction conflict
- `tests/test_path_d_eval_thresholds.py` — sweep produces per-threshold metrics dict, picks best by PF on validation, not test
- Sanity: existing 156+ tests stay green; Path C tests untouched

## 4. Outcome Resolver (M15)

```python
@dataclass
class Outcome:
    hit_tp: bool
    hit_sl: bool
    unresolved: bool
    same_candle_conflict: bool
    final_R: float
    mfe_R: float
    mae_R: float
    bars_to_outcome: int
    tp_used: float

def resolve_outcome_m15(entry, sl, tp, direction, m15_future, timeout_bars=192) -> Outcome:
    risk = abs(entry - sl)  # > 0 required upstream
    mfe = mae = 0.0
    R_tp = abs(tp - entry) / risk
    for k, bar in enumerate(m15_future.iloc[:timeout_bars]):
        # update MFE/MAE (in R)
        if direction == "BUY":
            mfe = max(mfe, (bar.High - entry) / risk)
            mae = min(mae, (bar.Low  - entry) / risk)
            hit_sl = bar.Low  <= sl
            hit_tp = bar.High >= tp
        else:
            mfe = max(mfe, (entry - bar.Low)  / risk)
            mae = min(mae, (entry - bar.High) / risk)
            hit_sl = bar.High >= sl
            hit_tp = bar.Low  <= tp
        if hit_sl and hit_tp:
            return Outcome(hit_tp=False, hit_sl=True, unresolved=False,
                           same_candle_conflict=True, final_R=-1.0,
                           mfe_R=mfe, mae_R=mae, bars_to_outcome=k+1, tp_used=tp)
        if hit_sl:
            return Outcome(False, True, False, False, -1.0, mfe, mae, k+1, tp)
        if hit_tp:
            return Outcome(True, False, False, False, R_tp, mfe, mae, k+1, tp)
    return Outcome(False, False, True, False, 0.0, mfe, mae, timeout_bars, tp)
```

Labeler attaches: `outcome_class ∈ {TP, SL, UNRESOLVED, SAME_CANDLE_SL_FIRST}`; aggregate stats (counts and %) printed during training **and** persisted in report.

### Class assignment

- **Mode A1 / A2 directional label:**
  - TP hit (long) → 1 (BUY)
  - TP hit (short) → -1 (SELL)
  - SL hit OR unresolved → 0 (NO_TRADE)
  - SAME_CANDLE_SL_FIRST → 0 (NO_TRADE)
- **Mode B filter label:**
  - TP hit → 1 (GOOD)
  - SL hit OR SAME_CANDLE_SL_FIRST → 0 (BAD)
  - UNRESOLVED → 0 (BAD) under `FILTER_UNRESOLVED_POLICY=bad` (v1 default)
  - (counts still tracked separately for honesty)

## 5. Sample Harvester

Single pass over history (step=4 on H1, matching current backtest):

```
for each H1 cutoff:
    sig = baseline_engine.analyze(slice)         # AI fully OFF
    if sig.tier in {WEAK, NORMAL, STRONG}:
        tp = sig.tp1 if LABEL_TP_TARGET=='tp1' else sig.tp2 or sig.tp1
        outcome = resolve_outcome_m15(sig.entry, sig.sl, tp, sig.direction, m15_future)
        emit(row, baseline_sample=True, is_synthetic=False, …)
    elif (Mode A2 enabled) and i % synth_stride == 0:   # default stride=8
        entry = m15.Close[-1]; atr = m15.ATR[-1]
        for d in (BUY, SELL):
            synth_sl = entry - sign(d)*1.5*atr
            synth_tp = entry + sign(d)*3.0*atr           # RR=2
            outcome = resolve_outcome_m15(...)
            emit(row, baseline_sample=False, is_synthetic=True, …)
```

Filter dataset = rows where `baseline_sample=True`.
Directional A1 dataset = `baseline_sample=True` rows.
Directional A2 dataset = all rows.

## 6. Features (~45)

29 existing (`build_ai_features` `internal`) **+** baseline-context (engine exposes during harvest):
`bull_score`, `bear_score`, `score_gap=|bull-bear|`, `final_score`, `signal_tier` (one-hot WEAK/NORMAL/STRONG/NO_SIGNAL), `direction` (one-hot BUY/SELL), `rr`, `killzone` (one-hot 5 zones + none), `hour_ny`, `day_of_week`, `is_weak`, `is_normal`, `is_strong`, `atr_percentile_h1`, `range_vs_atr_m15`, `trend_align_h4_h1_m15` (int -3..+3), `dist_to_recent_high_atr`, `dist_to_recent_low_atr`, **`is_synthetic`** (mandatory).

## 7. Training

```python
default_params = dict(
    objective="binary",          # filter; "multiclass" + num_class=3 for directional
    learning_rate=0.03,
    max_depth=5,
    num_leaves=31,
    min_data_in_leaf=120,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    class_weight="balanced",
    n_estimators=600,
    n_jobs=-1, verbose=-1,
)
# time-based 70/15/15
# early_stopping=40 evaluated on VALIDATION split (not test)
```

Save:
- `models_cache/path_d_directional_a1_lgb.joblib`
- `models_cache/path_d_directional_a2_lgb.joblib`
- `models_cache/path_d_trade_outcome_lgb.joblib`  ← filter model (matches user-specified path)

## 8. Engine Seam

```python
# engine.py — MasterSignalEngine.__init__
def __init__(self, ..., ai_model=None,
             filter_model: TradeFilterModel | None = None,
             hybrid_policy: HybridPolicy | None = None): ...
```

Engine return dict gains `bull_score`, `bear_score` (additive, backward-compatible).

If `filter_model` set and `hybrid_policy` set: after baseline tier decided, engine calls `hybrid_policy.decide(tier, ai_directional, filter_pred, thresholds)`. On BLOCK → tier="NO_SIGNAL", `ai_blocked=True`, levels stripped.

## 9. Hybrid Policy (mode E)

```python
def decide(tier, ai_dir, ai_filter, T):
    # STRONG: keep unless filter very confident bad
    if tier == "STRONG":
        if ai_filter and ai_filter.bad_prob >= T.strong_block:    # 0.80
            return BLOCK
        return KEEP
    # NORMAL: require filter approval
    if tier == "NORMAL":
        if ai_filter and ai_filter.good_prob < T.normal:          # 0.55
            return BLOCK
        # directional conflict
        if ai_dir and ai_dir.direction != baseline_dir and ai_dir.confidence > 0.65:
            return BLOCK
        return KEEP
    # WEAK: high bar
    if tier == "WEAK":
        if ai_filter and ai_filter.good_prob < T.weak:            # 0.70
            return BLOCK
        return KEEP
    return KEEP  # no_signal stays no_signal
```

## 10. Evaluator & Report (`scripts/eval_path_d.py`)

**Modes evaluated on same history (single harness):**

AI modes:
- **A** baseline only (AI off)
- **B** Path C directional (existing)
- **C** Path D directional A1
- **D** Path D directional A2
- **E** Path D filter only (default thresholds)
- **F** Hybrid (mode E policy above)

**Non-AI baselines (added per user request, prove AI > simple tier rules):**
- **G** baseline all (= mode A)
- **H** baseline without WEAK
- **I** baseline STRONG only
- **J** baseline STRONG + NORMAL only

**Per-mode metrics:** signals_generated, blocked, W/L, WR, Expectancy R, PF, Max DD, avg R, n_trades; per-tier (WEAK/NORMAL/STRONG); per-killzone.

**Threshold sweep (filter / hybrid):** thresholds ∈ {0.50, 0.55, 0.60, 0.65, 0.70, 0.75} — computed on **validation** to pick best (PF tiebreak: higher kept_trades). Test set evaluated **once** with the chosen threshold.

**Outcome distribution stats** persisted in report: % TP / % SL / % UNRESOLVED / % SAME_CANDLE_SL_FIRST per dataset.

## 11. Acceptance Criteria

**Minimum success (else report says "do not deploy"):**
- PF > Path C PF on out-of-sample test
- Expectancy > Path C Expectancy on test
- WEAK either materially improves or auto-blocks
- `kept_trades >= 25% of baseline_trades` on test — **hard floor**; below this, even great PF/WR rejected
- Honest report on where it works and where it doesn't

**Good success:** PF > 1.10, Expectancy > 0, WR ≥ 40–45% at RR > 1.5, trade count not collapsed.

**Excellent:** PF > 1.20, Expectancy > +0.05R, stable across BUY/SELL and sessions.

## 12. Constraints (binding)

- Do not break existing 156+ tests; Path C remains runnable.
- Path D is opt-in via env / explicit ctor.
- No claiming profitability if PF < 1.
- No accuracy-driven decisions.
- No random splits.
- Threshold chosen on validation, applied to test once.
- Engine changes additive (new optional ctor args, additive return-dict fields).

## 13. Out of Scope (v1)

- Walk-forward CV (mention in report as next step).
- DXY / US10Y / external data.
- Online / incremental learning.
- Live engine wiring beyond ctor args (no production deployment).
- Hyperparameter tuning beyond defaults.

## 14. Next Step

Transition to `superpowers:writing-plans` to produce a task-by-task execution plan tied to TDD per task.
