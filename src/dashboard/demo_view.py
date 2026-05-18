"""
Demo View — Presentation-grade showcase page.

Optimised for live demo / 발표 시연. Layout follows the impact-first rule:
viewer should grasp the headline number in under 3 seconds, then drill into
the comparison narrative ("AI 타겟팅 vs 무차별 살포 vs 무대응") and finally
into segment-level mechanics.

Sections (top → bottom):

1.  **Hero**  — single hero KPI ("₩XXX 투자 → ₩YYY 순이익, Z.ZZx ROI").
    Updates live with the budget slider.
2.  **Budget slider** with one-line caption stating won-for-won return.
3.  **3 scenario cards** — 무대응 / 무차별 살포 / AI 타겟팅 — apples-to-apples.
4.  **Single-axis area chart** — budget vs net gain, with the LP sweet spot
    marked and the slider position highlighted.
5.  **Top-3 segment insights** — where the LP concentrated spend, plus a
    one-line "AI auto-excluded" note. Full 8-row table tucked in an expander.
6.  **Conclusion** — one bold line tying the demo together.

Numbers come from the same LP solution shown on the Budget Optimization
page (`budget_optimization.csv` + `budget_whatif.csv` + `budget_results.csv`).
Uniform-treatment scenario reads from `uniform_treatment_clv.json` (CLV side)
and `budget_optimization.csv` `cost_per_action` column (cost side).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.dashboard.utils.dashboard_helpers import (
    format_currency_krw,
    get_lang,
    tr,
)

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_lp_artifacts() -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Read the LP per-customer solution + what-if scenarios."""
    bo_path = RESULTS_DIR / "budget_optimization.csv"
    wh_path = RESULTS_DIR / "budget_whatif.csv"
    bo = None
    wh = None
    if bo_path.exists():
        try:
            bo = pd.read_csv(bo_path)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("demo_view: failed to read %s: %s", bo_path, e)
    if wh_path.exists():
        try:
            wh = pd.read_csv(wh_path)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("demo_view: failed to read %s: %s", wh_path, e)
    return bo, wh


def _load_segment_results() -> Optional[pd.DataFrame]:
    """Read per-segment LP roll-up (budget_results.csv)."""
    seg_path = RESULTS_DIR / "budget_results.csv"
    if not seg_path.exists():
        return None
    try:
        return pd.read_csv(seg_path)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("demo_view: failed to read budget_results.csv: %s", e)
        return None


def _prepare_state(
    bo: pd.DataFrame, wh: Optional[pd.DataFrame]
) -> Optional[Dict[str, Any]]:
    """Build greedy arrays + whatif anchor metadata.

    Returns ``None`` when the LP artefact is missing required columns or
    has zero allocated budget.
    """
    required = (
        "clv",
        "churn_prob",
        "allocated_budget",
        "expected_revenue_saved_krw",
    )
    missing = [c for c in required if c not in bo.columns]
    if missing:
        logger.warning("demo_view: LP file missing cols %s", missing)
        return None

    clv = pd.to_numeric(bo["clv"], errors="coerce").fillna(0.0).to_numpy(float)
    churn = pd.to_numeric(bo["churn_prob"], errors="coerce").fillna(0.0).to_numpy(float)
    churn = np.clip(churn, 0.0, 1.0)
    baseline_clv = float((clv * (1.0 - churn)).sum())

    cost_full = pd.to_numeric(bo["allocated_budget"], errors="coerce").fillna(0.0).to_numpy(float)
    saved_full = pd.to_numeric(bo["expected_revenue_saved_krw"], errors="coerce").fillna(0.0).to_numpy(float)

    eligible_mask = (cost_full > 0.0) & (saved_full > 0.0)
    if not eligible_mask.any():
        return None

    # Sort the eligible LP-selected customers by ROI per won (desc), then by
    # absolute saved (desc) as tie-break. Build a sorted DataFrame so the
    # segment table can groupby on a budget-driven slice at render time.
    bo_eligible = bo[eligible_mask].copy()
    bo_eligible["_roi_per_won"] = (
        saved_full[eligible_mask] / np.maximum(cost_full[eligible_mask], 1.0)
    )
    bo_sorted = bo_eligible.sort_values(
        by=["_roi_per_won", "expected_revenue_saved_krw"],
        ascending=[False, False],
    ).reset_index(drop=True)
    bo_sorted["_cum_cost"] = bo_sorted["allocated_budget"].cumsum()
    bo_sorted["_cum_saved"] = bo_sorted["expected_revenue_saved_krw"].cumsum()

    cum_cost = bo_sorted["_cum_cost"].to_numpy(float)
    cum_saved = bo_sorted["_cum_saved"].to_numpy(float)
    lp_total_cost = float(cum_cost[-1])
    lp_total_saved = float(cum_saved[-1])
    lp_n = int(len(bo_sorted))

    # What-if anchors (LP reruns at 50 % / 100 % / 200 % of nominal).
    anchors: Dict[str, Dict[str, float]] = {}
    if wh is not None and not wh.empty and {
        "scenario_name", "total_allocated", "retained_value", "customers_treated",
    }.issubset(wh.columns):
        for _, row in wh.iterrows():
            anchors[str(row["scenario_name"])] = {
                "budget": float(row["total_allocated"]),
                "saved": float(row["retained_value"]),
                "customers": int(row["customers_treated"]),
            }

    if "budget_200pct" in anchors:
        budget_max = float(anchors["budget_200pct"]["budget"])
    else:
        budget_max = float(lp_total_cost * 1.5)

    # Uniform-treatment totals — what we'd spend AND save if we sprayed
    # coupons at every customer with their assigned cost_per_action.
    uniform_cost_total = 0.0
    if "cost_per_action" in bo.columns:
        uniform_cost_total = float(
            pd.to_numeric(bo["cost_per_action"], errors="coerce").fillna(0.0).sum()
        )

    n_total = int(len(bo))

    return {
        "baseline_clv": baseline_clv,
        "cum_cost": cum_cost,
        "cum_saved": cum_saved,
        "lp_sorted": bo_sorted,  # DataFrame, used by segment groupby at render time
        "lp_total_cost": lp_total_cost,
        "lp_total_saved": lp_total_saved,
        "lp_n": lp_n,
        "n_total": n_total,
        "uniform_cost_total": uniform_cost_total,
        # uniform_saved_total is injected by render_demo from the
        # uniform_treatment_clv.json artefact, since it can't be derived
        # from budget_optimization.csv alone.
        "uniform_saved_total": 0.0,
        "anchors": anchors,
        "budget_max": budget_max,
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _simulate(state: Dict[str, Any], budget: float) -> Dict[str, float]:
    """Greedy on LP per-customer solution up to ``lp_total_cost``; linear
    interpolation on the what-if anchors above that.
    """
    baseline = float(state["baseline_clv"])
    cum_cost = state["cum_cost"]
    cum_saved = state["cum_saved"]
    lp_cost = float(state["lp_total_cost"])
    lp_saved = float(state["lp_total_saved"])
    lp_n = int(state["lp_n"])
    anchors = state["anchors"]

    if budget <= 0.0:
        return {
            "baseline_clv": baseline, "post_clv": baseline, "delta_clv": 0.0,
            "spend": 0.0, "net_gain": 0.0, "n_treated": 0, "roi_multiple": 0.0,
        }

    if budget <= lp_cost or not anchors:
        k = int(np.searchsorted(cum_cost, budget, side="right"))
        if k <= 0:
            spend, saved, n = 0.0, 0.0, 0
        elif k >= cum_cost.size:
            spend, saved, n = lp_cost, lp_saved, lp_n
        else:
            spend = float(cum_cost[k - 1])
            saved = float(cum_saved[k - 1])
            n = int(k)
    else:
        a100 = anchors.get("budget_100pct")
        a200 = anchors.get("budget_200pct")
        if a100 and a200 and a200["budget"] > a100["budget"]:
            x0, x1 = a100["budget"], a200["budget"]
            x = float(min(budget, x1))
            frac = (x - x0) / (x1 - x0)
            saved = a100["saved"] + frac * (a200["saved"] - a100["saved"])
            n = int(round(a100["customers"] + frac * (a200["customers"] - a100["customers"])))
            spend = x
        else:
            spend, saved, n = lp_cost, lp_saved, lp_n

    post = baseline + saved
    net = saved - spend
    roi = (saved / spend) if spend > 0 else 0.0
    return {
        "baseline_clv": baseline, "post_clv": post, "delta_clv": saved,
        "spend": spend, "net_gain": net, "n_treated": n, "roi_multiple": roi,
    }


def _build_curve(state: Dict[str, Any], n_points: int = 200) -> pd.DataFrame:
    """Pre-compute the budget → net-gain curve."""
    budget_max = float(state["budget_max"])
    if budget_max <= 0:
        return pd.DataFrame(columns=["budget", "post_clv", "net_gain", "spend"])
    grid = np.linspace(0.0, budget_max, n_points)
    rows = [_simulate(state, float(b)) for b in grid]
    return pd.DataFrame(
        {
            "budget": grid,
            "post_clv": [r["post_clv"] for r in rows],
            "net_gain": [r["net_gain"] for r in rows],
            "spend": [r["spend"] for r in rows],
        }
    )


def _simulate_uniform(state: Dict[str, Any], budget: float) -> Dict[str, float]:
    """Uniform-spray scenario scaled to ``budget``.

    Uniform spray applies an assigned coupon to *every* customer at a total
    cost of ``uniform_cost_total`` and yields ``uniform_saved_total`` of
    saved revenue. Scaling down to a smaller budget means we hit
    ``budget / uniform_cost_total`` of the population with proportional
    spend and proportional saved revenue (random selection, so per-won ROI
    stays constant). Above ``uniform_cost_total`` the scenario plateaus.
    """
    uc = float(state.get("uniform_cost_total", 0.0))
    us = float(state.get("uniform_saved_total", 0.0))
    n = int(state.get("n_total", 0))

    if budget <= 0.0 or uc <= 0.0:
        return {
            "spend": 0.0, "saved": 0.0, "net_gain": 0.0,
            "n_treated": 0, "roi_multiple": 0.0,
        }
    spend = float(min(budget, uc))
    frac = spend / uc
    saved = us * frac
    n_treated = int(round(n * frac))
    roi = (saved / spend) if spend > 0 else 0.0
    return {
        "spend": spend,
        "saved": saved,
        "net_gain": saved - spend,
        "n_treated": n_treated,
        "roi_multiple": roi,
    }


def _segment_at_budget(state: Dict[str, Any], budget: float) -> pd.DataFrame:
    """Greedy-slice the LP solution to ``budget`` and groupby segment.

    Returns a DataFrame with columns: segment, allocated_budget_krw,
    customers, expected_revenue_saved_krw, expected_retained, roi.
    Sorted by ``allocated_budget_krw`` descending so the top rows are the
    LP's largest bets at the current budget.
    """
    sorted_df = state.get("lp_sorted")
    if sorted_df is None or sorted_df.empty or "segment" not in sorted_df.columns:
        return pd.DataFrame()

    lp_cost = float(state.get("lp_total_cost", 0.0))
    if budget <= 0.0:
        return pd.DataFrame()
    if budget >= lp_cost:
        selected = sorted_df
    else:
        selected = sorted_df[sorted_df["_cum_cost"] <= budget]
    if selected.empty:
        return pd.DataFrame()

    agg_kwargs = {
        "allocated_budget_krw": ("allocated_budget", "sum"),
        "customers": ("allocated_budget", "size"),
        "expected_revenue_saved_krw": ("expected_revenue_saved_krw", "sum"),
    }
    if "expected_retained" in selected.columns:
        agg_kwargs["expected_retained"] = ("expected_retained", "sum")
    grouped = selected.groupby("segment", as_index=False).agg(**agg_kwargs)
    grouped["roi"] = (
        grouped["expected_revenue_saved_krw"]
        / grouped["allocated_budget_krw"].clip(lower=1.0)
    )
    return grouped.sort_values("allocated_budget_krw", ascending=False).reset_index(drop=True)


def _excluded_segments(state: Dict[str, Any]) -> pd.DataFrame:
    """Return segments the LP gave 0 allocation to (auto-excluded by the model).

    Reads from the original per-customer LP file so the customer counts
    reflect the full population, not just the eligible subset.
    """
    bo_path = RESULTS_DIR / "budget_optimization.csv"
    if not bo_path.exists():
        return pd.DataFrame()
    try:
        bo_full = pd.read_csv(
            bo_path, usecols=["allocated_budget", "segment"]
        )
    except Exception:  # pragma: no cover - defensive
        return pd.DataFrame()
    if "segment" not in bo_full.columns:
        return pd.DataFrame()
    excluded = bo_full[bo_full["allocated_budget"] <= 0]
    if excluded.empty:
        return pd.DataFrame()
    return (
        excluded.groupby("segment", as_index=False)
        .agg(customers=("allocated_budget", "size"))
        .sort_values("customers", ascending=False)
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_full_krw(x: Any) -> str:
    """Format a KRW amount with full digits, comma separator, no SI suffix."""
    if x is None:
        return "—"
    try:
        n = float(x)
    except (TypeError, ValueError):
        return "—"
    if n != n or n in (float("inf"), float("-inf")):
        return "—"
    return f"₩{n:,.0f}"


def _scenario_card_html(
    title: str, icon: str, color: str, rows: list[tuple[str, str]],
    highlight: bool = False,
) -> str:
    """Build a uniform-styled scenario card via inline HTML.

    Returns a markdown string suitable for ``st.markdown(unsafe_allow_html=True)``.
    """
    border = f"3px solid {color}" if highlight else "1px solid rgba(120,120,120,0.25)"
    shadow = "0 4px 18px rgba(0,0,0,0.10)" if highlight else "0 1px 3px rgba(0,0,0,0.04)"
    bg = f"{color}10" if highlight else "rgba(255,255,255,0.03)"
    rows_html = "".join(
        f"<div style='display:flex;justify-content:space-between;"
        f"padding:6px 0;border-bottom:1px dashed rgba(120,120,120,0.18);"
        f"font-size:14px;'>"
        f"<span style='opacity:0.75'>{label}</span>"
        f"<span style='font-weight:600'>{value}</span>"
        f"</div>"
        for label, value in rows
    )
    return (
        f"<div style='border:{border};border-radius:14px;padding:18px 22px;"
        f"background:{bg};box-shadow:{shadow};height:100%;'>"
        f"<div style='font-size:22px;font-weight:700;color:{color};"
        f"margin-bottom:8px;'>{icon} {title}</div>"
        f"{rows_html}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# render_demo
# ---------------------------------------------------------------------------

def render_demo(st_module, config: Dict[str, Any], data_loader=None) -> None:
    """Render the at-a-glance demo / showcase page (presentation-grade)."""
    st = st_module
    lang = get_lang()
    _tr = lambda s: tr(s, lang)

    st.title(f"🚀 {_tr('Demo: Coupon Recommendation Impact on CLV')}")
    st.caption(
        _tr(
            "Move the budget slider to see how our coupon recommendations "
            "convert spend into customer lifetime value in real time. "
            "Numbers come from the same LP solution shown on the Budget "
            "Optimization page."
        )
    )

    bo, wh = _load_lp_artifacts()
    if bo is None or bo.empty:
        st.warning(
            _tr(
                "No LP solution available yet. Run the pipeline first: "
                "`docker compose up pipeline`"
            )
        )
        return

    state = _prepare_state(bo, wh)
    if state is None:
        st.warning(
            _tr(
                "LP artefact is present but contains no allocated budget. "
                "Rerun the pipeline with a non-zero budget."
            )
        )
        return

    # Uniform-treatment CLV summary (artefact from run_uniform_treatment_clv).
    uniform_payload: Dict[str, Any] = {}
    if data_loader is not None and hasattr(data_loader, "load_uniform_treatment_clv"):
        try:
            uniform_payload = data_loader.load_uniform_treatment_clv() or {}
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("demo_view: load_uniform_treatment_clv failed: %s", e)
    # Inject uniform saved total into state so _simulate_uniform can scale.
    state["uniform_saved_total"] = float(uniform_payload.get("delta_clv", 0.0))

    lp_cost = float(state["lp_total_cost"])
    lp_saved = float(state["lp_total_saved"])
    lp_n = int(state["lp_n"])
    n_total = int(state["n_total"])
    uniform_cost_total = float(state["uniform_cost_total"])
    budget_max = float(state["budget_max"])
    default_budget = float(lp_cost) if lp_cost > 0 else budget_max / 2.0
    step = max(round(budget_max / 200.0, -3), 1000.0)
    baseline_clv = float(state["baseline_clv"])

    # ------------------------------------------------------------------
    # 1. Budget slider (placed before hero so hero reacts to its value)
    # ------------------------------------------------------------------
    st.markdown("### 💰 " + _tr("Adjust budget"))
    budget = st.slider(
        label=_tr("Budget (KRW)"),
        min_value=0.0,
        max_value=budget_max,
        value=default_budget,
        step=step,
        format="%.0f",
        key="demo_budget_slider",
        help=_tr(
            "Slider range: 0 to the 200 % LP what-if scenario. The headline "
            "Budget Optimization number lives at the 100 % anchor."
        ),
    )
    summary = _simulate(state, float(budget))

    # ------------------------------------------------------------------
    # 2. Hero KPI — the one-screen punch line
    # ------------------------------------------------------------------
    hero_spend = _format_full_krw(summary["spend"])
    hero_net = _format_full_krw(summary["net_gain"])
    hero_roi = (
        f"{summary['roi_multiple']:.2f}x"
        if summary["roi_multiple"] > 0
        else "—"
    )
    hero_n = f"{summary['n_treated']:,}"
    hero_caption = (
        f"{_tr('won-for-won return')}: "
        f"<b>{summary['roi_multiple']:.2f}{_tr('won per won')}</b>"
        if summary["roi_multiple"] > 0
        else "—"
    )
    hero_label = _tr("Today's headline")
    hero_invested = _tr("invested →")
    hero_targeted = _tr("customers precisely targeted")
    st.markdown(
        f"""
<div style='border-radius:18px;padding:28px 32px;margin:18px 0 28px 0;
            background:linear-gradient(135deg,#1f77b410,#2ca02c14);
            border:1px solid rgba(31,119,180,0.25);'>
  <div style='font-size:14px;letter-spacing:0.05em;text-transform:uppercase;
              opacity:0.7;margin-bottom:6px;'>{hero_label}</div>
  <div style='font-size:18px;opacity:0.85;margin-bottom:4px;'>
    {hero_spend} {hero_invested}
  </div>
  <div style='font-size:56px;font-weight:800;line-height:1.05;
              color:#1f77b4;margin:2px 0 6px 0;'>{hero_net}</div>
  <div style='font-size:20px;opacity:0.85;'>
    <b style='color:#2ca02c;'>{hero_roi} ROI</b> · {hero_n} {hero_targeted}
  </div>
  <div style='font-size:13px;opacity:0.7;margin-top:10px;'>{hero_caption}</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # 3. Three scenario cards — 무대응 / 무차별 / AI 타겟팅
    # ------------------------------------------------------------------
    st.markdown("### " + _tr("Three scenarios side by side"))
    st.caption(
        _tr(
            "All three scenarios are evaluated at the same budget the "
            "slider is set to — drag it to see them all move together."
        )
    )

    # Uniform scenario — proportionally scaled to current slider budget.
    uni = _simulate_uniform(state, float(budget))

    # Total CLV per scenario (baseline + scenario-specific saved revenue).
    # Anchors all three cards on the same absolute scale so 무대응 is no
    # longer a row of useless zeros — it shows the baseline CLV everyone
    # starts from.
    total_clv_no = baseline_clv
    total_clv_uni = baseline_clv + uni["saved"]
    total_clv_ai = baseline_clv + summary["delta_clv"]

    no_card = _scenario_card_html(
        title=_tr("No intervention"),
        icon="⛔",
        color="#888888",
        rows=[
            (_tr("Total CLV"), _format_full_krw(total_clv_no)),
            (_tr("Customers treated"), "0"),
            (_tr("Coupon spend"), "₩0"),
            (_tr("Saved revenue"), "₩0"),
            (_tr("Net gain"), "₩0"),
            (_tr("ROI"), "—"),
        ],
    )
    uni_card = _scenario_card_html(
        title=_tr("Spray & pray (uniform)"),
        icon="❌",
        color="#d62728",
        rows=[
            (_tr("Total CLV"), _format_full_krw(total_clv_uni)),
            (_tr("Customers treated"), f"{uni['n_treated']:,}"),
            (_tr("Coupon spend"), _format_full_krw(uni["spend"])),
            (_tr("Saved revenue"), _format_full_krw(uni["saved"])),
            (_tr("Net gain"), _format_full_krw(uni["net_gain"])),
            (_tr("ROI"), f"{uni['roi_multiple']:.2f}x" if uni["roi_multiple"] else "—"),
        ],
    )
    ai_card = _scenario_card_html(
        title=_tr("AI targeting (ours)"),
        icon="✅",
        color="#2ca02c",
        rows=[
            (_tr("Total CLV"), _format_full_krw(total_clv_ai)),
            (_tr("Customers treated"), f"{summary['n_treated']:,}"),
            (_tr("Coupon spend"), _format_full_krw(summary["spend"])),
            (_tr("Saved revenue"), _format_full_krw(summary["delta_clv"])),
            (_tr("Net gain"), _format_full_krw(summary["net_gain"])),
            (
                _tr("ROI"),
                f"{summary['roi_multiple']:.2f}x"
                if summary["roi_multiple"]
                else "—",
            ),
        ],
        highlight=True,
    )

    cols = st.columns(3)
    cols[0].markdown(no_card, unsafe_allow_html=True)
    cols[1].markdown(uni_card, unsafe_allow_html=True)
    cols[2].markdown(ai_card, unsafe_allow_html=True)

    if (
        uni["roi_multiple"] > 0
        and summary["roi_multiple"] > 0
        and summary["spend"] > 0
    ):
        # Apples-to-apples at same budget: AI vs uniform at the slider value.
        extra_net = summary["net_gain"] - uni["net_gain"]
        roi_lift_pct = (
            summary["roi_multiple"] / uni["roi_multiple"] - 1.0
        ) * 100.0
        st.success(
            _tr(
                "At the same budget, AI generates **{extra}** more "
                "net gain than uniform spray (**{lift:+.0f}%** higher ROI)."
            ).format(extra=_format_full_krw(extra_net), lift=roi_lift_pct)
        )

    st.markdown("---")

    # ------------------------------------------------------------------
    # 4. Simplified chart — single y-axis area, sweet spot, slider line
    # ------------------------------------------------------------------
    curve = _build_curve(state, n_points=200)
    anchors = state["anchors"]

    # Sweet spot = budget that maximises net gain in the explored range.
    if not curve.empty:
        sweet_idx = int(np.argmax(curve["net_gain"].values))
        sweet_budget = float(curve["budget"].iloc[sweet_idx])
        sweet_net = float(curve["net_gain"].iloc[sweet_idx])
    else:
        sweet_idx = -1
        sweet_budget = lp_cost
        sweet_net = lp_saved - lp_cost

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["budget"],
            y=curve["net_gain"],
            name=_tr("Net gain"),
            mode="lines",
            line=dict(color="#1f77b4", width=3),
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.18)",
        )
    )
    # LP what-if anchors as small grey markers (verification, not focus).
    if anchors:
        anchor_pretty = {
            "budget_50pct": _tr("LP @ 50%"),
            "budget_100pct": _tr("LP @ 100%"),
            "budget_200pct": _tr("LP @ 200%"),
        }
        xs = [anchors[k]["budget"] for k in anchors]
        ys = [anchors[k]["saved"] - anchors[k]["budget"] for k in anchors]
        labels = [anchor_pretty.get(k, k) for k in anchors]
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="markers", name=_tr("LP what-if anchors"),
                marker=dict(color="#888888", size=8, symbol="diamond"),
                hovertext=labels, hoverinfo="text",
            )
        )
    # Sweet spot (max net gain).
    if sweet_idx >= 0:
        fig.add_trace(
            go.Scatter(
                x=[sweet_budget],
                y=[sweet_net],
                mode="markers+text",
                name=_tr("Optimal budget"),
                marker=dict(color="#2ca02c", size=20, symbol="star",
                            line=dict(color="white", width=2)),
                text=[_tr("Optimal")],
                textposition="top center",
                hovertext=[
                    f"{_tr('Optimal')}: {_format_full_krw(sweet_budget)} → "
                    f"{_format_full_krw(sweet_net)} {_tr('net gain')}"
                ],
                hoverinfo="text",
            )
        )
    # Current slider position.
    fig.add_vline(
        x=float(budget),
        line=dict(color="#d62728", width=2, dash="dash"),
        annotation_text=(
            f"{_tr('Current')}: {_format_full_krw(summary['net_gain'])}"
        ),
        annotation_position="top right",
        annotation_font=dict(size=13, color="#d62728"),
    )
    fig.update_layout(
        title=_tr("Budget vs Net Gain (drag the slider above)"),
        xaxis=dict(title=_tr("Budget (KRW)"), tickformat=","),
        yaxis=dict(title=_tr("Net gain (KRW)"), tickformat=","),
        legend=dict(orientation="h", y=-0.22),
        height=420,
        margin=dict(l=40, r=40, t=60, b=60),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 5. Top-3 segment insights — re-sliced live by the slider
    # ------------------------------------------------------------------
    seg_live = _segment_at_budget(state, float(budget))
    if seg_live is not None and not seg_live.empty:
        st.markdown("### 🎯 " + _tr("Where AI concentrated the budget"))
        live_total = float(seg_live["allocated_budget_krw"].sum())
        st.caption(
            _tr(
                "Re-grouped at the current slider budget. Allocation shifts as "
                "you change the budget — the LP picks the next-best segment "
                "once a cheaper one fills up."
            )
        )

        top3 = seg_live.head(3)
        top_roi_row = seg_live.sort_values("roi", ascending=False).head(1)

        ins_cols = st.columns(3)
        colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
        for i, (_, row) in enumerate(top3.iterrows()):
            seg = str(row["segment"])
            alloc = float(row["allocated_budget_krw"])
            cust = int(row["customers"])
            seg_roi = float(row["roi"])
            share = alloc / max(live_total, 1.0) * 100.0
            ins_cols[i].markdown(
                _scenario_card_html(
                    title=seg,
                    icon=f"#{i+1}",
                    color=colors[i],
                    rows=[
                        (_tr("Allocated"), _format_full_krw(alloc)),
                        (_tr("Share of budget"), f"{share:.1f}%"),
                        (_tr("Customers"), f"{cust:,}"),
                        (_tr("ROI"), f"{seg_roi:.2f}x"),
                    ],
                ),
                unsafe_allow_html=True,
            )

        # Top ROI shout-out (independent of allocation)
        if not top_roi_row.empty:
            tr_row = top_roi_row.iloc[0]
            st.info(
                "⭐ "
                + _tr("Highest ROI segment: ")
                + f"**{tr_row['segment']}** — "
                + f"{int(tr_row['customers']):,} {_tr('customers')}, "
                + _format_full_krw(float(tr_row["allocated_budget_krw"]))
                + " "
                + _tr("spend, ")
                + f"**{float(tr_row['roi']):.2f}x ROI**"
            )

        # Excluded segments — independent of slider (LP zero-allocation rows).
        excluded = _excluded_segments(state)
        if not excluded.empty:
            ex_total = int(excluded["customers"].sum())
            ex_names = ", ".join(
                str(s) for s in excluded["segment"].astype(str).tolist()
            )
            st.warning(
                "❎ "
                + _tr("AI auto-excluded ")
                + f"**{ex_total:,}** "
                + _tr("customers (")
                + ex_names
                + ") — "
                + _tr("treating them would either waste budget or hurt retention.")
            )

        # Full segment table (re-grouped at current budget) — collapsed
        with st.expander("📋 " + _tr("Show full segment allocation table at current budget")):
            display = pd.DataFrame(
                {
                    _tr("Segment"): seg_live["segment"].astype(str),
                    _tr("Allocated Budget"): seg_live["allocated_budget_krw"].apply(
                        format_currency_krw
                    ),
                    _tr("Customers"): seg_live["customers"].apply(
                        lambda v: f"{int(v):,}" if pd.notna(v) else "—"
                    ),
                    _tr("Expected Retained"): (
                        seg_live["expected_retained"].apply(
                            lambda v: f"{float(v):,.2f}" if pd.notna(v) else "—"
                        )
                        if "expected_retained" in seg_live.columns
                        else "—"
                    ),
                    _tr("Expected Revenue Saved"): seg_live[
                        "expected_revenue_saved_krw"
                    ].apply(format_currency_krw),
                    "ROI": seg_live["roi"].apply(
                        lambda v: f"{float(v):.2f}x" if pd.notna(v) else "—"
                    ),
                }
            )
            st.dataframe(display, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # 6. Reconciliation badge (kept — useful credibility marker)
    # ------------------------------------------------------------------
    st.info(
        _tr("Anchor point: at the LP's nominal budget ")
        + f"**{format_currency_krw(lp_cost)}** "
        + _tr("the saved revenue equals ")
        + f"**{format_currency_krw(lp_saved)}** "
        + _tr("— this is the headline figure on the Budget Optimization page.")
    )

    # ------------------------------------------------------------------
    # 7. Conclusion line — apples-to-apples at the current slider budget
    # ------------------------------------------------------------------
    if (
        uni["roi_multiple"] > 0
        and summary["roi_multiple"] > 0
        and summary["spend"] > 0
    ):
        extra_net = summary["net_gain"] - uni["net_gain"]
        roi_lift_pct = (
            summary["roi_multiple"] / uni["roi_multiple"] - 1.0
        ) * 100.0
        conclusion_body = _tr(
            "At a budget of <b>{budget}</b>, AI picks <b>{selected}</b> "
            "of <b>{total}</b> customers and delivers <b>{extra}</b> "
            "more net gain than uniform spray (<b>{lift:+.0f}%</b> higher ROI)."
        ).format(
            budget=_format_full_krw(summary["spend"]),
            selected=f"{summary['n_treated']:,}",
            total=f"{n_total:,}",
            extra=_format_full_krw(extra_net),
            lift=roi_lift_pct,
        )
        st.markdown(
            "<div style='border-radius:14px;padding:18px 22px;margin:24px 0;"
            "background:linear-gradient(90deg,#2ca02c14,#1f77b414);"
            "border-left:5px solid #2ca02c;font-size:16px;'>"
            "💡 <b>" + _tr("Conclusion") + ":</b> "
            + conclusion_body
            + "</div>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # 8. Method (collapsed by default)
    # ------------------------------------------------------------------
    with st.expander(_tr("How is this computed?")):
        st.markdown(
            _tr(
                "- **Baseline CLV** = Σ `clv × (1 − churn_prob)` across all "
                "customers in `budget_optimization.csv`.\n"
                "- **Selection rule** = ROI-per-won greedy on the LP "
                "solution. Customers the LP has chosen are ranked by "
                "`expected_revenue_saved_krw / allocated_budget` and "
                "treated in order until the budget is exhausted.\n"
                "- **At slider = LP nominal budget** the greedy selection "
                "trivially equals the full LP allocation, so the headline "
                "**CLV after recommendations** and **Net gain** match the "
                "Budget Optimization page exactly.\n"
                "- **Above the LP cap** the curve linearly interpolates "
                "between the 100 % and 200 % what-if scenarios from "
                "`budget_whatif.csv` — these are independent LP reruns "
                "shown as grey diamonds on the chart.\n"
                "- **Uniform-spend cost** = Σ `cost_per_action` across all "
                "20 000 customers (what we'd spend if everyone got their "
                "segment's coupon).\n"
                "- **Net gain** = Σ `expected_revenue_saved_krw` − Σ "
                "`allocated_budget` of treated customers."
            )
        )
