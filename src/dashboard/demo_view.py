"""
Demo View — At-a-glance project showcase page.

Tells the headline business story of the system in four big KPI cards:

1.  **Baseline CLV**  — total expected CLV with NO coupon intervention,
    computed as ``Σ clv * (1 - churn_probability)`` across all customers.
2.  **CLV after recommendations** — baseline + Σ ``expected_revenue_saved``
    over the customers selected greedily by ROI-per-won from the LP
    optimization solution until the budget slider is exhausted.
3.  **Net gain** — Σ ``expected_revenue_saved`` − Σ ``allocated_budget``
    for the same selection (i.e. ΔCLV minus coupon spend).
4.  **Customers treated** — how many out of the LP-selected customers
    fit under the current budget.

A draggable budget slider lets the viewer move budget between 0 and the
200 % LP what-if scenario (typically ₩100M). The CLV / net-gain curve
below the cards is pre-computed once on a 200-point grid so the slider
feedback is sub-millisecond.

Data sources (kept consistent with the **Budget Optimization** page):

* ``results/budget_optimization.csv`` — per-customer LP allocation
  (``allocated_budget``, ``expected_revenue_saved_krw``, ``clv``,
  ``churn_prob``). Used for the greedy selection on the 0 → LP-cap
  segment of the curve.
* ``results/budget_whatif.csv`` — 50 % / 100 % / 200 % what-if scenarios
  the LP was rerun against. Used as anchor points above the LP cap
  (linearly interpolated) and as labelled markers on the curve so the
  viewer can verify the headline number matches the Budget Optimization
  page exactly.

Falls back to a friendly warning when the LP artefact has not yet been
produced by the pipeline.
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


def _prepare_state(
    bo: pd.DataFrame, wh: Optional[pd.DataFrame]
) -> Optional[Dict[str, Any]]:
    """Build greedy arrays + whatif anchor metadata.

    Returns ``None`` when the LP artefact is missing required columns or
    has zero allocated budget (i.e. pipeline has not yet produced a usable
    solution).
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

    cost = pd.to_numeric(bo["allocated_budget"], errors="coerce").fillna(0.0).to_numpy(float)
    saved = pd.to_numeric(bo["expected_revenue_saved_krw"], errors="coerce").fillna(0.0).to_numpy(float)

    eligible = (cost > 0.0) & (saved > 0.0)
    cost_e = cost[eligible]
    saved_e = saved[eligible]

    if cost_e.size == 0:
        return None

    # ROI per won — tie-break by absolute saved so high-impact rows go first.
    roi = saved_e / np.maximum(cost_e, 1.0)
    order = np.lexsort((-saved_e, -roi))
    cost_sorted = cost_e[order]
    saved_sorted = saved_e[order]
    cum_cost = np.cumsum(cost_sorted)
    cum_saved = np.cumsum(saved_sorted)
    lp_total_cost = float(cum_cost[-1])
    lp_total_saved = float(cum_saved[-1])
    lp_n = int(cost_e.size)

    # What-if anchors (3 LP reruns at 50 % / 100 % / 200 % of nominal).
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

    # Slider ceiling — favour the 200 % what-if when available so the demo
    # can visualise the "what if we doubled the budget" story; otherwise
    # cap at 1.5× the LP solution.
    if "budget_200pct" in anchors:
        budget_max = float(anchors["budget_200pct"]["budget"])
    else:
        budget_max = float(lp_total_cost * 1.5)

    return {
        "baseline_clv": baseline_clv,
        "cum_cost": cum_cost,
        "cum_saved": cum_saved,
        "lp_total_cost": lp_total_cost,
        "lp_total_saved": lp_total_saved,
        "lp_n": lp_n,
        "anchors": anchors,
        "budget_max": budget_max,
    }


def _simulate(state: Dict[str, Any], budget: float) -> Dict[str, float]:
    """Greedy on LP per-customer solution up to ``lp_total_cost``; linear
    interpolation on the what-if anchors above that.

    ``customers_treated`` follows the same rule — exact greedy count up to
    the LP cap, interpolated whatif customer count beyond it.
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
            "baseline_clv": baseline,
            "post_clv": baseline,
            "delta_clv": 0.0,
            "spend": 0.0,
            "net_gain": 0.0,
            "n_treated": 0,
            "roi_multiple": 0.0,
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
        # Above LP cap — linearly interpolate between 100% and 200% whatif.
        a100 = anchors.get("budget_100pct")
        a200 = anchors.get("budget_200pct")
        if a100 and a200 and a200["budget"] > a100["budget"]:
            x0, x1 = a100["budget"], a200["budget"]
            y0_saved, y1_saved = a100["saved"], a200["saved"]
            y0_n, y1_n = a100["customers"], a200["customers"]
            x = float(min(budget, x1))
            frac = (x - x0) / (x1 - x0)
            saved = y0_saved + frac * (y1_saved - y0_saved)
            n = int(round(y0_n + frac * (y1_n - y0_n)))
            spend = x
        else:
            spend, saved, n = lp_cost, lp_saved, lp_n

    post = baseline + saved
    net = saved - spend
    roi = (saved / spend) if spend > 0 else 0.0
    return {
        "baseline_clv": baseline,
        "post_clv": post,
        "delta_clv": saved,
        "spend": spend,
        "net_gain": net,
        "n_treated": n,
        "roi_multiple": roi,
    }


def _format_full_krw(x: Any) -> str:
    """Format a KRW amount with full digits, comma separator, no SI suffix.

    Example: ``142_155_554`` -> ``"₩142,155,554"``. ``None`` / ``NaN`` /
    ``inf`` render as ``"—"`` (matches ``format_currency_krw`` fallback).
    """
    if x is None:
        return "—"
    try:
        n = float(x)
    except (TypeError, ValueError):
        return "—"
    if n != n or n in (float("inf"), float("-inf")):
        return "—"
    return f"₩{n:,.0f}"


def _build_curve(state: Dict[str, Any], n_points: int = 200) -> pd.DataFrame:
    """Pre-compute the budget → CLV / net-gain curve."""
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


def render_demo(st_module, config: Dict[str, Any], data_loader=None) -> None:
    """Render the at-a-glance demo / showcase page."""
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

    lp_cost = float(state["lp_total_cost"])
    lp_saved = float(state["lp_total_saved"])
    lp_n = int(state["lp_n"])
    budget_max = float(state["budget_max"])
    default_budget = float(lp_cost) if lp_cost > 0 else budget_max / 2.0
    step = max(round(budget_max / 200.0, -3), 1000.0)

    # ------------------------------------------------------------------
    # Budget slider — placed BEFORE the KPI cards so the cards reflect
    # the current selection without an extra rerun lag.
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
    # 5 large KPI cards
    # ------------------------------------------------------------------
    # Pull the uniform-treatment CLV artefact (added by data_loader). When
    # the pipeline has not been rerun yet the loader returns ``{}`` and we
    # gracefully degrade the middle card to "—" rather than crashing.
    uniform_clv: Dict[str, Any] = {}
    if data_loader is not None and hasattr(data_loader, "load_uniform_treatment_clv"):
        try:
            uniform_clv = data_loader.load_uniform_treatment_clv() or {}
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("demo_view: load_uniform_treatment_clv failed: %s", e)
            uniform_clv = {}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        label=_tr("Uncouponed CLV (baseline)"),
        value=format_currency_krw(summary["baseline_clv"]),
    )
    if uniform_clv:
        c2.metric(
            label=_tr("Uniform-Treatment CLV (avg coupon for all)"),
            value=format_currency_krw(uniform_clv.get("uniform_treatment_clv")),
            delta=format_currency_krw(uniform_clv.get("delta_clv")),
        )
    else:
        c2.metric(
            label=_tr("Uniform-Treatment CLV (avg coupon for all)"),
            value="—",
            delta=_tr("Run pipeline to compute"),
        )
    c3.metric(
        label=_tr("CLV after recommendations"),
        value=format_currency_krw(summary["post_clv"]),
        delta=format_currency_krw(summary["delta_clv"]),
    )
    c4.metric(
        label=_tr("Net gain (ΔCLV − coupon cost)"),
        value=_format_full_krw(summary["net_gain"]),
        delta=(
            f"{summary['roi_multiple']:.2f}x ROI"
            if summary["roi_multiple"] > 0
            else "—"
        ),
    )
    c5.metric(
        label=_tr("Customers treated"),
        value=f"{summary['n_treated']:,} / {lp_n:,}",
        delta=format_currency_krw(summary["spend"]) + " " + _tr("spent"),
    )

    # Reconciliation badge — keeps the viewer oriented when comparing
    # against the Budget Optimization page.
    st.info(
        _tr(
            "Anchor point: at the LP's nominal budget "
        )
        + f"**{format_currency_krw(lp_cost)}** "
        + _tr("the saved revenue equals ")
        + f"**{format_currency_krw(lp_saved)}** "
        + _tr("— this is the headline figure on the Budget Optimization page.")
    )

    st.markdown("---")

    # ------------------------------------------------------------------
    # Pre-computed curve + anchor markers
    # ------------------------------------------------------------------
    curve = _build_curve(state, n_points=200)
    baseline_clv = float(state["baseline_clv"])
    anchors = state["anchors"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["budget"],
            y=curve["post_clv"],
            name=_tr("CLV after recommendations"),
            line=dict(color="#1f77b4", width=3),
            mode="lines",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve["budget"],
            y=curve["net_gain"],
            name=_tr("Net gain"),
            line=dict(color="#2ca02c", width=3, dash="dot"),
            mode="lines",
            yaxis="y2",
        )
    )
    # LP what-if anchors as orange markers (independent LP reruns at
    # different budget caps — verify the curve passes through them).
    if anchors:
        anchor_pretty = {
            "budget_50pct": _tr("LP @ 50%"),
            "budget_100pct": _tr("LP @ 100%"),
            "budget_200pct": _tr("LP @ 200%"),
        }
        xs, ys, labels = [], [], []
        for key, meta in anchors.items():
            xs.append(meta["budget"])
            ys.append(baseline_clv + meta["saved"])
            labels.append(
                f"{anchor_pretty.get(key, key)} — "
                f"{format_currency_krw(meta['saved'])} "
                f"({meta['customers']:,} {_tr('customers')})"
            )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                name=_tr("LP what-if anchors"),
                marker=dict(color="#ff7f0e", size=11, symbol="diamond"),
                text=[anchor_pretty.get(k, k) for k in anchors.keys()],
                textposition="top center",
                hovertext=labels,
                hoverinfo="text",
                yaxis="y1",
            )
        )
    # Horizontal baseline reference
    fig.add_hline(
        y=baseline_clv,
        line=dict(color="#aaaaaa", width=1, dash="dash"),
        annotation_text=_tr("Baseline CLV"),
        annotation_position="top left",
        yref="y1",
    )
    # Current slider position
    fig.add_vline(
        x=float(budget),
        line=dict(color="#d62728", width=2, dash="dash"),
        annotation_text=_tr("Current budget"),
        annotation_position="top right",
    )
    fig.update_layout(
        title=_tr("Budget → CLV / Net Gain (LP-grounded)"),
        xaxis=dict(title=_tr("Budget (KRW)")),
        yaxis=dict(
            title=_tr("CLV after recommendations (KRW)"),
            tickformat=",",
        ),
        yaxis2=dict(
            title=_tr("Net gain (KRW)"),
            overlaying="y",
            side="right",
            tickformat=",",
            showgrid=False,
        ),
        legend=dict(orientation="h", y=-0.25),
        height=460,
        margin=dict(l=40, r=40, t=60, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Segment-level budget allocation table — read directly from the LP
    # roll-up artefact so it stays in lock-step with the Budget Optimization
    # page rather than re-aggregating here.
    # ------------------------------------------------------------------
    seg_path = RESULTS_DIR / "budget_results.csv"
    if seg_path.exists():
        try:
            seg_df = pd.read_csv(seg_path)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("demo_view: failed to read %s: %s", seg_path, e)
            seg_df = None
        if seg_df is not None and not seg_df.empty:
            st.markdown("### 📊 " + _tr("Segment-level Budget Allocation"))
            display = pd.DataFrame(
                {
                    _tr("Segment"): seg_df["segment"].astype(str),
                    _tr("Allocated Budget"): seg_df["allocated_budget_krw"].apply(
                        format_currency_krw
                    ),
                    _tr("Customers"): seg_df["customers"].apply(
                        lambda v: f"{int(v):,}" if pd.notna(v) else "—"
                    ),
                    _tr("Expected Retained"): seg_df["expected_retained"].apply(
                        lambda v: f"{float(v):,.2f}" if pd.notna(v) else "—"
                    ),
                    _tr("Expected Revenue Saved"): seg_df[
                        "expected_revenue_saved_krw"
                    ].apply(format_currency_krw),
                    "ROI": seg_df["roi"].apply(
                        lambda v: f"{float(v):.2f}x" if pd.notna(v) else "—"
                    ),
                }
            )
            st.dataframe(display, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Explanatory text — keeps the page demo-friendly and reinforces the
    # consistency with the Budget Optimization page.
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
                "shown as orange diamonds on the chart.\n"
                "- **Net gain** = Σ `expected_revenue_saved_krw` − Σ "
                "`allocated_budget` of treated customers."
            )
        )
