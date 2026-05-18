"""
Enhanced Recommendations Dashboard View.

Provides a comprehensive view of personalized retention recommendations
including KPI summary, distribution analysis, segment breakdowns,
cost-benefit analysis, and filterable recommendation tables.

All configurable parameters are sourced from config/simulator_config.yaml.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# i18n (iter15 AGENT D) — defensive import, module-level closure.
# -------------------------------------------------------------------------
try:
    from src.dashboard.utils.dashboard_helpers import get_lang, tr

    def _tr(s: str) -> str:
        try:
            return tr(s, get_lang())
        except Exception:
            return s
except Exception:  # pragma: no cover - defensive fallback
    def _tr(s: str) -> str:
        return s


# Defensively import shared helpers (F1 work). Fall back to minimal
# inline formatters so this page renders even if helpers are unavailable.
try:
    from src.dashboard.utils.dashboard_helpers import (
        compute_overall_roi,
        format_count,
        format_currency_krw,
    )
except Exception:  # pragma: no cover - defensive fallback
    def format_count(value: Any, integer: bool = True, suffix: str = "") -> str:
        if value is None:
            return "—"
        try:
            if isinstance(value, float) and (value != value):
                return "—"
            if integer:
                out = f"{int(value):,}"
            else:
                out = f"{float(value):,.1f}"
        except Exception:
            return "—"
        return f"{out}{suffix}" if suffix else out

    def format_currency_krw(x: Any) -> str:
        if x is None:
            return "—"
        try:
            n = float(x)
        except Exception:
            return "—"
        if n != n:
            return "—"
        if abs(n) >= 1_000_000_000:
            return f"₩{n / 1_000_000_000:,.2f}B"
        if abs(n) >= 1_000_000:
            return f"₩{n / 1_000_000:,.1f}M"
        if abs(n) >= 1_000:
            return f"₩{n / 1_000:,.1f}K"
        return f"₩{n:,.0f}"

    def compute_overall_roi(
        revenue_saved: float,
        cost_or_budget: float,
        scope_label: str = "budget",
    ) -> Dict[str, Any]:
        label_map = {
            "budget": "ROI (budget envelope)",
            "treated": "ROI (treated only)",
            "segment_avg": "Avg per-segment ROI",
        }
        label = label_map.get(scope_label, "ROI")
        try:
            if not cost_or_budget:
                return {"value": None, "display": "—", "label": label,
                        "tooltip": "denominator is zero"}
            val = float(revenue_saved) / float(cost_or_budget)
            return {
                "value": val,
                "display": f"{val:.2f}x",
                "label": label,
                "tooltip": (
                    f"{float(revenue_saved):,.0f} ÷ "
                    f"{float(cost_or_budget):,.0f}"
                ),
            }
        except Exception:
            return {"value": None, "display": "—", "label": label,
                    "tooltip": "computation failed"}

# -------------------------------------------------------------------------
# iter13 G4: defensive DashboardArtifact integration. G2 may not yet have
# shipped — we import optimistically and fall back to legacy behavior.
# -------------------------------------------------------------------------
try:  # pragma: no cover - defensive import
    from src.dashboard.data_loader import DashboardArtifact  # type: ignore
except Exception:  # pragma: no cover
    DashboardArtifact = None  # type: ignore[assignment]


def _load_artifact_safely(loader_callable, *args, **kwargs):
    """Try a data_loader method with ``as_artifact=True`` (legacy-safe).

    Returns ``(payload, artifact_or_none)``. Falls back to the legacy
    return shape when the loader does not yet accept ``as_artifact``.
    """
    if loader_callable is None:
        return None, None
    try:
        result = loader_callable(*args, as_artifact=True, **kwargs)
    except TypeError:
        try:
            payload = loader_callable(*args, **kwargs)
        except Exception:
            return None, None
        return payload, None
    except Exception:
        return None, None
    payload = getattr(result, "data", result)
    return payload, result


def _artifact_marked_unreal(artifact: Any) -> bool:
    """Return True when ``artifact.is_real`` is explicitly False."""
    if artifact is None:
        return False
    is_real = getattr(artifact, "is_real", None)
    if is_real is None:
        return False
    return bool(is_real) is False


# Action type display names and colors
ACTION_COLORS = {
    "coupon": "#2ecc71",
    "push_notification": "#3498db",
    "email_campaign": "#9b59b6",
    "email": "#9b59b6",
    "loyalty_points": "#f39c12",
    "personal_outreach": "#e74c3c",
    "exclusive_offer": "#1abc9c",
    "no_action": "#95a5a6",
}

ACTION_LABELS = {
    "coupon": "Coupon / Discount",
    "push_notification": "Push Notification",
    "email_campaign": "Email Campaign",
    "email": "Email Campaign",
    "loyalty_points": "Loyalty Points",
    "personal_outreach": "Personal Outreach",
    "exclusive_offer": "Exclusive Offer",
    "no_action": "No Action",
}


def render_recommendations_view(st_module, config: Dict, data_loader=None):
    """Render enhanced personalized recommendations page.

    Shows:
    - KPI summary cards (total recs, avg uplift, top action, coverage)
    - Recommendation type distribution (donut chart)
    - Expected uplift by recommendation type (box + bar)
    - Priority score distribution (histogram)
    - Segment-level recommendation breakdown (stacked bar)
    - Cost-benefit analysis with retention offers
    - Filterable recommendation table
    - Top-K prioritized recommendations

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module
    st.header(_tr("Personalized Recommendations"))
    st.markdown(_tr(
        "AI-driven retention recommendations based on churn risk, "
        "CLV, uplift scores, and customer segment affinity."
    ))

    if data_loader is None:
        from src.dashboard.app import get_data_loader
        data_loader = get_data_loader(config)

    # Load data. iter13 G4: probe retention_offers with as_artifact=True so
    # we can distinguish a real artifact from the fixture fallback and hide
    # the mid-strip cost-benefit cards when the underlying CSV is missing.
    recs = data_loader.load_recommendations()
    retention_offers, retention_offers_artifact = _load_artifact_safely(
        getattr(data_loader, "load_retention_offers", None),
    )
    if retention_offers is None:
        retention_offers = pd.DataFrame()
    predictions = data_loader.load_predictions()

    if recs.empty:
        st.warning(_tr("No recommendations available."))
        return

    # ==================================================================
    # Section 1: KPI Summary Cards
    # ==================================================================
    _render_kpi_cards(st, recs, retention_offers)

    # ==================================================================
    # Section 2: Recommendation Distribution
    # ==================================================================
    st.markdown("---")
    _render_distribution_section(st, recs)

    # ==================================================================
    # Section 3: Uplift Analysis
    # ==================================================================
    st.markdown("---")
    _render_uplift_analysis(st, recs)

    # ==================================================================
    # Section 4: Segment Breakdown
    # ==================================================================
    st.markdown("---")
    _render_segment_breakdown(st, recs, predictions)

    # ==================================================================
    # Section 5: Cost-Benefit Analysis
    # ==================================================================
    # iter13 G4: when retention_offers is explicitly NOT real (fixture
    # fallback because results/retention_offers.csv is missing), hide the
    # mid-strip ROI/cost cards and surface a warning instead — the top
    # KPIs above already show real-population stats from recommendations.csv.
    if _artifact_marked_unreal(retention_offers_artifact):
        st.markdown("---")
        st.warning(_tr(
            "Retention offer breakdown not yet computed — top KPIs above "
            "show full population stats from real recommendations.csv."
        ))
    elif not retention_offers.empty:
        st.markdown("---")
        _render_cost_benefit_analysis(st, config, retention_offers)

    # ==================================================================
    # Section 6: Filterable Recommendation Table
    # ==================================================================
    st.markdown("---")
    _render_recommendation_table(st, recs, retention_offers)


# =========================================================================
# Internal rendering helpers
# =========================================================================


def _render_kpi_cards(st, recs: pd.DataFrame, offers: pd.DataFrame):
    """Render KPI summary cards for recommendations.

    Both KPI strips on this page used to show a card labelled
    "Avg Expected Uplift" with two different values (one across all
    customers, one across the treated subset). This top strip explicitly
    scopes the headline uplift to the **full population** so a downstream
    analyst can never confuse it with the mid-page treated-only figure.
    Reconciliation between "High Priority" and treated-coupon counts is
    surfaced inline immediately below the cards.
    """
    kc1, kc2, kc3, kc4 = st.columns(4)

    total_recs = len(recs)
    kc1.metric(_tr("Total Recommendations"), format_count(total_recs))

    if "expected_uplift" in recs.columns:
        avg_uplift_all = recs["expected_uplift"].mean()
        kc2.metric(
            _tr("Avg Predicted Uplift (all customers)"),
            f"{avg_uplift_all:.2%}",
            help=(
                "Mean of expected_uplift across ALL "
                f"{format_count(total_recs)} recommendations (population = "
                "full customer base, including no_action). The mid-page "
                "'Avg Treated Uplift' card uses the treated subset only — "
                "the two should not match."
            ),
        )
    else:
        kc2.metric(_tr("Avg Predicted Uplift (all customers)"), "N/A")

    if "recommendation_type" in recs.columns:
        top_action = recs["recommendation_type"].value_counts().idxmax()
        top_label = ACTION_LABELS.get(top_action, top_action)
        kc3.metric(_tr("Top Action Type"), top_label)
    else:
        kc3.metric(_tr("Top Action Type"), "N/A")

    high_priority_count: Optional[int] = None
    if "priority_score" in recs.columns:
        high_priority_count = int((recs["priority_score"] >= 0.7).sum())
        kc4.metric(
            _tr("High Priority"),
            format_count(high_priority_count),
            help=(
                "Customers with priority_score ≥ 0.70. NOT every "
                "high-priority customer receives a coupon — see the "
                "reconciliation row below for the treated split."
            ),
        )
    elif not offers.empty and "priority_score" in offers.columns:
        kc4.metric(_tr("Offers Generated"), format_count(len(offers)))
    else:
        kc4.metric(_tr("High Priority"), "N/A")

    # Reconciliation row: High Priority vs treated (coupon recipients).
    # Closes the iter9 audit finding that 12,708 high-priority customers
    # silently land in `no_action` without an inline reason.
    if high_priority_count is not None and "recommendation_type" in recs.columns:
        treated_mask = recs["recommendation_type"] != "no_action"
        treated_count = int(treated_mask.sum())
        # Of the high-priority cohort, how many actually received a treatment?
        if "priority_score" in recs.columns:
            hp_treated = int(
                ((recs["priority_score"] >= 0.7) & treated_mask).sum()
            )
        else:
            hp_treated = treated_count

        hp_no_action = max(high_priority_count - hp_treated, 0)
        if high_priority_count > 0:
            treated_pct = hp_treated / high_priority_count * 100.0
            no_action_pct = hp_no_action / high_priority_count * 100.0
        else:
            treated_pct = 0.0
            no_action_pct = 0.0

        st.info(
            f"**{_tr('Priority vs treated reconciliation')}** — "
            f"{_tr('Of')} "
            f"{format_count(high_priority_count)} "
            f"{_tr('high-priority customers,')} "
            f"{format_count(hp_treated)} ({treated_pct:.0f}%) "
            f"{_tr('receive a treatment offer;')} "
            f"{format_count(hp_no_action)} "
            f"({no_action_pct:.0f}%) "
            f"{_tr('get `no_action` because their')} "
            f"**{_tr('predicted uplift × CLV did not exceed the cost threshold')}** "
            f"{_tr('for any offer in the catalog. Total treated across all priorities:')} "
            f"{format_count(treated_count)}."
        )


def _render_distribution_section(st, recs: pd.DataFrame):
    """Render recommendation type distribution with donut and bar charts."""
    st.subheader(_tr("Recommendation Distribution"))

    if "recommendation_type" not in recs.columns:
        st.info(_tr("No recommendation type data available."))
        return

    type_counts = recs["recommendation_type"].value_counts().reset_index()
    type_counts.columns = ["recommendation_type", "count"]

    col_donut, col_bar = st.columns(2)

    with col_donut:
        colors = [
            ACTION_COLORS.get(t, "#95a5a6")
            for t in type_counts["recommendation_type"]
        ]
        fig_donut = go.Figure(data=[go.Pie(
            labels=type_counts["recommendation_type"],
            values=type_counts["count"],
            hole=0.4,
            marker=dict(colors=colors),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>"
                          "Count: %{value}<br>"
                          "Share: %{percent}<extra></extra>",
        )])
        fig_donut.update_layout(
            title=_tr("Recommendation Type Distribution"),
            showlegend=True,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_bar:
        fig_bar = px.bar(
            type_counts,
            x="recommendation_type",
            y="count",
            color="recommendation_type",
            color_discrete_map=ACTION_COLORS,
            title=_tr("Recommendations by Type"),
            labels={
                "recommendation_type": _tr("Action Type"),
                "count": _tr("Number of Recommendations"),
            },
            text="count",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)


def _render_uplift_analysis(st, recs: pd.DataFrame):
    """Render expected uplift analysis by recommendation type."""
    st.subheader(_tr("Expected Uplift Analysis"))

    if "expected_uplift" not in recs.columns:
        st.info(_tr("No uplift data available for recommendations."))
        return

    if "recommendation_type" not in recs.columns:
        # Simple histogram
        fig = px.histogram(
            recs, x="expected_uplift", nbins=30,
            title=_tr("Expected Uplift Distribution"),
            color_discrete_sequence=["#3498db"],
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    col_box, col_avg = st.columns(2)

    with col_box:
        fig_box = px.box(
            recs,
            x="recommendation_type",
            y="expected_uplift",
            color="recommendation_type",
            color_discrete_map=ACTION_COLORS,
            title=_tr("Predicted Uplift Distribution by Action Type"),
            labels={
                "recommendation_type": _tr("Action Type"),
                "expected_uplift": _tr("Predicted Uplift (if treated)"),
            },
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with col_avg:
        avg_uplift = (
            recs.groupby("recommendation_type")["expected_uplift"]
            .mean()
            .reset_index()
        )
        avg_uplift.columns = ["recommendation_type", "avg_uplift"]
        # Drop `no_action` from this chart: its non-zero value is
        # "predicted uplift if treated" for the not-treated population,
        # NOT realized uplift, which conflates two different quantities
        # in a chart whose axis is labeled "Average Expected Uplift".
        # Realized uplift on customers who received no treatment is 0
        # by construction. (iter9 audit finding A5/§4.)
        avg_uplift_treated = avg_uplift[
            avg_uplift["recommendation_type"] != "no_action"
        ].copy()
        avg_uplift_treated = avg_uplift_treated.sort_values(
            "avg_uplift", ascending=False
        )

        if avg_uplift_treated.empty:
            st.info(_tr(
                "No treated actions available — uplift-by-action chart "
                "requires at least one non-`no_action` recommendation."
            ))
        else:
            fig_avg = px.bar(
                avg_uplift_treated,
                x="recommendation_type",
                y="avg_uplift",
                color="recommendation_type",
                color_discrete_map=ACTION_COLORS,
                title=_tr("Average Predicted Uplift by Treated Action"),
                labels={
                    "recommendation_type": _tr("Action Type"),
                    "avg_uplift": _tr("Avg Predicted Uplift (treated)"),
                },
                text=avg_uplift_treated["avg_uplift"].apply(
                    lambda v: f"{v:.2%}"
                ),
            )
            fig_avg.update_traces(textposition="outside")
            fig_avg.update_layout(showlegend=False)
            st.plotly_chart(fig_avg, use_container_width=True)
            st.caption(_tr(
                "`no_action` excluded: realized uplift on untreated "
                "customers is 0 by definition; including its "
                "'predicted uplift if treated' on the same axis as "
                "treated actions is misleading."
            ))

    # Priority score distribution
    if "priority_score" in recs.columns:
        st.markdown(f"#### {_tr('Priority Score Distribution')}")
        fig_priority = px.histogram(
            recs, x="priority_score", nbins=30,
            color_discrete_sequence=["#e67e22"],
            title=_tr("Priority Score Distribution"),
            labels={"priority_score": _tr("Priority Score")},
        )
        mean_priority = recs["priority_score"].mean()
        fig_priority.add_vline(
            x=mean_priority, line_dash="dash", line_color="red",
            annotation_text=f"{_tr('Mean')}: {mean_priority:.2f}",
        )
        st.plotly_chart(fig_priority, use_container_width=True)


def _render_segment_breakdown(
    st, recs: pd.DataFrame, predictions: pd.DataFrame,
):
    """Render segment-level recommendation breakdown."""
    st.subheader(_tr("Segment-Level Breakdown"))

    # Try to merge segment info from predictions if not in recs
    if "segment" not in recs.columns and not predictions.empty:
        if "customer_id" in recs.columns and "customer_id" in predictions.columns:
            merged = recs.merge(
                predictions[["customer_id", "segment"]],
                on="customer_id",
                how="left",
            )
        else:
            merged = recs.copy()
    else:
        merged = recs.copy()

    if "segment" not in merged.columns or "recommendation_type" not in merged.columns:
        st.info(_tr("Segment or recommendation type data not available."))
        return

    # Stacked bar: action types per segment
    cross_tab = pd.crosstab(
        merged["segment"], merged["recommendation_type"],
    ).reset_index()
    action_cols = [c for c in cross_tab.columns if c != "segment"]

    fig_stacked = go.Figure()
    for action in action_cols:
        color = ACTION_COLORS.get(action, "#95a5a6")
        fig_stacked.add_trace(go.Bar(
            name=action,
            x=cross_tab["segment"],
            y=cross_tab[action],
            marker_color=color,
        ))
    fig_stacked.update_layout(
        title=_tr("Recommendation Types by Segment"),
        xaxis_title=_tr("Segment"),
        yaxis_title=_tr("Count"),
        barmode="stack",
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

    # Segment summary stats
    if "expected_uplift" in merged.columns:
        seg_stats = merged.groupby("segment").agg(
            count=("customer_id", "count") if "customer_id" in merged.columns
            else ("expected_uplift", "count"),
            avg_uplift=("expected_uplift", "mean"),
            max_uplift=("expected_uplift", "max"),
        ).reset_index()
        seg_stats["avg_uplift"] = seg_stats["avg_uplift"].round(4)
        seg_stats["max_uplift"] = seg_stats["max_uplift"].round(4)
        st.markdown(f"#### {_tr('Segment Summary')}")
        st.dataframe(seg_stats, use_container_width=True)


def _render_cost_benefit_analysis(
    st, config: Dict, offers: pd.DataFrame,
):
    """Render cost-benefit analysis for retention offers."""
    st.subheader(_tr("Cost-Benefit Analysis"))

    currency = config.get("budget", {}).get("currency", "KRW")

    if offers.empty:
        st.info(_tr("No retention offer data available."))
        return

    # KPI cards for offers
    oc1, oc2, oc3, oc4 = st.columns(4)

    total_cost = (
        float(offers["estimated_cost_krw"].sum())
        if "estimated_cost_krw" in offers.columns else 0.0
    )
    total_revenue = (
        float(offers["expected_revenue_saved_krw"].sum())
        if "expected_revenue_saved_krw" in offers.columns else 0.0
    )
    avg_uplift_treated = (
        float(offers["expected_uplift"].mean())
        if "expected_uplift" in offers.columns else 0.0
    )
    treated_n = int(len(offers))

    # Single source of truth for the "Overall ROI" KPI. Pinning every
    # ROI tile to compute_overall_roi() closes the iter9 audit finding
    # that Pages 05 / 09 / 12 each silently used a different denominator
    # (3.5x / 9.0x / 3.8x). On this page the denominator is the actual
    # cost issued for treated customers, so scope_label="treated".
    roi_info = compute_overall_roi(
        revenue_saved=total_revenue,
        cost_or_budget=total_cost,
        scope_label="treated",
    )

    oc1.metric(
        _tr("Total Campaign Cost"),
        f"{format_currency_krw(total_cost)} ({currency})",
        help=(
            f"Sum of estimated_cost_krw across "
            f"{format_count(treated_n)} issued offers."
        ),
    )
    oc2.metric(
        _tr("Est. Revenue Saved"),
        f"{format_currency_krw(total_revenue)} ({currency})",
        help=(
            "Sum of expected_revenue_saved_krw across the same "
            "treated subset — NOT the campaign budget envelope."
        ),
    )
    oc3.metric(
        _tr(roi_info["label"]),  # e.g. "ROI (treated only)"
        roi_info["display"],
        help=(
            f"{roi_info['tooltip']}. Denominator = actual treated cost "
            "(not the LP budget envelope). Page 05 reports ROI against "
            "the full budget envelope (~3.5x); Page 12 reports ROI "
            "against the planned budget (~3.8x). The three values are "
            "consistent — they use different denominators."
        ),
    )
    oc4.metric(
        _tr("Avg Treated Uplift"),
        f"{avg_uplift_treated:.2%}",
        help=(
            "Mean expected_uplift across the "
            f"{format_count(treated_n)} treated customers only "
            "(coupon / non-no_action). The top-of-page card "
            "'Avg Predicted Uplift (all customers)' averages over "
            "the full population and is intentionally lower."
        ),
    )

    col_cost, col_roi = st.columns(2)

    with col_cost:
        if "offer_type" in offers.columns and "estimated_cost_krw" in offers.columns:
            cost_by_type = (
                offers.groupby("offer_type")["estimated_cost_krw"]
                .sum()
                .reset_index()
            )
            fig_cost = px.bar(
                cost_by_type,
                x="offer_type",
                y="estimated_cost_krw",
                title=f"{_tr('Total Cost by Offer Type')} ({currency})",
                labels={
                    "offer_type": _tr("Offer Type"),
                    "estimated_cost_krw": f"{_tr('Cost')} ({currency})",
                },
                color="offer_type",
                text=cost_by_type["estimated_cost_krw"].apply(
                    lambda v: f"{v:,.0f}"
                ),
            )
            fig_cost.update_traces(textposition="outside")
            fig_cost.update_layout(showlegend=False)
            st.plotly_chart(fig_cost, use_container_width=True)

    with col_roi:
        if all(c in offers.columns for c in ["offer_type", "estimated_cost_krw", "expected_revenue_saved_krw"]):
            roi_by_type = offers.groupby("offer_type").agg(
                total_cost=("estimated_cost_krw", "sum"),
                total_revenue=("expected_revenue_saved_krw", "sum"),
            ).reset_index()
            roi_by_type["roi"] = (
                roi_by_type["total_revenue"] / roi_by_type["total_cost"].clip(lower=1)
            )
            fig_roi = px.bar(
                roi_by_type.sort_values("roi", ascending=False),
                x="offer_type",
                y="roi",
                title=_tr("ROI by Offer Type"),
                labels={
                    "offer_type": _tr("Offer Type"),
                    "roi": _tr("ROI (x)"),
                },
                color="offer_type",
                text=roi_by_type.sort_values("roi", ascending=False)["roi"].apply(
                    lambda v: f"{v:.1f}x"
                ),
            )
            fig_roi.update_traces(textposition="outside")
            fig_roi.update_layout(showlegend=False)
            st.plotly_chart(fig_roi, use_container_width=True)

    # Scatter: cost vs revenue saved, colored by risk
    if all(c in offers.columns for c in ["estimated_cost_krw", "expected_revenue_saved_krw"]):
        color_col = "risk_level" if "risk_level" in offers.columns else None
        size_col = None
        n_neg_uplift = 0
        if "expected_uplift" in offers.columns:
            # Plotly markers require non-negative sizes; uplift can be negative
            # (e.g., predicted negative effect for no-action customers). Clip to 0
            # so those rows render at the minimum marker size instead of crashing.
            offers = offers.copy()
            n_neg_uplift = int((offers["expected_uplift"] < 0).sum())
            offers["size_metric"] = offers["expected_uplift"].clip(lower=0)
            size_col = "size_metric"
        fig_scatter = px.scatter(
            offers,
            x="estimated_cost_krw",
            y="expected_revenue_saved_krw",
            color=color_col,
            size=size_col,
            render_mode="svg",
            title=_tr("Cost vs Revenue Saved per Customer"),
            labels={
                "estimated_cost_krw": f"{_tr('Estimated Cost')} ({currency})",
                "expected_revenue_saved_krw": f"{_tr('Est. Revenue Saved')} ({currency})",
            },
            hover_data=["customer_id"] if "customer_id" in offers.columns else None,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        if n_neg_uplift > 0:
            st.caption(
                _tr("Note: {n:,} rows with negative expected uplift are shown at minimum marker size.").format(
                    n=n_neg_uplift
                )
            )


def _render_recommendation_table(
    st, recs: pd.DataFrame, offers: pd.DataFrame,
):
    """Render filterable recommendation table and top-K list."""
    st.subheader(_tr("Recommendation Details"))

    # Filters
    filter_col1, filter_col2 = st.columns(2)

    display_df = recs.copy()

    with filter_col1:
        if "recommendation_type" in recs.columns:
            all_types = [_tr("All")] + sorted(recs["recommendation_type"].unique().tolist())
            selected_type = st.selectbox(_tr("Filter by Action Type"), all_types)
            if selected_type != _tr("All"):
                display_df = display_df[
                    display_df["recommendation_type"] == selected_type
                ]

    with filter_col2:
        if "priority_score" in recs.columns:
            min_priority = float(st.slider(
                _tr("Minimum Priority Score"),
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
            ))
            display_df = display_df[
                display_df["priority_score"] >= min_priority
            ]

    # Sort by priority
    if "priority_score" in display_df.columns:
        display_df = display_df.sort_values("priority_score", ascending=False)

    st.markdown(
        f"**{_tr('Showing')} {len(display_df)} {_tr('recommendations')}**"
    )
    st.dataframe(display_df, use_container_width=True)

    # Top-K prioritized list
    if "priority_score" in recs.columns:
        st.markdown(f"#### {_tr('Top Priority Recommendations')}")
        top_k = min(10, len(recs))
        top_recs = recs.nlargest(top_k, "priority_score")
        st.dataframe(top_recs, use_container_width=True)

    # Retention offers detail table
    if not offers.empty:
        st.markdown(f"#### {_tr('Detailed Retention Offers')}")
        display_cols = [
            c for c in [
                "customer_id", "segment", "risk_level", "churn_probability",
                "offer_type", "offer_detail", "expected_uplift",
                "estimated_cost_krw", "expected_revenue_saved_krw",
                "priority_score",
            ]
            if c in offers.columns
        ]
        st.dataframe(
            offers[display_cols].head(20),
            use_container_width=True,
        )
