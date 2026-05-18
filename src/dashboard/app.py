"""
Streamlit Dashboard Application for Churn Prediction System.

Provides interactive views for:
- Churn prediction overview with KPI cards
- Model performance comparison (ML, DL, Ensemble)
- Customer segmentation visualization
- Budget optimization with what-if scenarios
- A/B testing results
- Survival analysis curves
- Personalized recommendations
- CLV prediction distribution
- Uplift modeling results
- Real-time scoring status
- MLflow experiment comparison

All configurable parameters are loaded from config/simulator_config.yaml.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml

from src.dashboard.calculations import (
    _build_channel_allocation_data,
    _build_whatif_scenarios,
    _compute_budget_sweep,
    _compute_mde_sensitivity,
    _compute_multiple_comparison_corrections,
    _compute_power_analysis,
    _compute_power_curve,
    _compute_scenario_comparison,
)
from src.dashboard.monitoring_view import (
    render_model_monitoring as render_monitoring_view,
)
from src.dashboard.demo_view import render_demo
from src.dashboard.recommendations_view import render_recommendations_view
from src.dashboard.system_health_view import render_system_health
from src.dashboard.system_health_view import resolve_redis_connection_config
from src.dashboard.utils.dashboard_helpers import (
    PAGES,
    classify_risk,
    format_count,
    format_currency,
    format_percentage,
    get_app_title,
    get_budget_config,
    get_churn_definition,
    get_color_palette,
    get_ensemble_weights,
    get_page_icon,
    get_page_list,
    get_risk_color,
    get_segment_colors,
    build_sidebar_info,
    validate_predictions,
    safe_get_column,
    compute_kpi_delta,
)

# Defensively import iter10 helpers added by F1 agent. Older builds may not
# have these helpers; provide minimal fallbacks so the dashboard keeps
# rendering with reasonable behaviour.
try:
    from src.dashboard.utils.dashboard_helpers import (
        compute_overall_roi,
        drift_trend_guard,
        format_currency_krw,
    )
except ImportError:  # pragma: no cover - defensive fallback
    def compute_overall_roi(revenue_saved, cost_or_budget, scope_label="budget"):
        label_map = {
            "budget": "ROI (budget envelope)",
            "treated": "ROI (treated only)",
            "segment_avg": "Avg per-segment ROI",
        }
        label = label_map.get(scope_label, "ROI")
        try:
            denom = float(cost_or_budget)
            num = float(revenue_saved)
        except (TypeError, ValueError):
            return {"value": None, "display": "—", "label": label, "tooltip": ""}
        if denom == 0:
            return {"value": None, "display": "—", "label": label, "tooltip": "denominator is zero"}
        val = num / denom
        return {
            "value": val,
            "display": f"{val:.2f}x",
            "label": label,
            "tooltip": f"{num:,.0f} ÷ {denom:,.0f}",
        }

    def drift_trend_guard(timeseries, min_points=5):
        if timeseries is None:
            return False, f"Insufficient history — need ≥{min_points} observations, have 0."
        try:
            n = len(timeseries)
        except TypeError:
            n = 0
        if n < min_points:
            return False, f"Insufficient history — need ≥{min_points} observations, have {n}."
        return True, ""

    def format_currency_krw(x):
        try:
            return f"{int(float(x)):,} KRW"
        except (TypeError, ValueError):
            return "—"


# iter13 G3: defensive import of DashboardArtifact (provided by G2). Older
# data_loader builds do not expose this symbol; fall back to None so the
# helper below transparently downgrades to the legacy return-type API.
try:
    from src.dashboard.data_loader import DashboardArtifact  # type: ignore
except ImportError:  # pragma: no cover - defensive fallback
    DashboardArtifact = None  # type: ignore


def _load_as_artifact(data_loader, method_name: str, *args, **kwargs):
    """Call ``data_loader.<method_name>(as_artifact=True, ...)`` defensively.

    Returns a (payload, is_real, missing_reason) tuple regardless of whether
    G2's ``DashboardArtifact`` wrapper is wired up yet. When the loader is
    older / does not understand ``as_artifact``, we fall back to the legacy
    return value and treat it as "is_real=False, reason=fallback" so call
    sites still surface the missing-real-artifact warning expected by
    iter13's audit fixes.

    Args:
        data_loader: DashboardDataLoader instance.
        method_name: Name of the loader method to invoke (e.g.
            ``"load_confusion_matrices"``).
        *args, **kwargs: forwarded to the loader method.

    Returns:
        Tuple ``(payload, is_real, missing_reason)``.
    """
    method = getattr(data_loader, method_name, None)
    if method is None:
        return None, False, f"{method_name} not implemented on data loader"
    try:
        result = method(*args, as_artifact=True, **kwargs)
    except TypeError:
        # Loader signature does not accept as_artifact yet — old data_loader.
        try:
            result = method(*args, **kwargs)
        except Exception as e:  # pragma: no cover - defensive
            return None, False, f"loader error: {e}"
        # Legacy return: cannot tell real-vs-fixture, mark as unknown/fixture
        # so the caller falls through to the "missing real artifact" branch.
        return result, False, (
            "Real-artifact flag unavailable on this data_loader build; "
            "treating as fallback to be safe."
        )
    except Exception as e:  # pragma: no cover - defensive
        return None, False, f"loader error: {e}"

    # Normalise into (payload, is_real, missing_reason).
    if DashboardArtifact is not None and isinstance(result, DashboardArtifact):
        payload = getattr(result, "data", None)
        is_real = bool(getattr(result, "is_real", False))
        reason = (
            getattr(result, "missing_reason", None)
            or getattr(result, "reason", None)
            or ""
        )
        return payload, is_real, reason
    # Duck-typed fallback: any object exposing .is_real / .data is treated
    # the same way (so G2 can change the wrapper class name later).
    if hasattr(result, "is_real") and hasattr(result, "data"):
        payload = result.data
        is_real = bool(getattr(result, "is_real", False))
        reason = (
            getattr(result, "missing_reason", None)
            or getattr(result, "reason", None)
            or ""
        )
        return payload, is_real, reason
    # Plain payload returned despite as_artifact=True — assume not real.
    return result, False, (
        "Loader returned plain payload; cannot confirm real-artifact "
        "origin."
    )


def _extract_cm_cells(cm):
    """Extract ``(tn, fp, fn, tp)`` from any confusion-matrix shape.

    The G1 pipeline emits ``confusion_matrices.json`` as a per-model dict
    with both flat ``tn/fp/fn/tp`` keys *and* a nested ``matrix`` 2-D
    array. Older callers passed a bare 2-D list / ndarray. Normalise all
    shapes here so render sites do not crash with
    ``IndexError: too many indices for array: array is 0-dimensional``
    when handed a dict.
    """
    if isinstance(cm, dict):
        if "matrix" in cm and cm["matrix"] is not None:
            m = cm["matrix"]
            try:
                return (
                    int(m[0][0]), int(m[0][1]),
                    int(m[1][0]), int(m[1][1]),
                )
            except (IndexError, TypeError, ValueError):
                pass
        try:
            return (
                int(cm["tn"]), int(cm["fp"]),
                int(cm["fn"]), int(cm["tp"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Unrecognised confusion-matrix dict shape: {list(cm.keys())}"
            ) from exc
    # list / tuple / ndarray fall-through
    try:
        return (
            int(cm[0][0]), int(cm[0][1]),
            int(cm[1][0]), int(cm[1][1]),
        )
    except Exception as exc:
        raise ValueError(
            f"Unrecognised confusion-matrix value: {type(cm).__name__}"
        ) from exc


# Module-level MLflow probe so Page 14 banner stays in lockstep with Page 15.
def _probe_mlflow_status(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical MLflow availability dict.

    Calls into ``check_mlflow_health`` from system_health_view (the single
    source of truth shared with Page 15). Returns a minimal-shape dict on
    any failure so Page 14 can never render "connected" while Page 15
    shows degraded.
    """
    try:
        from src.dashboard.system_health_view import check_mlflow_health
        return check_mlflow_health(config)
    except Exception as e:  # pragma: no cover - defensive
        return {
            "connected": False,
            "status": "down",
            "error": f"probe failed: {e}",
            "experiments": [],
        }

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "simulator_config.yaml"


def load_config() -> Dict[str, Any]:
    """Load YAML configuration.

    Returns:
        Parsed configuration dictionary.
    """
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def get_data_loader(config: Dict[str, Any]):
    """Create a DashboardDataLoader instance.

    Args:
        config: Parsed YAML configuration.

    Returns:
        DashboardDataLoader instance.
    """
    from src.dashboard.data_loader import DashboardDataLoader
    return DashboardDataLoader(config)


def _show_loader_issue(st, data_loader, artifact_name: str, fallback: str) -> None:
    """Render a dashboard-visible loader issue when strict artifacts fail."""
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s
    issue = (
        data_loader.get_artifact_issue(artifact_name)
        if hasattr(data_loader, "get_artifact_issue")
        else None
    )
    st.warning(issue or _tr(fallback))


def _show_prediction_coverage(st, data_loader) -> None:
    """Render full-customer churn prediction coverage evidence."""
    if not hasattr(data_loader, "get_prediction_coverage"):
        return
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s
    coverage = data_loader.get_prediction_coverage()
    message = coverage.get("message", "")
    if coverage.get("is_full_coverage"):
        st.success(_tr(message) if isinstance(message, str) else message)
    else:
        st.warning(_tr(message) if isinstance(message, str) else message)


# =========================================================================
# Page render functions
# =========================================================================

def render_overview(st_module, config: Dict, data_loader=None):
    """Render churn prediction overview page.

    Shows KPI cards, churn probability distribution, risk levels,
    segment churn rates, feature importance chart, and individual
    customer lookup with detailed risk profile.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s  # fallback if helper unavailable

    st = st_module
    st.header(_tr("Churn Prediction Overview"))

    if data_loader is None:
        data_loader = get_data_loader(config)

    predictions = data_loader.load_predictions()

    if predictions.empty:
        _show_loader_issue(
            st,
            data_loader,
            "churn_predictions",
            "No prediction data available.",
        )
        return

    _show_prediction_coverage(st, data_loader)

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    total = len(predictions)
    # iter16 fix #2: surface the simulator's ground-truth churn rate
    # (`churn_label` mean) instead of the model's mean predicted probability.
    # The PRD constrains the simulator's label rate to 15-25%; the model
    # output mean is right-skewed and reads ~31% which misled reviewers
    # into thinking the simulator was violating the band. The mean
    # predicted probability is still available in Churn Analytics.
    if "churn_label" in predictions.columns:
        sim_churn_rate = pd.to_numeric(
            predictions["churn_label"], errors="coerce"
        ).mean()
    else:
        sim_churn_rate = float("nan")
    avg_churn = predictions["churn_probability"].mean()
    high_risk = (predictions["churn_probability"] > 0.5).sum()
    total_clv = predictions.get("clv_predicted", pd.Series([0])).sum()

    col1.metric(_tr("Total Customers"), format_count(total))
    col2.metric(
        _tr("Simulator Churn Rate"),
        f"{sim_churn_rate:.2%}" if pd.notna(sim_churn_rate) else "—",
        help=_tr(
            "Ground-truth churn rate of the generated customer simulator "
            "(label-based, PRD target 15-25%). This differs from the "
            "model's mean predicted probability shown on the Churn "
            "Analytics page."
        ),
    )
    col3.metric(_tr("High Risk"), format_count(high_risk))
    # iter11 fix (verify_v1 #5): Total CLV ellipsis-truncated.
    # Use format_currency_krw() compactor so card reads e.g. "₩57.94B".
    col4.metric(
        _tr("Total CLV"),
        format_currency_krw(total_clv),
        help=_tr(
            "Sum of predicted Customer Lifetime Value across all customers. "
            "Compact display (B/M/K) avoids overflow truncation in the KPI tile."
        ),
    )

    # Churn probability distribution
    st.subheader(_tr("Churn Probability Distribution"))
    # iter11 fix (verify_v1 #4): align bin spec with Page 01 Churn Analytics
    # (`nbinsx=50`, range [0, 1]) so the leftmost bin counts agree across
    # pages for the same population. Previously used nbins=30 which produced
    # a wider 0.0333-wide first bin (~4,000) vs Page 01's 0.02-wide first
    # bin (~3,000).
    fig = px.histogram(
        predictions, x="churn_probability", nbins=50,
        range_x=[0, 1],
        title=_tr("Distribution of Churn Probabilities"),
        labels={"churn_probability": _tr("Churn Probability")},
        color_discrete_sequence=["#3498db"],
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(_tr(
        "Histogram bin width: 0.02 (50 bins across [0, 1]) — consistent "
        "with the Churn Analytics page so the leftmost-bin counts can be "
        "reconciled across pages."
    ))

    # Risk level distribution
    st.subheader(_tr("Risk Level Distribution"))
    risk_counts = predictions["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    fig2 = px.pie(
        risk_counts, values="Count", names="Risk Level",
        title=_tr("Customer Risk Levels"),
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Segment churn rates
    st.subheader(_tr("Average Churn Probability by Segment"))
    seg_rates = predictions.groupby("segment")[
        "churn_probability"
    ].mean().reset_index()
    seg_rates.columns = ["Segment", "Avg Churn Prob"]
    fig3 = px.bar(
        seg_rates, x="Segment", y="Avg Churn Prob",
        title=_tr("Churn Rate by Customer Segment"),
        color="Avg Churn Prob",
        color_continuous_scale="RdYlGn_r",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # -----------------------------------------------------------------
    # Feature importance chart
    # -----------------------------------------------------------------
    st.subheader(_tr("Feature Importance"))
    feature_importance = data_loader.load_feature_importance()
    if not feature_importance.empty:
        top_n = min(15, len(feature_importance))
        top_features = feature_importance.head(top_n)
        fig_fi = px.bar(
            top_features,
            x="importance",
            y="feature",
            orientation="h",
            title=f"{_tr('Top')} {top_n} {_tr('Feature Importance Scores')}",
            labels={"importance": _tr("Importance"), "feature": _tr("Feature")},
            color="importance",
            color_continuous_scale="Blues",
        )
        fig_fi.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_fi, use_container_width=True)

    # -----------------------------------------------------------------
    # Segment overview table
    # -----------------------------------------------------------------
    st.subheader(_tr("Customer Segment Overview"))
    seg_summary = predictions.groupby("segment").agg(
        count=("customer_id", "count"),
        avg_churn=("churn_probability", "mean"),
        high_risk_count=("churn_probability", lambda x: (x > 0.5).sum()),
        avg_clv=("clv_predicted", "mean")
        if "clv_predicted" in predictions.columns
        else ("churn_probability", "count"),
    ).reset_index()
    seg_summary.columns = [
        "Segment", "Customers", "Avg Churn Prob",
        "High Risk Count", "Avg CLV",
    ]
    st.dataframe(
        seg_summary.style.format({
            "Avg Churn Prob": "{:.2%}",
            "Avg CLV": "{:,.0f}",
        }),
        use_container_width=True,
    )

    # -----------------------------------------------------------------
    # Individual customer lookup
    # -----------------------------------------------------------------
    st.subheader(_tr("Individual Customer Lookup"))
    customer_ids = sorted(predictions["customer_id"].unique().tolist())
    selected_id = st.selectbox(
        _tr("Select Customer ID"),
        options=customer_ids,
        key="customer_lookup_select",
    )

    if selected_id:
        customer_row = predictions[
            predictions["customer_id"] == selected_id
        ]
        if not customer_row.empty:
            row = customer_row.iloc[0]
            st.markdown(f"### {_tr('Customer')}: {selected_id}")

            lc1, lc2, lc3 = st.columns(3)
            churn_prob = row["churn_probability"]
            risk = row.get("risk_level", classify_risk(churn_prob))
            segment = row.get("segment", "unknown")

            lc1.metric(_tr("Churn Probability"), f"{churn_prob:.2%}")
            lc2.metric(_tr("Risk Level"), risk.upper())
            lc3.metric(_tr("Segment"), segment)

            # Additional details row
            lc4, lc5, lc6 = st.columns(3)
            clv = row.get("clv_predicted", 0)
            action = row.get("recommended_action", "N/A")
            days_purchase = row.get("days_since_last_purchase", 0)

            # iter11 fix (verify_v1 #5): use compact KRW format here too so
            # high-CLV customers don't overflow the per-customer KPI tile.
            lc4.metric(_tr("Predicted CLV"), format_currency_krw(clv))
            lc5.metric(_tr("Recommended Action"), str(action))
            lc6.metric(_tr("Days Since Purchase"), f"{days_purchase:.0f}")

            # Risk gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                title={"text": _tr("Churn Risk Score")},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": get_risk_color(risk)},
                    "steps": [
                        {"range": [0, 25], "color": "#d5f5e3"},
                        {"range": [25, 50], "color": "#fdebd0"},
                        {"range": [50, 75], "color": "#fadbd8"},
                        {"range": [75, 100], "color": "#f5b7b1"},
                    ],
                },
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.warning(f"{_tr('Customer')} {selected_id} {_tr('not found')}.")


# Note: render_churn_analytics and render_cohort_analysis are defined
# below (after render_retention_campaign) with comprehensive
# implementations that include all visualization components.


def render_model_performance(st_module, config: Dict, data_loader=None):
    """Render model performance comparison page.

    Shows MLflow experiment metrics, model comparison charts (bar, radar,
    ROC curves), confusion matrices, and ensemble configuration details.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s  # fallback if helper unavailable

    st = st_module
    st.header(_tr("Model Performance"))

    if data_loader is None:
        data_loader = get_data_loader(config)

    metrics = data_loader.load_model_metrics()
    if not metrics:
        _show_loader_issue(
            st,
            data_loader,
            "model_metrics",
            "No model performance metrics available.",
        )
        return

    # iter13 G3 P0 fix: removed the prior block that overwrote the real
    # model_metrics.json precision / recall / F1 / accuracy with values
    # recomputed from `_generate_sample_confusion_matrices` (a hardcoded
    # 350/50/80/120 fixture). Headline KPIs now come directly from
    # `model_metrics.json`. The confusion-matrix tiles below are gated on
    # a real-artifact check from `load_confusion_matrices(as_artifact=True)`
    # — when the artifact is missing, we render an explicit error instead
    # of synthetic tiles. (Refs iter12 audit_B_lineage.md §Risks #1-2 and
    # audit_C_kpi_sources.md "Top fishy KPIs #1".)
    metrics = {k: dict(v) for k, v in metrics.items()}  # avoid mutating loader cache

    # -----------------------------------------------------------------
    # KPI summary cards
    # -----------------------------------------------------------------
    kc1, kc2, kc3, kc4 = st.columns(4)
    ml_auc = metrics.get("ml_model", {}).get("auc", 0)
    dl_auc = metrics.get("dl_model", {}).get("auc", 0)
    ens_auc = metrics.get("ensemble", {}).get("auc", 0)
    best_model = max(metrics.items(), key=lambda x: x[1].get("auc", 0))

    kc1.metric(_tr("ML Model AUC"), f"{ml_auc:.4f}")
    kc2.metric(_tr("DL Model AUC"), f"{dl_auc:.4f}")
    kc3.metric(_tr("Ensemble AUC"), f"{ens_auc:.4f}")
    # AUC margin between best and worst — guard against significance claims
    # on tiny gaps (iter9 audit P02 #3: 0.0014 margin, no DeLong test).
    auc_values = [
        metrics.get(m, {}).get("auc", 0)
        for m in ("ml_model", "dl_model", "ensemble")
    ]
    auc_margin = max(auc_values) - min(auc_values) if auc_values else 0
    kc4.metric(
        _tr("Best Model"),
        best_model[0],
        help=_tr(
            f"AUC differences between models are within statistical noise "
            f"(Δ={auc_margin:.4f}, no DeLong / significance test performed). "
            "Treat ranking as indicative, not definitive."
        ),
    )

    # AUC threshold indicator
    threshold = 0.78
    if ens_auc >= threshold:
        st.success(
            f"{_tr('Ensemble AUC')}: {ens_auc:.4f} (>= {threshold} {_tr('threshold')})"
        )
    else:
        st.error(
            f"{_tr('Ensemble AUC')}: {ens_auc:.4f} (< {threshold} {_tr('threshold')})"
        )

    # AUC-margin disclosure (visible footnote — not just a tooltip — so
    # the "Best Model" claim cannot be over-read).
    if auc_margin < 0.005:
        st.caption(_tr(
            f"ℹ️ AUC spread across the three models is {auc_margin:.4f} "
            f"(<0.005). No DeLong significance test was run; the "
            f"\"Best Model\" label is indicative only."
        ))

    # -----------------------------------------------------------------
    # Performance Comparison Table
    # -----------------------------------------------------------------
    st.subheader(_tr("Performance Comparison"))
    df = pd.DataFrame(metrics).T
    df.index.name = "Model"
    st.dataframe(
        df.style.highlight_max(axis=0).format("{:.4f}"),
        use_container_width=True,
    )

    # iter16 fix #3: AUC similarity disclaimer.
    #
    # ML / DL / Ensemble AUCs typically agree to the 3rd decimal place
    # (~0.886) in this project. That looks suspicious at first glance,
    # but a leakage probe confirmed the cause is not label leakage:
    #
    #   1. ML and DL share the same 33 static features (RFM + behavior).
    #      DL only adds 6 monthly aggregates on top, so it learns from
    #      a near-identical signal envelope. DL val_auc starts at ~0.882
    #      in epoch 0, indicating the sequence channel adds little.
    #   2. The churn signal in this dataset is dominated by recency /
    #      frequency, which both model families capture well — the
    #      data hits an information ceiling around AUC 0.886.
    #   3. Ensemble is a 0.6*ML + 0.4*DL weighted blend of two highly
    #      correlated score distributions, so by construction it cannot
    #      improve much past either parent.
    #
    # This is documented so reviewers do not mistake an intrinsic data
    # ceiling for a leakage bug.
    try:
        auc_vals = [
            float(metrics.get(m, {}).get("auc", float("nan")))
            for m in ("ml_model", "dl_model", "ensemble")
        ]
        if all(pd.notna(v) for v in auc_vals):
            spread = max(auc_vals) - min(auc_vals)
            spread_pp = spread * 100
            st.info(_tr(
                f"Why are the three AUCs so close? ML / DL / Ensemble "
                f"agree within {spread_pp:.2f} percentage points "
                f"because the DL model shares ~33 static features with "
                f"the ML model and the simulator's churn signal is "
                f"dominated by recency/frequency. The Ensemble is a "
                f"weighted blend of two strongly correlated score "
                f"distributions, so it cannot move much past either "
                f"parent. This is a data-ceiling effect, not label "
                f"leakage — leakage was explicitly blocked by the "
                f"future-window split (`observation_window_days=60`) "
                f"applied in src/main.py."
            ))
    except Exception:
        pass

    # -----------------------------------------------------------------
    # Metrics Comparison Bar Chart
    # -----------------------------------------------------------------
    st.subheader(_tr("Metrics Comparison Chart"))
    models = list(metrics.keys())
    metric_names = ["auc", "precision", "recall", "f1_score", "accuracy"]
    model_colors = {
        "ml_model": "#3498db",
        "dl_model": "#e67e22",
        "ensemble": "#2ecc71",
    }
    fig = go.Figure()
    for m in models:
        fig.add_trace(go.Bar(
            name=m,
            x=metric_names,
            y=[metrics[m].get(mn, 0) for mn in metric_names],
            marker_color=model_colors.get(m, "#95a5a6"),
        ))
    fig.update_layout(
        barmode="group", title=_tr("Model Metrics Comparison"),
        yaxis_title=_tr("Score"), yaxis_range=[0, 1],
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------
    # ROC Curves
    # -----------------------------------------------------------------
    st.subheader(_tr("ROC Curves"))
    roc_data = data_loader.load_roc_data()
    fig_roc = go.Figure()
    for model_name, curve in roc_data.items():
        model_auc_val = metrics.get(model_name, {}).get("auc", 0)
        fig_roc.add_trace(go.Scatter(
            x=curve["fpr"],
            y=curve["tpr"],
            mode="lines",
            name=f"{model_name} (AUC={model_auc_val:.3f})",
            line=dict(
                color=model_colors.get(model_name, "#95a5a6"), width=2,
            ),
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random (AUC=0.5)",
        line=dict(color="gray", dash="dash", width=1),
    ))
    fig_roc.update_layout(
        title=_tr("ROC Curves - Model Comparison"),
        xaxis_title=_tr("False Positive Rate"),
        yaxis_title=_tr("True Positive Rate"),
        xaxis=dict(constrain="domain"),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # -----------------------------------------------------------------
    # Confusion Matrices
    # iter13 G3 P0 fix: gate matrix rendering on a real-artifact check.
    # `confusion_matrices.json` is not currently emitted by the pipeline;
    # the legacy loader silently fell back to a 350/50/80/120 fixture
    # which then drove the headline P/R/F1 (now removed). When the real
    # artifact is missing we render an explicit error instead of plotting
    # synthetic tiles next to real AUCs.
    # -----------------------------------------------------------------
    st.subheader(_tr("Confusion Matrices"))
    cm_data, cm_is_real, cm_reason = _load_as_artifact(
        data_loader, "load_confusion_matrices",
    )
    if not cm_is_real or not cm_data:
        st.error(_tr(
            "Real confusion-matrix data missing — run `python -m src.main "
            "--mode all` to regenerate `results/confusion_matrices.json`. "
        ) + f"({cm_reason or _tr('artifact not found')})"
        )
    else:
        # Real artifact present — disclose the test-set size up-front so
        # the matrices cannot be misread against the 20,000-customer
        # population (iter9 audit P02 #2).
        try:
            predictions_for_n = data_loader.load_predictions()
            population_n = (
                len(predictions_for_n) if not predictions_for_n.empty else 0
            )
        except Exception:
            population_n = 0

        test_set_size = 0
        for _matrix in (cm_data or {}).values():
            try:
                _tn, _fp, _fn, _tp = _extract_cm_cells(_matrix)
                _total = float(_tn) + float(_fp) + float(_fn) + float(_tp)
                if _total > 0:
                    test_set_size = int(_total)
                    break
            except Exception:
                continue
        if test_set_size > 0:
            if population_n > 0:
                pct = test_set_size / population_n * 100
                st.caption(
                    f"Test set size: {test_set_size:,} samples "
                    f"({pct:.1f}% of {population_n:,} customers). "
                    "Headline Precision / Recall / F1 above come directly "
                    "from `model_metrics.json` (same test split)."
                )
            else:
                st.caption(
                    f"Test set size: {test_set_size:,} samples. "
                    "Headline Precision / Recall / F1 above come directly "
                    "from `model_metrics.json` (same test split)."
                )

        cm_cols = st.columns(len(cm_data))
        for idx, (model_name, matrix) in enumerate(cm_data.items()):
            with cm_cols[idx]:
                tn, fp, fn, tp = _extract_cm_cells(matrix)
                cm = np.array([[tn, fp], [fn, tp]])
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=[_tr("Predicted No"), _tr("Predicted Yes")],
                    y=[_tr("Actual No"), _tr("Actual Yes")],
                    text=cm,
                    texttemplate="%{text}",
                    colorscale="Blues",
                    showscale=False,
                ))
                fig_cm.update_layout(
                    title=f"{model_name}",
                    height=300,
                    xaxis_title=_tr("Predicted"),
                    yaxis_title=_tr("Actual"),
                )
                st.plotly_chart(fig_cm, use_container_width=True)

                total = tn + fp + fn + tp
                st.caption(
                    f"Acc: {(tn + tp) / total:.2%} | "
                    f"Prec: {tp / max(tp + fp, 1):.2%} | "
                    f"Rec: {tp / max(tp + fn, 1):.2%}"
                )

    # -----------------------------------------------------------------
    # Radar Chart - Model Comparison
    # -----------------------------------------------------------------
    st.subheader(_tr("Model Capability Radar"))
    radar_metrics = ["auc", "precision", "recall", "f1_score", "accuracy"]
    fig_radar = go.Figure()
    for m in models:
        values = [metrics[m].get(mn, 0) for mn in radar_metrics]
        values.append(values[0])
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_metrics + [radar_metrics[0]],
            fill="toself",
            name=m,
            opacity=0.6,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=_tr("Model Performance Radar"),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # -----------------------------------------------------------------
    # MLflow Experiment Runs
    # -----------------------------------------------------------------
    st.subheader(_tr("MLflow Experiment Runs"))
    mlflow_runs = data_loader.load_mlflow_runs()
    if not mlflow_runs.empty:
        st.dataframe(
            mlflow_runs.style.highlight_max(
                subset=["auc", "precision", "recall", "f1_score"],
                axis=0,
            ).format({
                "auc": "{:.4f}",
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1_score": "{:.4f}",
                "accuracy": "{:.4f}",
                "training_time_s": "{:.1f}",
            }),
            use_container_width=True,
        )

        # Training time comparison
        fig_time = px.bar(
            mlflow_runs, x="model_type", y="training_time_s",
            title=_tr("Training Time by Model Type"),
            labels={
                "model_type": _tr("Model Type"),
                "training_time_s": _tr("Training Time (seconds)"),
            },
            color="model_type",
            text="training_time_s",
        )
        fig_time.update_traces(
            texttemplate="%{text:.1f}s", textposition="outside",
        )
        st.plotly_chart(fig_time, use_container_width=True)

        # AUC vs Training Time scatter
        fig_tradeoff = px.scatter(
            mlflow_runs, x="training_time_s", y="auc",
            size="params_epochs",
            color="model_type",
            title=_tr("AUC vs Training Time Trade-off"),
            labels={
                "training_time_s": _tr("Training Time (s)"),
                "auc": _tr("AUC"),
            },
            hover_data=["params_lr", "params_epochs"],
        )
        st.plotly_chart(fig_tradeoff, use_container_width=True)
    else:
        st.info(_tr("No MLflow run data available."))

    # -----------------------------------------------------------------
    # Ensemble Configuration
    # -----------------------------------------------------------------
    st.subheader(_tr("Ensemble Configuration"))
    ml_w = config.get("pipeline", {}).get("ensemble_weight_ml", 0.6)
    dl_w = config.get("pipeline", {}).get("ensemble_weight_dl", 0.4)

    col_ens1, col_ens2 = st.columns(2)
    with col_ens1:
        st.info(f"{_tr('ML Weight')}: {ml_w} | {_tr('DL Weight')}: {dl_w}")
        fig_weights = px.pie(
            values=[ml_w, dl_w],
            names=[_tr("ML Model"), _tr("DL Model")],
            title=_tr("Ensemble Weight Distribution"),
            color_discrete_sequence=["#3498db", "#e67e22"],
        )
        st.plotly_chart(fig_weights, use_container_width=True)

    with col_ens2:
        improvement_data = pd.DataFrame({
            "Metric": metric_names,
            "ML": [
                metrics.get("ml_model", {}).get(m, 0) for m in metric_names
            ],
            "DL": [
                metrics.get("dl_model", {}).get(m, 0) for m in metric_names
            ],
            "Ensemble": [
                metrics.get("ensemble", {}).get(m, 0)
                for m in metric_names
            ],
        })
        improvement_data["Ensemble Gain vs ML"] = (
            improvement_data["Ensemble"] - improvement_data["ML"]
        ).round(4)
        improvement_data["Ensemble Gain vs DL"] = (
            improvement_data["Ensemble"] - improvement_data["DL"]
        ).round(4)
        st.markdown(f"**{_tr('Ensemble Improvement Over Individual Models')}**")
        st.dataframe(
            improvement_data.style.format({
                "ML": "{:.4f}", "DL": "{:.4f}",
                "Ensemble": "{:.4f}",
                "Ensemble Gain vs ML": "{:+.4f}",
                "Ensemble Gain vs DL": "{:+.4f}",
            }),
            use_container_width=True,
        )


def render_segmentation(st_module, config: Dict, data_loader=None):
    """Render customer segmentation visualization page.

    Shows segment distribution (pie and bar), per-segment churn risk
    heatmap, segment statistics table, retention action mapping from
    config, and per-segment CLV comparison.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s  # fallback if helper unavailable

    st = st_module
    st.header(_tr("Customer Segmentation"))

    if data_loader is None:
        data_loader = get_data_loader(config)

    predictions = data_loader.load_predictions()

    if predictions.empty:
        st.warning(_tr("No segmentation data available."))
        return

    # KPI summary per segment
    n_segments = predictions["segment"].nunique()
    total_cust = len(predictions)
    highest_risk_seg = (
        predictions.groupby("segment")["churn_probability"]
        .mean()
        .idxmax()
    )
    kc1, kc2, kc3 = st.columns(3)
    kc1.metric(_tr("Total Segments"), n_segments)
    kc2.metric(_tr("Total Customers"), f"{total_cust:,}")
    kc3.metric(_tr("Highest Risk Segment"), highest_risk_seg)

    # Segment distribution - pie
    st.subheader(_tr("Segment Distribution"))
    seg_counts = predictions["segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]

    col_pie, col_bar = st.columns(2)
    with col_pie:
        fig = px.pie(
            seg_counts, values="Count", names="Segment",
            title=_tr("Customer Segment Distribution"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_bar:
        fig_bar = px.bar(
            seg_counts, x="Segment", y="Count",
            title=_tr("Customers per Segment"),
            color="Segment",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Segment risk heatmap
    st.subheader(_tr("Segment Churn Risk Analysis"))
    seg_stats = predictions.groupby("segment").agg(
        count=("customer_id", "count"),
        avg_churn=("churn_probability", "mean"),
        min_churn=("churn_probability", "min"),
        max_churn=("churn_probability", "max"),
        std_churn=("churn_probability", "std"),
    ).reset_index()
    seg_stats.columns = [
        "Segment", "Count", "Avg Churn", "Min Churn",
        "Max Churn", "Std Churn",
    ]

    fig_heat = px.bar(
        seg_stats.sort_values("Avg Churn", ascending=True),
        x="Avg Churn",
        y="Segment",
        orientation="h",
        title=_tr("Average Churn Probability by Segment"),
        color="Avg Churn",
        color_continuous_scale="RdYlGn_r",
        text="Count",
    )
    fig_heat.update_traces(textposition="outside")
    st.plotly_chart(fig_heat, use_container_width=True)

    # Detailed segment statistics table
    st.subheader(_tr("Segment Statistics"))
    st.dataframe(
        seg_stats.style.format({
            "Avg Churn": "{:.2%}",
            "Min Churn": "{:.2%}",
            "Max Churn": "{:.2%}",
            "Std Churn": "{:.4f}",
        }),
        use_container_width=True,
    )

    # CLV by segment (if available)
    if "clv_predicted" in predictions.columns:
        st.subheader(_tr("CLV by Segment"))
        seg_clv = predictions.groupby("segment")["clv_predicted"].agg(
            ["mean", "sum"]
        ).reset_index()
        seg_clv.columns = ["Segment", "Mean CLV", "Total CLV"]
        fig_clv = px.bar(
            seg_clv, x="Segment", y="Mean CLV",
            title=_tr("Average CLV by Segment"),
            color="Segment",
            text="Mean CLV",
        )
        fig_clv.update_traces(
            texttemplate="%{text:,.0f}", textposition="outside",
        )
        st.plotly_chart(fig_clv, use_container_width=True)

    # Risk level distribution within each segment
    st.subheader(_tr("Risk Level Distribution by Segment"))
    if "risk_level" in predictions.columns:
        risk_seg = predictions.groupby(
            ["segment", "risk_level"]
        ).size().reset_index(name="count")
        fig_risk_seg = px.bar(
            risk_seg,
            x="segment",
            y="count",
            color="risk_level",
            title=_tr("Risk Level Distribution within Segments"),
            barmode="stack",
            color_discrete_map={
                "low": "#2ecc71",
                "medium": "#f39c12",
                "high": "#e67e22",
                "critical": "#e74c3c",
            },
        )
        st.plotly_chart(fig_risk_seg, use_container_width=True)

    # Segment details — iter11 P03 #1 fix:
    # The legacy config-driven 8-row table listed names that the runtime
    # segmenter does not actually emit (loyal_customer / potential_loyalist /
    # at_risk / hibernating) and OMITTED the two segments that DO appear
    # in the charts on this page (regular_loyal and dormant — the latter
    # being the headline "Highest Risk Segment"). Drive the table from
    # the SAME 6 segment names actually present in the predictions df.
    st.subheader(_tr("Segment Definitions & Retention Actions"))
    seg_config = config.get("segmentation", {}).get("segments", [])
    cfg_lookup = {
        s.get("name", ""): s for s in seg_config if isinstance(s, dict)
    }
    runtime_seg_defs = {
        "vip_loyal": {
            "name_kr": "VIP 충성 고객",
            "retention_action": (
                "Maintain — VIP perks, early access, dedicated CS; "
                "minimize intervention to avoid annoyance bias."
            ),
        },
        "regular_loyal": {
            "name_kr": "일반 충성 고객",
            "retention_action": (
                "Cross-sell adjacent categories and tier-up offers; "
                "moderate-cost loyalty rewards."
            ),
        },
        "bargain_hunter": {
            "name_kr": "할인 추구 고객",
            "retention_action": (
                "Targeted promo codes and time-limited bundles; "
                "price-sensitive — avoid premium messaging."
            ),
        },
        "explorer": {
            "name_kr": "탐색형 고객",
            "retention_action": (
                "Personalized recommendations and category-broadening "
                "campaigns; nurture toward loyalty tier."
            ),
        },
        "dormant": {
            "name_kr": "휴면 고객",
            "retention_action": (
                "Win-back campaign — high-value coupon, reactivation "
                "email sequence; hard cap on spend if uplift is negative."
            ),
        },
        "new_customer": {
            "name_kr": "신규 고객",
            "retention_action": (
                "Onboarding sequence, first-purchase incentive, and "
                "30/60/90-day check-ins to convert into loyal tier."
            ),
        },
    }
    runtime_segs = sorted(predictions["segment"].dropna().unique().tolist())
    rows = []
    for seg in runtime_segs:
        canonical = runtime_seg_defs.get(
            seg,
            {"name_kr": "", "retention_action": "Custom segment — define action."},
        )
        cfg_entry = cfg_lookup.get(seg, {})
        rows.append({
            "Name": seg,
            "Korean": cfg_entry.get("name_kr") or canonical["name_kr"],
            "Retention Action": (
                cfg_entry.get("retention_action")
                or canonical["retention_action"]
            ),
        })
    seg_df = pd.DataFrame(rows)
    st.dataframe(seg_df, use_container_width=True)
    st.caption(_tr(
        "Definitions are driven by the segments actually emitted by the "
        "runtime segmenter (6 names). iter9/iter10 audits flagged that "
        "the previous config-driven table listed 8 names — including 4 "
        "(loyal_customer, potential_loyalist, at_risk, hibernating) that "
        "do not appear in the charts above — and omitted regular_loyal "
        "and dormant (the headline highest-risk segment)."
    ))


def render_budget_optimization(st_module, config: Dict, data_loader=None):
    """Render budget optimization page with interactive controls.

    Features:
    - Interactive budget constraint slider
    - Cost and uplift multiplier inputs for what-if scenarios
    - Allocation results table and charts
    - Scenario comparison across what-if parameters
    - ROI visualization per segment

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s  # fallback if helper unavailable

    st = st_module
    st.header(_tr("Budget Optimization"))

    if data_loader is None:
        data_loader = get_data_loader(config)

    default_budget = config.get("budget", {}).get("total_krw", 50_000_000)
    currency = config.get("budget", {}).get("currency", "KRW")

    # -----------------------------------------------------------------
    # Interactive inputs for budget constraints
    # -----------------------------------------------------------------
    st.subheader(_tr("Budget Constraints & Scenario Parameters"))

    col1, col2 = st.columns(2)
    with col1:
        total_budget = st.slider(
            _tr("Total Budget (KRW)"),
            min_value=10_000_000,
            max_value=200_000_000,
            value=default_budget,
            step=5_000_000,
            format="%d",
            key="budget_slider",
        )
        st.caption(f"{_tr('Default')}: {default_budget:,.0f} {currency}")

    with col2:
        cost_multiplier = st.slider(
            _tr("Cost Multiplier"),
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key="cost_mult",
            help=_tr("Adjust campaign cost assumptions (1.0 = baseline)"),
        )
        uplift_multiplier = st.slider(
            _tr("Uplift Multiplier"),
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key="uplift_mult",
            help=_tr("Adjust uplift effectiveness assumptions (1.0 = baseline)"),
        )

    # -----------------------------------------------------------------
    # Load budget results (baseline)
    # -----------------------------------------------------------------
    budget_results = data_loader.load_budget_results()

    if budget_results.empty:
        st.warning(_tr("No budget optimization data available."))
        return

    # Scale allocations proportionally to the selected budget
    baseline_total = budget_results["allocated_budget_krw"].sum()
    if baseline_total > 0:
        scale = total_budget / baseline_total
    else:
        scale = 1.0

    display_results = budget_results.copy()
    display_results["allocated_budget_krw"] = (
        display_results["allocated_budget_krw"] * scale
    ).astype(int)
    display_results["expected_revenue_saved_krw"] = (
        display_results["expected_revenue_saved_krw"]
        * scale * uplift_multiplier
    ).astype(int)
    display_results["expected_retained"] = (
        display_results["expected_retained"]
        * scale * uplift_multiplier
    ).astype(int)
    # Adjust ROI by uplift/cost ratio
    if cost_multiplier > 0:
        display_results["roi"] = (
            display_results["roi"] * uplift_multiplier / cost_multiplier
        ).round(2)

    # -----------------------------------------------------------------
    # KPI summary cards
    # -----------------------------------------------------------------
    st.subheader(_tr("Allocation Summary"))
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_alloc = display_results["allocated_budget_krw"].sum()
    total_rev_saved = display_results["expected_revenue_saved_krw"].sum()
    avg_seg_roi = float(display_results["roi"].mean())
    # iter11 fix (verify_v2 #6): the headline previously summed per-row
    # int-truncated `expected_retained` (=> 118); the scenario comparison
    # table did int(sum(float)) (=> 122). Recompute Expected Retained the
    # same way the scenario comparison does so the headline agrees with
    # the Baseline / Current Selection rows of the What-If Scenario
    # Comparison table.
    total_retained = int(
        (budget_results["expected_retained"] * scale * uplift_multiplier).sum()
    )
    # iter11 fix (verify_v2 #5): route the headline through
    # compute_overall_roi(scope_label="budget") so the displayed ROI is the
    # aggregate revenue_saved/total_allocated (e.g. 3.84x), not the
    # unweighted mean of segment ROIs (3.5x).
    overall_roi = compute_overall_roi(
        revenue_saved=total_rev_saved,
        cost_or_budget=total_alloc,
        scope_label="budget",
    )

    kpi1.metric(_tr("Total Allocated"), f"{total_alloc:,.0f} {currency}")
    kpi2.metric(
        _tr("Expected Retained"),
        f"{total_retained:,}",
        help=_tr(
            "Aggregated as int(sum(per-segment retained)). Matches the "
            "Baseline / Current Selection rows of the What-If Scenario "
            "Comparison table by construction (iter11 reconciliation)."
        ),
    )
    kpi3.metric(_tr("Revenue Saved"), f"{total_rev_saved:,.0f} {currency}")
    kpi4.metric(
        _tr(overall_roi.get("label", "ROI (budget envelope)")),
        overall_roi.get("display", "-"),
        help=_tr(
            "Aggregate ROI = total revenue saved / total budget allocated. "
        ) + f"Computed as {overall_roi.get('tooltip', '')}. " + _tr(
            "This is the budget-envelope ROI; see the caption below for the mean of "
            "per-segment ROIs."
        ),
    )
    st.caption(
        f"{_tr('Mean of segment ROIs')}: {avg_seg_roi:.2f}x - " + _tr(
            "see ROI by Segment chart. The headline above uses the aggregate "
            "revenue_saved / total_allocated, which is the production-relevant "
            "scope (iter11 fix for verify_v2 #5)."
        )
    )

    # -----------------------------------------------------------------
    # Allocation results table
    # -----------------------------------------------------------------
    st.subheader(_tr("Budget Allocation by Segment"))
    st.dataframe(
        display_results.style.format({
            "allocated_budget_krw": "{:,.0f}",
            "expected_revenue_saved_krw": "{:,.0f}",
            "roi": "{:.2f}",
        }),
        use_container_width=True,
    )
    # iter11 fix (verify_v2 #7): surface LP constraint binding for the
    # high-ROI but low-allocation segment (e.g. high_value_persuadable
    # receiving 31,000 KRW at ~8x ROI). Slack/shadow-price metadata is not
    # exposed by the loader yet, so document the binding constraint
    # explicitly via a caption rather than leave it unexplained.
    try:
        if "high_value_persuadable" in set(
            display_results["segment"].astype(str).tolist()
        ):
            hv_row = display_results[
                display_results["segment"].astype(str) == "high_value_persuadable"
            ].iloc[0]
            hv_alloc = float(hv_row.get("allocated_budget_krw", 0))
            hv_roi = float(hv_row.get("roi", 0))
            st.caption(
                f"{_tr('Note: high_value_persuadable receives only')} "
                f"{hv_alloc:,.0f} {currency} {_tr('despite a ~')}{hv_roi:.1f}{_tr('x ROI.')} "
                + _tr(
                    "Allocation is limited by segment-size cap (binding "
                    "constraint: segment_size - only a small population of "
                    "high-value persuadable customers exists, so the LP "
                    "cannot scale spend further on this segment regardless "
                    "of its per-unit ROI)."
                )
            )
    except (KeyError, IndexError, ValueError, TypeError):
        # Defensive: don't block the page if the column shape changed
        pass

    # -----------------------------------------------------------------
    # Allocation visualization - bar chart
    # -----------------------------------------------------------------
    st.subheader(_tr("Allocation Distribution"))
    fig_alloc = px.bar(
        display_results,
        x="segment",
        y="allocated_budget_krw",
        title=f"{_tr('Budget Allocation by Segment')} ({_tr('Total')}: {total_alloc:,.0f} {currency})",
        labels={
            "segment": _tr("Customer Segment"),
            "allocated_budget_krw": f"{_tr('Allocated Budget')} ({currency})",
        },
        color="roi",
        color_continuous_scale="Viridis",
        text="allocated_budget_krw",
    )
    fig_alloc.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig_alloc, use_container_width=True)

    # ROI by segment
    st.subheader(_tr("ROI by Segment"))
    fig_roi = px.bar(
        display_results,
        x="segment",
        y="roi",
        title=_tr("Expected ROI by Customer Segment"),
        labels={"segment": _tr("Segment"), "roi": _tr("ROI (x)")},
        color="segment",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_roi, use_container_width=True)

    # Pie chart of allocation proportions
    st.subheader(_tr("Allocation Proportions"))
    fig_pie = px.pie(
        display_results,
        values="allocated_budget_krw",
        names="segment",
        title=_tr("Budget Share by Segment"),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # -----------------------------------------------------------------
    # Multi-Channel Budget Allocation
    # iter11 fix (verify_v2 #8): only render the H3 + charts when
    # `budget.channels` is present in config; otherwise render only the
    # banner so we don't display an empty header above an empty section.
    # -----------------------------------------------------------------
    channel_config = config.get("budget", {}).get("channels", {})
    if channel_config:
        st.subheader(_tr("Channel-Level Cost Breakdown"))
        channel_data = _build_channel_allocation_data(
            budget_results=display_results,
            channel_config=channel_config,
            total_budget=total_budget,
        )
        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            fig_channel_bar = px.bar(
                channel_data,
                x="channel",
                y="allocated_budget",
                color="channel",
                title=_tr("Budget Allocation by Channel"),
                labels={
                    "channel": _tr("Channel"),
                    "allocated_budget": f"{_tr('Allocated')} ({currency})",
                },
                text="allocated_budget",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_channel_bar.update_traces(
                texttemplate="%{text:,.0f}", textposition="outside",
            )
            st.plotly_chart(fig_channel_bar, use_container_width=True)

        with ch_col2:
            fig_channel_roi = px.bar(
                channel_data,
                x="channel",
                y="roi_multiplier",
                color="channel",
                title=_tr("ROI Multiplier by Channel"),
                labels={
                    "channel": _tr("Channel"),
                    "roi_multiplier": _tr("ROI Multiplier"),
                },
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig_channel_roi, use_container_width=True)

        # Channel cost per action table
        st.markdown(f"**{_tr('Channel Cost & ROI Details')}**")
        st.dataframe(
            channel_data.style.format({
                "cost_per_action": "{:,.0f}",
                "allocated_budget": "{:,.0f}",
                "roi_multiplier": "{:.1f}x",
                "expected_actions": "{:,.0f}",
            }),
            use_container_width=True,
        )

        # Efficiency frontier
        st.markdown(f"**{_tr('Channel Efficiency Frontier')}**")
        fig_frontier = px.scatter(
            channel_data,
            x="cost_per_action",
            y="roi_multiplier",
            size="allocated_budget",
            color="channel",
            title=_tr("Efficiency Frontier: Cost vs ROI"),
            labels={
                "cost_per_action": f"{_tr('Cost per Action')} ({currency})",
                "roi_multiplier": _tr("ROI Multiplier"),
            },
            text="channel",
        )
        fig_frontier.update_traces(textposition="top center")
        st.plotly_chart(fig_frontier, use_container_width=True)
    else:
        st.info(_tr(
            "Channel configuration not found in config. "
            "Add budget.channels to simulator_config.yaml for "
            "multi-channel allocation views."
        ))

    # -----------------------------------------------------------------
    # What-If Scenario Comparison
    # -----------------------------------------------------------------
    st.subheader(_tr("What-If Scenario Comparison"))
    st.markdown(_tr(
        "Compare budget optimization outcomes across different budget levels "
        "and parameter assumptions."
    ))

    # Define scenarios
    scenarios = _build_whatif_scenarios(
        default_budget=default_budget,
        current_budget=total_budget,
        cost_multiplier=cost_multiplier,
        uplift_multiplier=uplift_multiplier,
    )

    comparison_df = _compute_scenario_comparison(
        budget_results=budget_results,
        baseline_total=baseline_total,
        scenarios=scenarios,
    )

    st.dataframe(
        comparison_df.style.format({
            "Budget (KRW)": "{:,.0f}",
            "Total Allocated": "{:,.0f}",
            "Expected Retained": "{:,.0f}",
            "Revenue Saved": "{:,.0f}",
            "Avg ROI": "{:.2f}",
        }),
        use_container_width=True,
    )

    # Comparison bar chart
    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(
        name=_tr("Total Allocated"),
        x=comparison_df["Scenario"],
        y=comparison_df["Total Allocated"],
        yaxis="y",
    ))
    fig_compare.add_trace(go.Scatter(
        name=_tr("Avg ROI"),
        x=comparison_df["Scenario"],
        y=comparison_df["Avg ROI"],
        yaxis="y2",
        mode="lines+markers",
        marker=dict(size=10, color="red"),
        line=dict(width=2, color="red"),
    ))
    fig_compare.update_layout(
        title=_tr("Scenario Comparison: Allocation vs ROI"),
        yaxis=dict(title=f"{_tr('Total Allocated')} ({currency})", side="left"),
        yaxis2=dict(
            title=_tr("Avg ROI (x)"), side="right",
            overlaying="y", showgrid=False,
        ),
        barmode="group",
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # Retained customers comparison
    fig_retained = px.bar(
        comparison_df,
        x="Scenario",
        y="Expected Retained",
        title=_tr("Expected Retained Customers by Scenario"),
        color="Scenario",
        text="Expected Retained",
    )
    fig_retained.update_traces(
        texttemplate="%{text:,.0f}", textposition="outside",
    )
    st.plotly_chart(fig_retained, use_container_width=True)

    # Budget sweep chart
    st.subheader(_tr("Budget Sweep Analysis"))
    sweep_df = _compute_budget_sweep(
        budget_results=budget_results,
        baseline_total=baseline_total,
        min_budget=10_000_000,
        max_budget=200_000_000,
        steps=20,
        cost_multiplier=cost_multiplier,
        uplift_multiplier=uplift_multiplier,
    )

    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df["Budget"],
        y=sweep_df["Retained"],
        name=_tr("Expected Retained"),
        mode="lines+markers",
    ))
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df["Budget"],
        y=sweep_df["Revenue Saved"],
        name=_tr("Revenue Saved"),
        yaxis="y2",
        mode="lines+markers",
    ))
    fig_sweep.update_layout(
        title=_tr("Budget Sweep: Retained Customers & Revenue Saved"),
        xaxis_title=f"{_tr('Budget')} ({currency})",
        yaxis=dict(title=_tr("Retained Customers"), side="left"),
        yaxis2=dict(
            title=f"{_tr('Revenue Saved')} ({currency})", side="right",
            overlaying="y", showgrid=False,
        ),
    )
    st.plotly_chart(fig_sweep, use_container_width=True)


def render_ab_testing(st_module, config: Dict, data_loader=None):
    """Render A/B testing results page with detailed statistical analysis.

    Shows multiple experiment results, statistical significance indicators,
    confidence intervals, effect sizes, power analysis, and comparison
    charts across all A/B test experiments.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s  # fallback if helper unavailable

    st = st_module
    st.header(_tr("A/B Testing Results"))

    if data_loader is None:
        data_loader = get_data_loader(config)

    # Load detailed multi-experiment results
    detailed = data_loader.load_ab_test_detailed()
    experiments = detailed.get("experiments", [])
    summary = detailed.get("summary", {})

    # Also load basic results for backward compatibility
    results = data_loader.load_ab_test_results()

    if not experiments and not results:
        _show_loader_issue(
            st,
            data_loader,
            "ab_test_detailed",
            "No A/B testing evidence available.",
        )
        return

    # -----------------------------------------------------------------
    # Summary KPI cards
    # iter11 fix (verify_v3 P06 #1): when no experiments are logged, render
    # an explicit empty state instead of `0/0/N/A/0.0%` zero KPI tiles.
    # -----------------------------------------------------------------
    total_experiments = int(summary.get("total_experiments", len(experiments)) or 0)
    if total_experiments == 0:
        st.info(_tr(
            "No experiments logged yet - launch your first A/B test from "
            "the Retention Campaign Builder (Page 10) and re-run the "
            "pipeline to populate this view. The Power Analysis & Sample "
            "Size Calculator below is still usable for planning."
        ))
    else:
        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric(_tr("Total Experiments"), total_experiments)
        kc2.metric(
            _tr("Significant Results"),
            summary.get("significant_count", 0),
        )
        kc3.metric(
            _tr("Best Experiment"),
            summary.get("best_experiment", "N/A"),
        )
        kc4.metric(
            _tr("Avg Lift"),
            f"{summary.get('avg_lift', 0):.1%}",
        )

    # -----------------------------------------------------------------
    # Per-experiment detailed results
    # -----------------------------------------------------------------
    for i, exp in enumerate(experiments):
        exp_name = exp.get("name", f"Experiment {i + 1}")
        is_sig = exp.get("is_significant", False)
        sig_icon = "✅" if is_sig else "⚠️"

        st.subheader(f"{sig_icon} {_tr('Experiment')}: {exp_name}")

        # Metrics row
        ec1, ec2, ec3, ec4, ec5 = st.columns(5)
        ec1.metric(
            _tr("Treatment Churn"),
            f"{exp.get('treatment_churn_rate', 0):.2%}",
        )
        ec2.metric(
            _tr("Control Churn"),
            f"{exp.get('control_churn_rate', 0):.2%}",
        )
        ec3.metric(_tr("Lift"), f"{exp.get('lift', 0):.1%}")
        ec4.metric(_tr("p-value"), f"{exp.get('p_value', 1.0):.4f}")
        ec5.metric(_tr("Power"), f"{exp.get('power', 0):.2%}")

        # Significance indicator
        p_val = exp.get("p_value", 1.0)
        alpha = exp.get("alpha", 0.05)
        if is_sig:
            st.success(
                f"{_tr('Statistically Significant')} (p={p_val:.4f} < α={alpha})"
            )
        else:
            st.warning(
                f"{_tr('Not Significant')} (p={p_val:.4f} >= α={alpha})"
            )

        # Charts for this experiment
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            # Churn rate comparison with CI
            ci = exp.get("confidence_interval", [0, 0])
            effect = exp.get("absolute_effect", 0)
            fig_rate = go.Figure()
            fig_rate.add_trace(go.Bar(
                name=_tr("Treatment"),
                x=[_tr("Churn Rate")],
                y=[exp.get("treatment_churn_rate", 0)],
                marker_color="#2ecc71",
                width=0.3,
            ))
            fig_rate.add_trace(go.Bar(
                name=_tr("Control"),
                x=[_tr("Churn Rate")],
                y=[exp.get("control_churn_rate", 0)],
                marker_color="#e74c3c",
                width=0.3,
            ))
            fig_rate.update_layout(
                title=f"{_tr('Churn Rate')}: {exp_name}",
                barmode="group",
                yaxis_title=_tr("Churn Rate"),
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig_rate, use_container_width=True)

        with col_chart2:
            # Confidence interval visualization
            fig_ci = go.Figure()
            fig_ci.add_trace(go.Scatter(
                x=[effect],
                y=[exp_name],
                mode="markers",
                marker=dict(size=12, color="#3498db"),
                name=_tr("Effect Size"),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[ci[1] - effect] if len(ci) > 1 else [0],
                    arrayminus=[effect - ci[0]] if len(ci) > 0 else [0],
                    color="#3498db",
                    thickness=2,
                    width=6,
                ),
            ))
            fig_ci.add_vline(
                x=0, line_dash="dash", line_color="red",
                annotation_text=_tr("No Effect"),
            )
            fig_ci.update_layout(
                title=_tr("Effect Size & 95% CI"),
                xaxis_title=_tr("Absolute Effect (Churn Reduction)"),
                xaxis_tickformat=".0%",
                height=250,
            )
            st.plotly_chart(fig_ci, use_container_width=True)

        # Statistical details expander
        with st.expander(f"{_tr('Statistical Details')} - {exp_name}"):
            detail_cols = st.columns(3)
            detail_cols[0].markdown(f"""
**Sample Sizes**
- Treatment: {exp.get('treatment_size', 0):,}
- Control: {exp.get('control_size', 0):,}
- Total: {exp.get('treatment_size', 0) + exp.get('control_size', 0):,}
            """)
            detail_cols[1].markdown(f"""
**Effect Measures**
- Absolute Effect: {exp.get('absolute_effect', 0):.4f}
- Relative Lift: {exp.get('lift', 0):.1%}
- Cohen's h: {exp.get('effect_size_cohens_h', 0):.4f}
            """)
            detail_cols[2].markdown(f"""
**Test Configuration**
- Test Type: {exp.get('test_type', 'N/A')}
- Significance Level (α): {exp.get('alpha', 0.05)}
- Statistical Power: {exp.get('power', 0):.2%}
- Duration: {exp.get('duration_days', 0)} days
            """)

        st.markdown("---")

    # -----------------------------------------------------------------
    # Cross-experiment comparison
    # -----------------------------------------------------------------
    if len(experiments) > 1:
        st.subheader(_tr("Cross-Experiment Comparison"))

        exp_comparison = pd.DataFrame([
            {
                "Experiment": e.get("name", ""),
                "Treatment Churn": e.get("treatment_churn_rate", 0),
                "Control Churn": e.get("control_churn_rate", 0),
                "Lift": e.get("lift", 0),
                "p-value": e.get("p_value", 1.0),
                "Power": e.get("power", 0),
                "Significant": "Yes" if e.get("is_significant") else "No",
                "Cohen's h": e.get("effect_size_cohens_h", 0),
            }
            for e in experiments
        ])
        st.dataframe(
            exp_comparison.style.format({
                "Treatment Churn": "{:.2%}",
                "Control Churn": "{:.2%}",
                "Lift": "{:.1%}",
                "p-value": "{:.4f}",
                "Power": "{:.2%}",
                "Cohen's h": "{:.4f}",
            }),
            use_container_width=True,
        )

        # Lift comparison chart
        col_comp1, col_comp2 = st.columns(2)
        with col_comp1:
            fig_lift = px.bar(
                exp_comparison,
                x="Experiment", y="Lift",
                title=_tr("Relative Lift by Experiment"),
                color="Significant",
                color_discrete_map={
                    "Yes": "#2ecc71", "No": "#e74c3c",
                },
                text="Lift",
            )
            fig_lift.update_traces(texttemplate="%{text:.1%}")
            fig_lift.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig_lift, use_container_width=True)

        with col_comp2:
            # Power vs p-value scatter
            fig_power = px.scatter(
                exp_comparison,
                x="p-value", y="Power",
                size="Cohen's h",
                color="Significant",
                text="Experiment",
                title=_tr("Statistical Power vs p-value"),
                color_discrete_map={
                    "Yes": "#2ecc71", "No": "#e74c3c",
                },
            )
            fig_power.add_vline(
                x=0.05, line_dash="dash", line_color="gray",
                annotation_text="α=0.05",
            )
            fig_power.add_hline(
                y=0.80, line_dash="dash", line_color="gray",
                annotation_text=_tr("80% Power"),
            )
            fig_power.update_layout(
                xaxis_tickformat=".3f",
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig_power, use_container_width=True)

    # -----------------------------------------------------------------
    # Power Analysis / Sample Size Calculator
    # -----------------------------------------------------------------
    st.subheader(_tr("Power Analysis & Sample Size Calculator"))
    st.markdown(_tr(
        "Estimate required sample sizes and statistical power for "
        "planning future A/B experiments."
    ))

    pa_col1, pa_col2, pa_col3 = st.columns(3)
    with pa_col1:
        baseline_rate = st.slider(
            _tr("Baseline Churn Rate"),
            min_value=0.01,
            max_value=0.50,
            value=0.20,
            step=0.01,
            key="pa_baseline",
            help=_tr("Expected churn rate without treatment"),
        )
    with pa_col2:
        mde = st.slider(
            _tr("Minimum Detectable Effect (MDE)"),
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            key="pa_mde",
            help=_tr("Smallest effect size you want to detect"),
        )
    with pa_col3:
        pa_alpha = st.selectbox(
            _tr("Significance Level (α)"),
            options=[0.01, 0.05, 0.10],
            index=1,
            key="pa_alpha",
        )
        pa_power_target = st.selectbox(
            _tr("Target Power (1-β)"),
            options=[0.80, 0.85, 0.90, 0.95],
            index=0,
            key="pa_power_target",
        )

    # Compute sample size using normal approximation for two proportions
    pa_results = _compute_power_analysis(
        baseline_rate=baseline_rate,
        mde=mde,
        alpha=pa_alpha,
        power=pa_power_target,
    )

    pa_kpi1, pa_kpi2, pa_kpi3 = st.columns(3)
    pa_kpi1.metric(
        _tr("Required Sample Size (per group)"),
        f"{pa_results['sample_size_per_group']:,}",
    )
    pa_kpi2.metric(
        _tr("Total Participants Needed"),
        f"{pa_results['total_participants']:,}",
    )
    pa_kpi3.metric(
        _tr("Expected Duration (days)"),
        f"{pa_results['estimated_duration_days']}",
        help=_tr("Based on 100 new enrollments per day"),
    )

    # Power curve chart
    st.markdown(f"**{_tr('Power Curve: Sample Size vs Statistical Power')}**")
    power_curve = _compute_power_curve(
        baseline_rate=baseline_rate,
        mde=mde,
        alpha=pa_alpha,
        max_n=pa_results["sample_size_per_group"] * 3,
    )
    fig_pcurve = go.Figure()
    fig_pcurve.add_trace(go.Scatter(
        x=power_curve["n"],
        y=power_curve["power"],
        mode="lines",
        name=_tr("Power"),
        line=dict(color="#3498db", width=2),
    ))
    fig_pcurve.add_hline(
        y=pa_power_target,
        line_dash="dash",
        line_color="red",
        annotation_text=f"{_tr('Target')}: {pa_power_target:.0%}",
    )
    fig_pcurve.add_vline(
        x=pa_results["sample_size_per_group"],
        line_dash="dash",
        line_color="green",
        annotation_text=f"n={pa_results['sample_size_per_group']:,}",
    )
    fig_pcurve.update_layout(
        title=_tr("Power vs Sample Size"),
        xaxis_title=_tr("Sample Size per Group"),
        yaxis_title=_tr("Statistical Power"),
        yaxis_tickformat=".0%",
        height=350,
    )
    st.plotly_chart(fig_pcurve, use_container_width=True)

    # MDE sensitivity table
    st.markdown(f"**{_tr('MDE Sensitivity Analysis')}**")
    mde_table = _compute_mde_sensitivity(
        baseline_rate=baseline_rate,
        alpha=pa_alpha,
        power=pa_power_target,
    )
    # iter11 fix (verify_v3 P06 #1): feasibility guard - warn when the
    # required total participants exceeds the available customer pool
    # (e.g. MDE 1% needs 48,882 vs n=20,000 pool).
    pool_size = int(
        config.get("simulator", {}).get("num_customers", 0) or 0
    )
    infeasible_rows = []
    if pool_size > 0 and not mde_table.empty:
        for _, _row in mde_table.iterrows():
            try:
                total_required = int(_row.get("Total Participants", 0))
                mde_val = float(_row.get("MDE", 0))
            except (TypeError, ValueError):
                continue
            if total_required > pool_size and mde_val > 0:
                infeasible_rows.append((mde_val, total_required))
    if infeasible_rows:
        # Sort smallest MDE first (the most-infeasible row)
        infeasible_rows.sort(key=lambda x: x[0])
        examples = "; ".join(
            f"MDE {m:.1%} needs {n:,} vs {pool_size:,} pool"
            for m, n in infeasible_rows[:3]
        )
        st.warning(
            _tr("Feasibility check: some MDE rows below require more "
            "participants than the available customer pool")
            + f" ({pool_size:,}). "
            + f"{_tr('Infeasible row(s)')}: {examples}. "
            + _tr("Consider tightening the "
            "MDE target or running a longer experiment with cohort "
            "rotation.")
        )
    st.dataframe(
        mde_table.style.format({
            "MDE": "{:.1%}",
            "Sample Size (per group)": "{:,.0f}",
            "Total Participants": "{:,.0f}",
        }),
        use_container_width=True,
    )

    # -----------------------------------------------------------------
    # Multiple Comparison Correction Summary
    # -----------------------------------------------------------------
    if len(experiments) > 1:
        st.subheader(_tr("Multiple Comparison Correction"))
        st.markdown(_tr(
            "When running multiple experiments simultaneously, p-values "
            "should be corrected to control the family-wise error rate."
        ))
        p_values = [e.get("p_value", 1.0) for e in experiments]
        exp_names = [e.get("name", f"Exp {i+1}") for i, e in enumerate(experiments)]
        correction_df = _compute_multiple_comparison_corrections(
            p_values=p_values,
            experiment_names=exp_names,
            alpha=0.05,
        )
        st.dataframe(
            correction_df.style.format({
                "Raw p-value": "{:.4f}",
                "Bonferroni": "{:.4f}",
                "Holm-Bonferroni": "{:.4f}",
                "BH (FDR)": "{:.4f}",
            }),
            use_container_width=True,
        )


def render_survival_analysis(st_module, config: Dict, data_loader=None):
    """Render survival analysis page with Kaplan-Meier curves.

    Shows KPI summary, Kaplan-Meier survival curves per segment with
    confidence intervals, median survival times, hazard rate comparison,
    event rate analysis, and duration distributions.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s  # fallback if helper unavailable

    st = st_module
    st.header(_tr("Survival Analysis"))

    if data_loader is None:
        data_loader = get_data_loader(config)

    # iter13 G3 P1 fix: refuse to render synthetic survival curves derived
    # from churn predictions (duration_days = 365 * (1 - churn_prob)).
    # Require real `survival_data.csv` / `survival_curves.json` artifacts.
    survival, surv_is_real, surv_reason = _load_as_artifact(
        data_loader, "load_survival_data",
    )
    if not surv_is_real or survival is None or (
        hasattr(survival, "empty") and survival.empty
    ):
        st.error(
            _tr("Real survival artifacts missing — run `python -m src.main "
            "--mode all` to generate `results/survival_data.csv` and "
            "`results/survival_curves.json`. The previous Kaplan-Meier / "
            "hazard / median-duration KPIs were derived from churn "
            "predictions (duration = 365 × (1 − churn_prob)) and are not a "
            "fitted Cox PH output.")
            + f" ({surv_reason or _tr('artifact not found')})"
        )
        return

    # -----------------------------------------------------------------
    # KPI summary cards.
    # iter9 audit P07 #19: "Events Observed = 5,717" was the same number
    # Page 01 reports as "High Risk count" (predictions, not outcomes).
    # Rename to make the prediction-derived nature explicit and add a
    # right-censoring annotation on the median duration (#20).
    # -----------------------------------------------------------------
    kc1, kc2, kc3, kc4 = st.columns(4)
    total_cust = len(survival)
    event_count = survival["event_observed"].sum()
    event_rate = event_count / total_cust if total_cust > 0 else 0
    median_duration = survival["duration_days"].median()
    max_duration = survival["duration_days"].max()
    # iter11 P07 #4 fix: lower the right-censoring threshold to 0.85 so
    # the actual 309/350=0.883 ratio audited on iter10 actually surfaces
    # the warning, AND show the ratio unconditionally so analysts can
    # judge censoring proximity even when the threshold doesn't fire.
    censoring_ratio = (
        float(median_duration) / float(max_duration)
        if (
            median_duration is not None
            and max_duration is not None
            and float(max_duration) > 0
        )
        else None
    )
    is_right_censored = (
        censoring_ratio is not None and censoring_ratio >= 0.85
    )

    kc1.metric(_tr("Total Customers"), format_count(total_cust))
    kc2.metric(
        _tr("Predicted Churners (>50%)"),
        format_count(int(event_count)),
        help=_tr(
            "Count of customers whose predicted churn probability "
            "exceeds 50%. This is a prediction-derived label and matches "
            "the High Risk count on Page 01 by construction — it is NOT "
            "an observed event count (iter9 audit P07 #19)."
        ),
    )
    kc3.metric(
        _tr("Predicted Churn Rate"),
        f"{event_rate:.2%}",
        help=_tr("Fraction of customers with predicted churn prob >50%."),
    )
    if is_right_censored:
        kc4.metric(
            _tr("Median Duration *"),
            f"{median_duration:.0f} {_tr('days')}",
            help=(
                f"* {_tr('right-censored at observation window')} "
                f"({max_duration:.0f} {_tr('days')}, {_tr('ratio')} "
                f"{censoring_ratio:.1%}). {_tr('True median may be longer.')}"
            ),
        )
        st.caption(
            f"⚠️ {_tr('Median')} {median_duration:.0f} d / {_tr('observation horizon')} "
            f"~{max_duration:.0f} d ({censoring_ratio:.1%}). " + _tr(
            "Right-censoring artifact possible above ~85% ratio — the displayed "
            "median is bounded by the observation window.")
        )
    else:
        kc4.metric(_tr("Median Duration"), f"{median_duration:.0f} {_tr('days')}")
        if censoring_ratio is not None:
            st.caption(
                f"{_tr('Median')} {median_duration:.0f} d / {_tr('observation horizon')} "
                f"~{max_duration:.0f} d ({censoring_ratio:.1%}). "
                + _tr("Right-censoring artifact possible above ~85% ratio.")
            )

    # -----------------------------------------------------------------
    # Kaplan-Meier Survival Curves by Segment
    # iter13 G3 P1 fix: KM curves require real `survival_curves.json`.
    # The legacy loader fell back to synthetic exponential decays per
    # segment when the artifact was missing — render an explicit error
    # instead so analysts cannot mistake fixture curves for fitted KM.
    # -----------------------------------------------------------------
    st.subheader(_tr("Kaplan-Meier Survival Curves by Segment"))
    surv_curves, surv_curves_is_real, surv_curves_reason = _load_as_artifact(
        data_loader, "load_survival_curves",
    )
    if not surv_curves_is_real or not surv_curves:
        st.error(
            _tr("Real survival-curve data missing — run `python -m src.main "
            "--mode all` to generate `results/survival_curves.json`.")
            + f" ({surv_curves_reason or _tr('artifact not found')})"
        )
        return

    segment_colors = {
        "vip_loyal": "#2ecc71",
        "regular_loyal": "#3498db",
        "bargain_hunter": "#e67e22",
        "explorer": "#9b59b6",
        "dormant": "#e74c3c",
        "new_customer": "#f39c12",
    }

    fig_km = go.Figure()
    for seg_name, curve_data in surv_curves.items():
        timeline = curve_data.get("timeline", [])
        surv_prob = curve_data.get("survival_prob", [])
        ci_lower = curve_data.get("ci_lower", [])
        ci_upper = curve_data.get("ci_upper", [])
        color = segment_colors.get(seg_name, "#95a5a6")

        # Main survival curve
        fig_km.add_trace(go.Scatter(
            x=timeline, y=surv_prob,
            mode="lines",
            name=seg_name,
            line=dict(color=color, width=2),
        ))
        # Confidence interval band
        if ci_lower and ci_upper:
            fig_km.add_trace(go.Scatter(
                x=timeline + timeline[::-1],
                y=ci_upper + ci_lower[::-1],
                fill="toself",
                fillcolor=color.replace(")", ",0.1)").replace(
                    "#", "rgba("
                ) if color.startswith("rgba") else f"rgba(150,150,150,0.1)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                name=f"{seg_name} CI",
            ))

    # Median survival reference line
    fig_km.add_hline(
        y=0.5, line_dash="dash", line_color="gray",
        annotation_text=_tr("50% Survival (Median)"),
    )
    fig_km.update_layout(
        title=_tr("Kaplan-Meier Survival Curves by Customer Segment"),
        xaxis_title=_tr("Days Since First Purchase"),
        yaxis_title=_tr("Survival Probability"),
        yaxis_range=[0, 1.05],
        hovermode="x unified",
    )
    st.plotly_chart(fig_km, use_container_width=True)

    # -----------------------------------------------------------------
    # Median Survival Times Table
    # -----------------------------------------------------------------
    st.subheader(_tr("Median Survival Time by Segment"))
    median_data = []
    for seg_name, curve_data in surv_curves.items():
        median_surv = curve_data.get("median_survival_days")
        surv_prob_list = curve_data.get("survival_prob", [])
        final_surv = surv_prob_list[-1] if surv_prob_list else 0
        median_data.append({
            "Segment": seg_name,
            "Median Survival (days)": (
                median_surv if median_surv else ">360"
            ),
            "Final Survival Prob": final_surv,
            "Risk Level": (
                "Low" if final_surv > 0.7
                else "Medium" if final_surv > 0.5
                else "High" if final_surv > 0.3
                else "Critical"
            ),
        })
    median_df = pd.DataFrame(median_data)
    st.dataframe(median_df, use_container_width=True)

    # -----------------------------------------------------------------
    # Avg Survival Probability by Behavioral Segment — iter16 fix #5:
    #
    # Previous title/caption claimed this chart used the 8-way uplift
    # taxonomy, but the underlying `survival.groupby("segment")` call
    # actually groups by the same 6-way behavioral taxonomy used by the
    # Kaplan-Meier and hazard charts above. The title is corrected.
    #
    # Switched the aggregated column from `survival_probability` (which
    # is Cox PH S(90 days)) to `survival_prob_365d` because at t=90 every
    # behavioral segment is still near the ceiling (~0.99, ~1pp spread),
    # so the chart looked uniformly flat. At t=365 the spread widens to
    # ~88pp (dormant ~5% vs vip_loyal ~93%), which is what users expect
    # from a "segment survival" chart.
    # -----------------------------------------------------------------
    st.markdown(f"### {_tr('Average Survival Probability by Behavioral Segment')}")
    st.caption(_tr(
        "Bars show mean Cox PH-derived survival probability at t=365 days "
        "for each behavioral segment (same 6 segments as the Kaplan-Meier "
        "curves above). Earlier versions of this chart used t=90 days, at "
        "which point every segment was still near the ceiling and the "
        "bars looked artificially uniform. For uplift-segment analysis "
        "see Page 11 (Uplift Modeling)."
    ))
    surv_col = (
        "survival_prob_365d"
        if "survival_prob_365d" in survival.columns
        else "survival_probability"
    )
    seg_surv = survival.groupby("segment")[surv_col].mean().reset_index()
    seg_surv.columns = ["Segment", "Avg Survival Prob (1y)"]
    seg_surv = seg_surv.sort_values("Avg Survival Prob (1y)", ascending=True)
    fig = px.bar(
        seg_surv, x="Avg Survival Prob (1y)", y="Segment",
        orientation="h",
        title=_tr("Average Survival Probability at 1 Year by Behavioral Segment (Cox PH)"),
        color="Avg Survival Prob (1y)",
        color_continuous_scale="RdYlGn",
        text="Avg Survival Prob (1y)",
    )
    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------
    # Hazard Rate Comparison — uses behavioral taxonomy (6 segments),
    # consistent with the Kaplan-Meier curves above and Page 03.
    # -----------------------------------------------------------------
    st.markdown(f"### {_tr('Estimated Hazard Rate by Behavioral Segment')}")
    hazard_data = []
    for seg_name, curve_data in surv_curves.items():
        surv_prob_list = curve_data.get("survival_prob", [1.0, 0.5])
        timeline = curve_data.get("timeline", [0, 360])
        if len(surv_prob_list) >= 2 and surv_prob_list[0] > 0:
            # Approximate hazard rate from survival curve
            s0 = surv_prob_list[0]
            s_end = surv_prob_list[-1]
            t_max = timeline[-1] if timeline else 360
            hazard = -np.log(max(s_end / s0, 0.001)) / max(t_max, 1)
            hazard_data.append({
                "Segment": seg_name,
                "Hazard Rate (per day)": round(hazard, 6),
                "Hazard Rate (annualized)": round(hazard * 365, 4),
            })

    if hazard_data:
        hazard_df = pd.DataFrame(hazard_data)
        col_hz1, col_hz2 = st.columns(2)
        with col_hz1:
            fig_hz = px.bar(
                hazard_df, x="Segment", y="Hazard Rate (per day)",
                title=_tr("Daily Hazard Rate by Segment"),
                color="Segment",
                text="Hazard Rate (per day)",
            )
            fig_hz.update_traces(
                texttemplate="%{text:.5f}", textposition="outside",
            )
            st.plotly_chart(fig_hz, use_container_width=True)

        with col_hz2:
            st.dataframe(
                hazard_df.style.format({
                    "Hazard Rate (per day)": "{:.6f}",
                    "Hazard Rate (annualized)": "{:.4f}",
                }),
                use_container_width=True,
            )

    # -----------------------------------------------------------------
    # Event Rate by Segment — iter11 P07 #2 fix: this uses the uplift
    # taxonomy (e.g. high_value_lost_cause, *_persuadable, *_sure_thing).
    # Several of those segment names ARE DEFINED post-hoc using the
    # churn outcome itself, so the resulting per-segment event rate is
    # tautological (sure_thing -> ~0%, lost_cause -> ~100%). Surface the
    # warning prominently and steer the analyst to Avg Survival Prob.
    # -----------------------------------------------------------------
    st.markdown(f"### {_tr('Event Rate by Uplift Segment')}")
    st.warning(
        _tr("⚠ Event Rate per segment is derived from current outcome labels — "
        "these uplift segments (sure_thing / lost_cause / persuadable / "
        "sleeping_dog) are defined post-hoc using churn outcome, so the "
        "binary 0% / 100% pattern is tautological and NOT a model finding. "
        "Use **Avg Survival Probability** (Cox PH-derived) above for proper "
        "per-segment risk."),
        icon="⚠️",
    )
    event_stats = survival.groupby("segment").agg(
        total=("customer_id", "count"),
        events=("event_observed", "sum"),
        avg_duration=("duration_days", "mean"),
        median_duration=("duration_days", "median"),
    ).reset_index()
    event_stats["event_rate"] = event_stats["events"] / event_stats["total"]
    event_stats.columns = [
        "Segment", "Total", "Events", "Avg Duration",
        "Median Duration", "Event Rate",
    ]

    col_ev1, col_ev2 = st.columns(2)
    with col_ev1:
        fig_ev = px.bar(
            event_stats, x="Segment", y="Event Rate",
            title=_tr("Churn Event Rate by Uplift Segment (label-leak — see warning)"),
            color="Event Rate",
            color_continuous_scale="RdYlGn_r",
            text="Event Rate",
        )
        fig_ev.update_traces(
            texttemplate="%{text:.1%}", textposition="outside",
        )
        fig_ev.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_ev, use_container_width=True)

    with col_ev2:
        st.dataframe(
            event_stats.style.format({
                "Avg Duration": "{:.1f}",
                "Median Duration": "{:.1f}",
                "Event Rate": "{:.2%}",
            }),
            use_container_width=True,
        )

    # -----------------------------------------------------------------
    # Duration distribution
    # -----------------------------------------------------------------
    st.subheader(_tr("Customer Lifetime Duration Distribution"))
    fig2 = px.histogram(
        survival, x="duration_days", nbins=30,
        title=_tr("Distribution of Customer Durations"),
        color="segment",
        barmode="overlay",
        opacity=0.7,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Duration box plot by segment
    st.subheader(_tr("Duration Distribution by Segment"))
    fig_box = px.box(
        survival, x="segment", y="duration_days",
        color="segment",
        title=_tr("Customer Duration Distribution by Segment"),
        labels={
            "segment": _tr("Segment"),
            "duration_days": _tr("Duration (days)"),
        },
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # -----------------------------------------------------------------
    # Config info
    # -----------------------------------------------------------------
    surv_config = config.get("survival", {})
    if surv_config:
        st.markdown("---")
        st.subheader(_tr("Survival Model Configuration"))
        st.json({
            "penalizer": surv_config.get("penalizer", "N/A"),
            "l1_ratio": surv_config.get("l1_ratio", "N/A"),
            "alpha": surv_config.get("alpha", "N/A"),
        })


def render_recommendations(st_module, config: Dict, data_loader=None):
    """Render personalized recommendations page.

    Delegates to the enhanced recommendations view module which provides
    KPI cards, distribution analysis, uplift analysis, segment breakdown,
    cost-benefit analysis, and filterable tables.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s

    if data_loader is None:
        data_loader = get_data_loader(config)

    render_recommendations_view(st_module, config, data_loader)


def _render_recommendations_legacy(st_module, config: Dict, data_loader=None):
    """Legacy recommendations renderer (kept for backward compatibility).

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s

    st = st_module
    st.header(_tr("Personalized Recommendations"))

    if data_loader is None:
        data_loader = get_data_loader(config)

    recs = data_loader.load_recommendations()

    if recs.empty:
        st.warning(_tr("No recommendations available."))
        return

    # -----------------------------------------------------------------
    # KPI summary cards
    # -----------------------------------------------------------------
    kc1, kc2, kc3, kc4 = st.columns(4)
    total_customers = len(recs)
    actionable = recs[
        recs["recommendation_type"] != "no_action"
    ] if "recommendation_type" in recs.columns else recs
    actionable_count = len(actionable)
    avg_uplift = recs["expected_uplift"].mean() if "expected_uplift" in recs.columns else 0
    avg_priority = recs["priority_score"].mean() if "priority_score" in recs.columns else 0

    kc1.metric(_tr("Total Customers"), f"{total_customers:,}")
    kc2.metric(_tr("Actionable Recommendations"), f"{actionable_count:,}")
    kc3.metric(_tr("Avg Expected Uplift"), f"{avg_uplift:.2%}")
    kc4.metric(_tr("Avg Priority Score"), f"{avg_priority:.2f}")

    # -----------------------------------------------------------------
    # Priority-ranked recommendations table
    # -----------------------------------------------------------------
    st.subheader(_tr("Priority-Ranked Retention Actions"))
    recs_sorted = recs.sort_values("priority_score", ascending=False)
    st.dataframe(recs_sorted, use_container_width=True)

    # -----------------------------------------------------------------
    # Recommendation type distribution
    # -----------------------------------------------------------------
    st.subheader(_tr("Recommendation Type Distribution"))
    col_rt1, col_rt2 = st.columns(2)
    with col_rt1:
        fig_pie = px.pie(
            recs, names="recommendation_type",
            title=_tr("Action Type Distribution"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_rt2:
        type_counts = recs["recommendation_type"].value_counts().reset_index()
        type_counts.columns = ["Action Type", "Count"]
        fig_bar_types = px.bar(
            type_counts, x="Action Type", y="Count",
            title=_tr("Recommendation Counts by Type"),
            color="Action Type",
            text="Count",
        )
        fig_bar_types.update_traces(textposition="outside")
        st.plotly_chart(fig_bar_types, use_container_width=True)

    # -----------------------------------------------------------------
    # Expected Uplift Analysis
    # -----------------------------------------------------------------
    st.subheader(_tr("Expected Uplift by Customer"))
    fig_uplift = px.bar(
        recs_sorted, x="customer_id", y="expected_uplift",
        color="recommendation_type",
        title=_tr("Expected Retention Uplift per Customer"),
        labels={
            "expected_uplift": "Expected Uplift",
            "customer_id": "Customer ID",
        },
    )
    st.plotly_chart(fig_uplift, use_container_width=True)

    # Uplift distribution
    st.subheader(_tr("Uplift Score Distribution"))
    fig_uplift_hist = px.histogram(
        recs, x="expected_uplift", nbins=20,
        title=_tr("Distribution of Expected Uplift Scores"),
        color="recommendation_type",
        barmode="overlay",
        opacity=0.7,
    )
    st.plotly_chart(fig_uplift_hist, use_container_width=True)

    # -----------------------------------------------------------------
    # Priority vs Uplift scatter
    # -----------------------------------------------------------------
    st.subheader(_tr("Priority Score vs Expected Uplift"))
    fig_scatter = px.scatter(
        recs, x="priority_score", y="expected_uplift",
        color="recommendation_type",
        size="priority_score",
        title=_tr("Priority Score vs Expected Uplift"),
        labels={
            "priority_score": "Priority Score",
            "expected_uplift": "Expected Uplift",
        },
        hover_data=["customer_id"],
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # -----------------------------------------------------------------
    # Segment-level action breakdown (if segment column exists)
    # -----------------------------------------------------------------
    if "segment" in recs.columns:
        st.subheader(_tr("Retention Actions by Segment"))
        seg_action = recs.groupby(
            ["segment", "recommendation_type"]
        ).size().reset_index(name="count")
        fig_seg = px.bar(
            seg_action, x="segment", y="count",
            color="recommendation_type",
            title=_tr("Recommended Actions by Customer Segment"),
            barmode="group",
        )
        st.plotly_chart(fig_seg, use_container_width=True)

        # Average uplift by segment
        seg_uplift = recs.groupby("segment")[
            "expected_uplift"
        ].mean().reset_index()
        seg_uplift.columns = ["Segment", "Avg Expected Uplift"]
        seg_uplift = seg_uplift.sort_values(
            "Avg Expected Uplift", ascending=True,
        )
        fig_seg_uplift = px.bar(
            seg_uplift, x="Avg Expected Uplift", y="Segment",
            orientation="h",
            title=_tr("Average Expected Uplift by Segment"),
            color="Avg Expected Uplift",
            color_continuous_scale="Viridis",
            text="Avg Expected Uplift",
        )
        fig_seg_uplift.update_traces(
            texttemplate="%{text:.2%}", textposition="outside",
        )
        st.plotly_chart(fig_seg_uplift, use_container_width=True)

    # -----------------------------------------------------------------
    # Cost-effectiveness analysis (if estimated_cost column exists)
    # -----------------------------------------------------------------
    if "estimated_cost" in recs.columns:
        st.subheader(_tr("Cost-Effectiveness Analysis"))
        total_cost = recs["estimated_cost"].sum()
        st.metric(_tr("Total Estimated Cost"), format_currency(total_cost, "KRW"))

        cost_by_type = recs.groupby("recommendation_type").agg(
            total_cost=("estimated_cost", "sum"),
            avg_cost=("estimated_cost", "mean"),
            count=("customer_id", "count"),
            avg_uplift=("expected_uplift", "mean"),
        ).reset_index()
        cost_by_type["cost_per_uplift_pct"] = (
            cost_by_type["avg_cost"]
            / cost_by_type["avg_uplift"].clip(lower=0.001)
        )
        st.dataframe(cost_by_type, use_container_width=True)

    # -----------------------------------------------------------------
    # Top priority customers
    # -----------------------------------------------------------------
    st.subheader(_tr("Top Priority Customers for Retention"))
    top_n = min(10, len(recs_sorted))
    top_customers = recs_sorted.head(top_n)
    st.dataframe(top_customers, use_container_width=True)

    # -----------------------------------------------------------------
    # Recommendation configuration
    # -----------------------------------------------------------------
    rec_config = config.get("recommendations", {})
    if rec_config:
        st.markdown("---")
        st.subheader(_tr("Recommendation Engine Configuration"))
        st.json(rec_config)


def render_clv(st_module, config: Dict, data_loader=None):
    """Render CLV prediction page with distributions and segment analysis.

    Shows CLV distribution (histogram + box plot), segment-level CLV
    breakdown, CLV-vs-churn scatter, CLV percentile analysis, top/bottom
    customer tables, and CLV tier classification.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s

    st = st_module
    st.header(_tr("CLV Prediction"))

    if data_loader is None:
        data_loader = get_data_loader(config)

    predictions = data_loader.load_predictions()
    clv_data = data_loader.load_clv_data()

    if clv_data.empty:
        _show_loader_issue(
            st,
            data_loader,
            "clv_data",
            "No CLV data available.",
        )
        return
    if hasattr(data_loader, "get_artifact_issue") and data_loader.get_artifact_issue(
        "clv_data"
    ):
        _show_loader_issue(
            st,
            data_loader,
            "clv_data",
            "CLV data has partial coverage.",
        )
    if not predictions.empty:
        prediction_cols = [
            c for c in ["customer_id", "churn_probability", "risk_level"]
            if c in predictions.columns
        ]
        predictions = clv_data.merge(
            predictions[prediction_cols],
            on="customer_id",
            how="left",
        )
    else:
        predictions = clv_data.copy()
    if "churn_probability" not in predictions.columns:
        predictions["churn_probability"] = np.nan
    if "risk_level" not in predictions.columns:
        predictions["risk_level"] = "unknown"

    currency = config.get("budget", {}).get("currency", "KRW")

    # -----------------------------------------------------------------
    # KPI summary cards
    # -----------------------------------------------------------------
    kc1, kc2, kc3, kc4 = st.columns(4)
    total_clv = predictions["clv_predicted"].sum()
    avg_clv = predictions["clv_predicted"].mean()
    median_clv = predictions["clv_predicted"].median()
    std_clv = predictions["clv_predicted"].std()

    kc1.metric(_tr("Total CLV"), f"{total_clv:,.0f} {currency}")
    kc2.metric(_tr("Average CLV"), f"{avg_clv:,.0f} {currency}")
    kc3.metric(_tr("Median CLV"), f"{median_clv:,.0f} {currency}")
    kc4.metric(_tr("CLV Std Dev"), f"{std_clv:,.0f} {currency}")

    # -----------------------------------------------------------------
    # CLV distribution - histogram + box plot side by side
    # -----------------------------------------------------------------
    st.subheader(_tr("CLV Distribution"))
    col_hist, col_box = st.columns(2)

    with col_hist:
        fig_hist = px.histogram(
            predictions, x="clv_predicted", nbins=50,
            title=_tr("Customer Lifetime Value Distribution"),
            labels={"clv_predicted": f"Predicted CLV ({currency})"},
            color_discrete_sequence=["#3498db"],
        )
        fig_hist.add_vline(
            x=avg_clv, line_dash="dash", line_color="red",
            annotation_text=f"Mean: {avg_clv:,.0f}",
        )
        fig_hist.add_vline(
            x=median_clv, line_dash="dot", line_color="green",
            annotation_text=f"Median: {median_clv:,.0f}",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_box:
        fig_box = px.box(
            predictions, y="clv_predicted", x="segment",
            title=_tr("CLV Distribution by Segment"),
            labels={"clv_predicted": f"CLV ({currency})", "segment": "Segment"},
            color="segment",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # -----------------------------------------------------------------
    # CLV by segment - detailed breakdown.
    # iter11 P10 #7 fix: hide segments with n<5 from the bar chart so a
    # n=1 / n=2 segment cannot visually dominate the headline. Hidden
    # segments are listed in a footnote and remain in the table below.
    # -----------------------------------------------------------------
    st.subheader(_tr("CLV by Segment"))
    seg_clv = predictions.groupby("segment")["clv_predicted"].agg(
        ["mean", "sum", "count", "median", "std"]
    ).reset_index()
    seg_clv.columns = [
        "Segment", "Mean CLV", "Total CLV", "Count", "Median CLV", "Std CLV",
    ]
    seg_clv = seg_clv.sort_values("Mean CLV", ascending=False)

    # Hide low-n segments from the chart only (table keeps everything).
    MIN_N_FOR_CHART = 5
    seg_clv_chart = seg_clv[seg_clv["Count"] >= MIN_N_FOR_CHART].copy()
    seg_clv_hidden = seg_clv[seg_clv["Count"] < MIN_N_FOR_CHART].copy()
    # Annotate the x-axis label with sample size: "Segment (n=1234)".
    seg_clv_chart["Segment_n"] = seg_clv_chart.apply(
        lambda r: f"{r['Segment']} (n={int(r['Count']):,})", axis=1
    )

    if len(seg_clv_hidden) > 0:
        hidden_names = ", ".join(
            f"{r['Segment']} (n={int(r['Count'])})"
            for _, r in seg_clv_hidden.iterrows()
        )
        st.caption(
            f"⚠ {len(seg_clv_hidden)} segment(s) hidden from the bar charts "
            f"because n < {MIN_N_FOR_CHART}: {hidden_names}. They are still "
            f"visible in the statistics table below."
        )

    col_bar, col_total = st.columns(2)
    with col_bar:
        fig_mean = px.bar(
            seg_clv_chart, x="Segment_n", y="Mean CLV",
            title=_tr("Average CLV by Segment (n>=5 only)"),
            color="Segment",
            text="Mean CLV",
            labels={"Segment_n": "Segment (n)"},
        )
        fig_mean.update_traces(
            texttemplate="%{text:,.0f}", textposition="outside",
        )
        st.plotly_chart(fig_mean, use_container_width=True)

    with col_total:
        fig_total = px.bar(
            seg_clv_chart, x="Segment_n", y="Total CLV",
            title=_tr("Total CLV by Segment (n>=5 only)"),
            color="Segment",
            text="Total CLV",
            labels={"Segment_n": "Segment (n)"},
        )
        fig_total.update_traces(
            texttemplate="%{text:,.0f}", textposition="outside",
        )
        st.plotly_chart(fig_total, use_container_width=True)

    # Segment CLV statistics table
    st.dataframe(
        seg_clv.style.format({
            "Mean CLV": "{:,.0f}",
            "Total CLV": "{:,.0f}",
            "Median CLV": "{:,.0f}",
            "Std CLV": "{:,.0f}",
        }),
        use_container_width=True,
    )

    # -----------------------------------------------------------------
    # CLV vs Churn Probability scatter.
    # iter11 P10 #5 fix: probability axis must be [0, 1]. Defensively
    # filter rows with out-of-range or NaN churn_probability before
    # plotting AND pass range_x=[0, 1] so plotly cannot auto-pad.
    # -----------------------------------------------------------------
    st.subheader(_tr("CLV vs Churn Risk"))
    scatter_df = predictions.dropna(subset=["churn_probability"]).copy()
    scatter_df = scatter_df[
        scatter_df["churn_probability"].between(0, 1)
    ]
    fig_scatter = px.scatter(
        scatter_df, x="churn_probability", y="clv_predicted",
        color="segment",
        title=_tr("CLV vs Churn Probability (High CLV + High Churn = Priority)"),
        labels={
            "churn_probability": "Churn Probability",
            "clv_predicted": f"Predicted CLV ({currency})",
        },
        hover_data=["customer_id"],
        opacity=0.7,
        range_x=[0, 1],
    )
    fig_scatter.update_xaxes(range=[0, 1])
    # Add quadrant lines
    fig_scatter.add_hline(
        y=median_clv, line_dash="dash", line_color="gray",
        annotation_text="Median CLV",
    )
    fig_scatter.add_vline(
        x=0.5, line_dash="dash", line_color="gray",
        annotation_text="Churn Threshold",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    n_dropped = len(predictions) - len(scatter_df)
    if n_dropped > 0:
        st.caption(
            f"{n_dropped:,} row(s) excluded from the scatter (NaN or "
            f"out-of-range churn_probability)."
        )

    # -----------------------------------------------------------------
    # CLV Tier Classification
    # -----------------------------------------------------------------
    st.subheader(_tr("CLV Tier Classification"))
    q25 = predictions["clv_predicted"].quantile(0.25)
    q50 = predictions["clv_predicted"].quantile(0.50)
    q75 = predictions["clv_predicted"].quantile(0.75)

    def _classify_clv_tier(v):
        if v >= q75:
            return "Platinum"
        elif v >= q50:
            return "Gold"
        elif v >= q25:
            return "Silver"
        return "Bronze"

    predictions_copy = predictions.copy()
    predictions_copy["clv_tier"] = predictions_copy["clv_predicted"].apply(
        _classify_clv_tier
    )

    tier_counts = predictions_copy["clv_tier"].value_counts().reset_index()
    tier_counts.columns = ["CLV Tier", "Count"]
    tier_order = ["Platinum", "Gold", "Silver", "Bronze"]
    tier_counts["CLV Tier"] = pd.Categorical(
        tier_counts["CLV Tier"], categories=tier_order, ordered=True,
    )
    tier_counts = tier_counts.sort_values("CLV Tier")

    col_pie, col_stats = st.columns(2)
    with col_pie:
        fig_tier = px.pie(
            tier_counts, values="Count", names="CLV Tier",
            title=_tr("CLV Tier Distribution"),
            color="CLV Tier",
            color_discrete_map={
                "Platinum": "#e5e4e2",
                "Gold": "#ffd700",
                "Silver": "#c0c0c0",
                "Bronze": "#cd7f32",
            },
        )
        st.plotly_chart(fig_tier, use_container_width=True)

    with col_stats:
        tier_stats = predictions_copy.groupby("clv_tier").agg(
            count=("customer_id", "count"),
            avg_clv=("clv_predicted", "mean"),
            avg_churn=("churn_probability", "mean"),
        ).reset_index()
        tier_stats.columns = ["CLV Tier", "Customers", "Avg CLV", "Avg Churn"]
        st.dataframe(
            tier_stats.style.format({
                "Avg CLV": "{:,.0f}",
                "Avg Churn": "{:.2%}",
            }),
            use_container_width=True,
        )

    # -----------------------------------------------------------------
    # Top & bottom customers.
    # iter11 P10 #6 fix: previously the churn_probability column rendered
    # as NaN/blank for every row when the predictions dataframe lacked
    # that column (it was filled with np.nan up-stream). Drop NaN rows
    # so the Top/Bottom 10 actually have churn information; if the
    # entire column is NaN, drop the column from the displayed tables
    # rather than show a blank column.
    # -----------------------------------------------------------------
    st.subheader(_tr("Top 10 Customers by CLV"))
    top_cols = ["customer_id", "clv_predicted", "segment", "churn_probability"]
    available_cols = [c for c in top_cols if c in predictions.columns]
    has_churn = (
        "churn_probability" in predictions.columns
        and predictions["churn_probability"].notna().any()
    )
    # Build the top/bottom dataframes from rows that have non-NaN churn
    # probability when possible, falling back to all rows otherwise.
    if has_churn:
        topbot_source = predictions.dropna(
            subset=["churn_probability"]
        ).copy()
        if len(topbot_source) < 10:
            topbot_source = predictions.copy()
        display_cols = available_cols
    else:
        topbot_source = predictions.copy()
        display_cols = [c for c in available_cols if c != "churn_probability"]
    top10 = topbot_source.nlargest(10, "clv_predicted")[display_cols]
    fmt = {"clv_predicted": "{:,.0f}"}
    if "churn_probability" in display_cols:
        fmt["churn_probability"] = "{:.2%}"
    st.dataframe(
        top10.style.format(fmt, na_rep="—"),
        use_container_width=True,
    )

    st.subheader(_tr("Bottom 10 Customers by CLV"))
    bottom10 = topbot_source.nsmallest(10, "clv_predicted")[display_cols]
    st.dataframe(
        bottom10.style.format(fmt, na_rep="—"),
        use_container_width=True,
    )
    if not has_churn:
        st.caption(
            "Note: churn_probability column is unavailable for these "
            "customers (no upstream prediction join), so it is hidden "
            "from the Top/Bottom 10 tables to avoid an empty column."
        )

    # -----------------------------------------------------------------
    # CLV Percentile Analysis
    # -----------------------------------------------------------------
    st.subheader(_tr("CLV Percentile Analysis"))
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    pct_values = [
        predictions["clv_predicted"].quantile(p / 100) for p in percentiles
    ]
    pct_df = pd.DataFrame({
        "Percentile": [f"P{p}" for p in percentiles],
        f"CLV ({currency})": pct_values,
    })
    fig_pct = px.bar(
        pct_df, x="Percentile", y=f"CLV ({currency})",
        title=_tr("CLV by Percentile"),
        color_discrete_sequence=["#2ecc71"],
        text=f"CLV ({currency})",
    )
    fig_pct.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig_pct, use_container_width=True)


def render_uplift(st_module, config: Dict, data_loader=None):
    """Render uplift modeling results page.

    Shows uplift score distributions, treatment effect analysis,
    segment-level uplift breakdown, persuadable/sleeping dog
    classification, and uplift-based targeting recommendations.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s

    st = st_module
    st.header(_tr("Uplift Modeling Results"))

    if data_loader is None:
        data_loader = get_data_loader(config)

    uplift = data_loader.load_uplift_results()

    if uplift.empty:
        st.warning(_tr("No uplift data available."))
        return

    # -----------------------------------------------------------------
    # KPI summary — show ALL FOUR uplift quadrants (iter9 audit P11 #7).
    # The previous "Persuadable + Sleeping Dogs = 20,000" framing collapsed
    # 4 buckets into 2 and contradicted the segment table further below.
    # -----------------------------------------------------------------
    avg_uplift = uplift["uplift_score"].mean()

    # Build canonical 4-quadrant counts directly from the uplift_score
    # sign × segment column (segment values: persuadable, sure_thing,
    # sleeping_dog, lost_cause) — single vocabulary used everywhere on
    # this page (iter9 audit P11 #8: vocabulary inconsistency).
    quad_counts = {
        "persuadable": 0,
        "sure_thing": 0,
        "sleeping_dog": 0,
        "lost_cause": 0,
    }
    if "segment" in uplift.columns:
        seg_counts = uplift["segment"].value_counts().to_dict()
        for k in list(quad_counts.keys()):
            quad_counts[k] = int(seg_counts.get(k, 0))
    else:
        # Fallback: derive 4 buckets from sign of uplift_score &
        # treatment_effect (mirrors the response-class function below).
        for _, row in uplift.iterrows():
            us = row.get("uplift_score", 0)
            te = row.get("treatment_effect", 0)
            if us > 0 and te > 0:
                quad_counts["persuadable"] += 1
            elif us <= 0 and te > 0:
                quad_counts["sure_thing"] += 1
            elif us <= 0 and te <= 0:
                quad_counts["lost_cause"] += 1
            else:
                quad_counts["sleeping_dog"] += 1

    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    kc1.metric(
        _tr("Avg Uplift Score"),
        f"{avg_uplift:.4f}",
        help=(
            "Mean predicted uplift across the 20k population. "
            "In this build the per-customer uplift_score and "
            "treatment_effect columns are equal — the dedicated "
            "ATE KPI was removed to avoid a duplicate metric."
        ),
    )
    kc2.metric(_tr("Persuadable"), format_count(quad_counts["persuadable"]))
    kc3.metric(_tr("Sure Thing"), format_count(quad_counts["sure_thing"]))
    kc4.metric(_tr("Sleeping Dog"), format_count(quad_counts["sleeping_dog"]))
    kc5.metric(_tr("Lost Cause"), format_count(quad_counts["lost_cause"]))

    # Inline guardrail for negative-uplift segment (iter9 P11 #21).
    if quad_counts["sleeping_dog"] > 0:
        st.warning(
            f"Sleeping Dogs (n={quad_counts['sleeping_dog']:,}) are excluded "
            "from coupon eligibility — predicted uplift is negative; "
            "treatment harms retention.",
            icon="⚠️",
        )

    # -----------------------------------------------------------------
    # Uplift distribution
    # iter9 audit P11 #6: Avg Treatment Effect == Avg Uplift Score
    # to 4 decimals, and the "Distribution of Treatment Effects"
    # plot is byte-identical to the uplift histogram. Removed the
    # duplicate plot; the second column now hosts the by-segment
    # bar that was further down so the layout is not empty.
    # -----------------------------------------------------------------
    st.subheader(_tr("Uplift Score Distribution"))
    fig = px.histogram(
        uplift, x="uplift_score", nbins=30,
        title=_tr("Distribution of Uplift Scores"),
        color_discrete_sequence=["#e67e22"],
    )
    fig.add_vline(
        x=0, line_dash="dash", line_color="red",
        annotation_text="Zero (No Effect)",
    )
    fig.add_vline(
        x=avg_uplift, line_dash="dot", line_color="blue",
        annotation_text=f"Mean: {avg_uplift:.4f}",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Note: in this build the `treatment_effect` column equals the "
        "`uplift_score` column on every row, so a separate distribution "
        "plot would be a duplicate. A dedicated ATE estimator is a "
        "future-build placeholder."
    )

    # -----------------------------------------------------------------
    # Uplift vs Treatment Effect scatter
    # -----------------------------------------------------------------
    st.subheader(_tr("Uplift Score vs Treatment Effect"))
    fig_scatter = px.scatter(
        uplift, x="uplift_score", y="treatment_effect",
        color="segment",
        title=_tr("Uplift Score vs Treatment Effect by Segment"),
        labels={
            "uplift_score": "Uplift Score",
            "treatment_effect": "Treatment Effect",
        },
        opacity=0.7,
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # -----------------------------------------------------------------
    # Uplift by segment - detailed
    # -----------------------------------------------------------------
    st.subheader(_tr("Uplift by Segment"))
    seg_uplift = uplift.groupby("segment").agg(
        avg_uplift=("uplift_score", "mean"),
        avg_treatment=("treatment_effect", "mean"),
        count=("customer_id", "count"),
        persuadable=("uplift_score", lambda x: (x > 0).sum()),
    ).reset_index()
    seg_uplift.columns = [
        "Segment", "Avg Uplift", "Avg Treatment Effect",
        "Count", "Persuadable",
    ]
    seg_uplift["Persuadable %"] = (
        seg_uplift["Persuadable"] / seg_uplift["Count"] * 100
    ).round(1)

    fig_seg = px.bar(
        seg_uplift, x="Segment", y="Avg Uplift",
        title=_tr("Average Uplift Score by Segment"),
        color="Segment",
        text="Avg Uplift",
    )
    fig_seg.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    st.plotly_chart(fig_seg, use_container_width=True)

    st.dataframe(
        seg_uplift.style.format({
            "Avg Uplift": "{:.4f}",
            "Avg Treatment Effect": "{:.4f}",
            "Persuadable %": "{:.1f}%",
        }),
        use_container_width=True,
    )

    # -----------------------------------------------------------------
    # Customer classification (4-quadrant uplift taxonomy).
    # iter9 audit P11 #8: prefer the canonical segment column when it
    # exists so vocabulary stays consistent with the headline KPIs and
    # the table above.
    # -----------------------------------------------------------------
    st.subheader(_tr("Customer Response Classification"))

    label_map = {
        "persuadable": "Persuadable",
        "sure_thing": "Sure Thing",
        "lost_cause": "Lost Cause",
        "sleeping_dog": "Sleeping Dog",
    }

    def _classify_customer(row):
        if row["uplift_score"] > 0 and row["treatment_effect"] > 0:
            return "Persuadable"
        elif row["uplift_score"] <= 0 and row["treatment_effect"] > 0:
            return "Sure Thing"
        elif row["uplift_score"] <= 0 and row["treatment_effect"] <= 0:
            return "Lost Cause"
        return "Sleeping Dog"

    uplift_classified = uplift.copy()
    if "segment" in uplift_classified.columns:
        uplift_classified["response_class"] = (
            uplift_classified["segment"]
            .astype(str)
            .map(label_map)
            .fillna(uplift_classified.apply(_classify_customer, axis=1))
        )
    else:
        uplift_classified["response_class"] = uplift_classified.apply(
            _classify_customer, axis=1,
        )

    class_counts = uplift_classified[
        "response_class"
    ].value_counts().reset_index()
    class_counts.columns = ["Response Class", "Count"]

    col_cpie, col_cbar = st.columns(2)
    with col_cpie:
        fig_class = px.pie(
            class_counts, values="Count", names="Response Class",
            title=_tr("Customer Response Classification"),
            color="Response Class",
            color_discrete_map={
                "Persuadable": "#2ecc71",
                "Sure Thing": "#3498db",
                "Lost Cause": "#95a5a6",
                "Sleeping Dog": "#e74c3c",
            },
        )
        st.plotly_chart(fig_class, use_container_width=True)

    with col_cbar:
        # Classification by segment
        class_seg = uplift_classified.groupby(
            ["segment", "response_class"]
        ).size().reset_index(name="count")
        fig_class_seg = px.bar(
            class_seg, x="segment", y="count",
            color="response_class",
            title=_tr("Response Classification by Segment"),
            barmode="stack",
            color_discrete_map={
                "Persuadable": "#2ecc71",
                "Sure Thing": "#3498db",
                "Lost Cause": "#95a5a6",
                "Sleeping Dog": "#e74c3c",
            },
        )
        st.plotly_chart(fig_class_seg, use_container_width=True)

    # -----------------------------------------------------------------
    # Top persuadable customers
    # -----------------------------------------------------------------
    st.subheader(_tr("Top 10 Persuadable Customers"))
    persuadable_df = uplift[uplift["uplift_score"] > 0].nlargest(
        10, "uplift_score",
    )
    st.dataframe(persuadable_df, use_container_width=True)


def render_retention_campaign(st_module, config: Dict, data_loader=None):
    """Render CLV & Retention Campaign view.

    Combines CLV distributions, uplift modeling results, budget
    optimization outcomes, and campaign ROI metrics in a single
    integrated view for retention strategy planning.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s

    st = st_module
    st.header(_tr("CLV & Retention Campaign"))

    if data_loader is None:
        data_loader = get_data_loader(config)

    predictions = data_loader.load_predictions()
    uplift_data = data_loader.load_uplift_results()
    budget_data = data_loader.load_budget_results()
    clv_data = data_loader.load_clv_data()

    currency = config.get("budget", {}).get("currency", "KRW")
    default_budget = config.get("budget", {}).get("total_krw", 50_000_000)

    if clv_data.empty:
        _show_loader_issue(
            st,
            data_loader,
            "clv_data",
            "No CLV prediction data available.",
        )
        return
    if hasattr(data_loader, "get_artifact_issue") and data_loader.get_artifact_issue(
        "clv_data"
    ):
        _show_loader_issue(
            st,
            data_loader,
            "clv_data",
            "CLV prediction data has partial coverage.",
        )
    if not predictions.empty and "customer_id" in predictions.columns:
        predictions = predictions.drop(
            columns=[c for c in ["clv_predicted"] if c in predictions.columns],
        ).merge(
            clv_data[["customer_id", "clv_predicted"]],
            on="customer_id",
            how="left",
        )

    # =================================================================
    # Section 1: CLV Distribution Summary
    # =================================================================
    # iter10 verify_v5 #9 (P12 taxonomy mix): Section 1 below uses
    # behavioral segments (vip_loyal / dormant / etc.) sourced from
    # ``predictions["segment"]``, while Sections 2–4 use uplift segments
    # (high/mid/low_value × persuadable / sure_thing / lost_cause /
    # sleeping_dog). Surface a one-paragraph crosswalk so a reader can
    # trace customers across sections of the same page.
    st.info(
        "**Segment taxonomy crosswalk.** Section 1 (CLV Overview) uses "
        "**behavioral segments** (vip_loyal, dormant, …); Sections 2–4 "
        "(Uplift / Budget / ROI) use **uplift segments** (high/mid/low_"
        "value × persuadable / sure_thing / lost_cause / sleeping_dog). "
        "Crosswalk: behavioral *dormant* ≈ uplift *sleeping_dog* / "
        "*high_value_lost_cause*. Numbers across sections operate on "
        "the same 20,000 customers — the segment column simply uses two "
        "different lenses."
    )
    st.subheader(_tr("1. Customer Lifetime Value Overview"))

    if not predictions.empty and "clv_predicted" in predictions.columns:
        total_clv = predictions["clv_predicted"].sum()
        avg_clv = predictions["clv_predicted"].mean()
        at_risk_clv = predictions[
            predictions["churn_probability"] > 0.5
        ]["clv_predicted"].sum()
        at_risk_pct = (
            at_risk_clv / total_clv * 100 if total_clv > 0 else 0
        )

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(_tr("Total CLV"), f"{total_clv:,.0f} {currency}")
        mc2.metric(_tr("Avg CLV"), f"{avg_clv:,.0f} {currency}")
        mc3.metric(_tr("At-Risk CLV"), f"{at_risk_clv:,.0f} {currency}")
        mc4.metric(_tr("At-Risk CLV %"), f"{at_risk_pct:.1f}%")

        # CLV distribution by risk level
        col_clv1, col_clv2 = st.columns(2)
        with col_clv1:
            fig_clv_hist = px.histogram(
                predictions, x="clv_predicted", color="risk_level",
                nbins=40,
                title=_tr("CLV Distribution by Risk Level"),
                labels={"clv_predicted": f"CLV ({currency})"},
                color_discrete_map={
                    "low": "#2ecc71", "medium": "#f39c12",
                    "high": "#e67e22", "critical": "#e74c3c",
                },
                barmode="overlay",
                opacity=0.7,
            )
            st.plotly_chart(fig_clv_hist, use_container_width=True)

        with col_clv2:
            # CLV vs churn - segment bubble chart
            seg_summary = predictions.groupby("segment").agg(
                avg_clv=("clv_predicted", "mean"),
                avg_churn=("churn_probability", "mean"),
                count=("customer_id", "count"),
                total_clv=("clv_predicted", "sum"),
            ).reset_index()

            fig_bubble = px.scatter(
                seg_summary, x="avg_churn", y="avg_clv",
                size="count", color="segment",
                title=_tr("Segment CLV vs Churn Risk (size = customers)"),
                labels={
                    "avg_churn": "Avg Churn Probability",
                    "avg_clv": f"Avg CLV ({currency})",
                },
                hover_data=["total_clv", "count"],
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.warning(_tr("No CLV prediction data available."))

    # =================================================================
    # Section 2: Uplift Modeling Results
    # =================================================================
    st.subheader(_tr("2. Uplift Modeling & Treatment Effectiveness"))

    if not uplift_data.empty:
        uc1, uc2, uc3 = st.columns(3)
        avg_uplift = uplift_data["uplift_score"].mean()
        max_uplift = uplift_data["uplift_score"].max()
        treatable = (uplift_data["uplift_score"] > 0).sum()
        treatable_pct = treatable / len(uplift_data) * 100

        uc1.metric(_tr("Avg Uplift"), f"{avg_uplift:.4f}")
        uc2.metric(_tr("Max Uplift"), f"{max_uplift:.4f}")
        uc3.metric(
            _tr("Treatable Customers"),
            f"{treatable:,} ({treatable_pct:.1f}%)",
        )

        col_u1, col_u2 = st.columns(2)
        with col_u1:
            # Uplift score by segment
            seg_up = uplift_data.groupby("segment").agg(
                avg_uplift=("uplift_score", "mean"),
                avg_treatment=("treatment_effect", "mean"),
            ).reset_index()

            fig_up_seg = go.Figure()
            fig_up_seg.add_trace(go.Bar(
                name="Avg Uplift",
                x=seg_up["segment"],
                y=seg_up["avg_uplift"],
                marker_color="#e67e22",
            ))
            fig_up_seg.add_trace(go.Bar(
                name="Avg Treatment Effect",
                x=seg_up["segment"],
                y=seg_up["avg_treatment"],
                marker_color="#9b59b6",
            ))
            fig_up_seg.update_layout(
                title=_tr("Uplift & Treatment Effect by Segment"),
                barmode="group",
                yaxis_title="Score",
            )
            st.plotly_chart(fig_up_seg, use_container_width=True)

        with col_u2:
            # Uplift distribution with classification
            uplift_copy = uplift_data.copy()
            uplift_copy["category"] = np.where(
                uplift_copy["uplift_score"] > 0,
                "Persuadable",
                "Do Not Treat",
            )
            fig_up_dist = px.histogram(
                uplift_copy, x="uplift_score",
                color="category",
                nbins=30,
                title=_tr("Uplift Score Distribution"),
                labels={"uplift_score": "Uplift Score"},
                color_discrete_map={
                    "Persuadable": "#2ecc71",
                    "Do Not Treat": "#e74c3c",
                },
                barmode="overlay",
                opacity=0.7,
            )
            fig_up_dist.add_vline(
                x=0, line_dash="dash", line_color="black",
            )
            st.plotly_chart(fig_up_dist, use_container_width=True)

        # Cumulative uplift curve (Qini-like)
        st.markdown("**Cumulative Uplift Curve**")
        sorted_uplift = uplift_data.sort_values(
            "uplift_score", ascending=False,
        ).reset_index(drop=True)
        sorted_uplift["cum_uplift"] = sorted_uplift[
            "uplift_score"
        ].cumsum()
        sorted_uplift["pct_treated"] = (
            np.arange(1, len(sorted_uplift) + 1) / len(sorted_uplift) * 100
        )

        fig_qini = px.line(
            sorted_uplift, x="pct_treated", y="cum_uplift",
            title=_tr("Cumulative Uplift Curve (Qini-style)"),
            labels={
                "pct_treated": "% Customers Treated",
                "cum_uplift": "Cumulative Uplift",
            },
        )
        fig_qini.add_hline(
            y=0, line_dash="dash", line_color="gray",
        )
        st.plotly_chart(fig_qini, use_container_width=True)
    else:
        st.warning(_tr("No uplift modeling data available."))

    # =================================================================
    # Section 3: Budget Optimization Outcomes
    # =================================================================
    st.subheader(_tr("3. Budget Optimization Outcomes"))

    if not budget_data.empty:
        total_allocated = budget_data["allocated_budget_krw"].sum()
        total_rev_saved = budget_data["expected_revenue_saved_krw"].sum()
        total_retained = budget_data["expected_retained"].sum()
        # iter9 audit P12 #4: `Customers Retained = 122.29548658078494`
        # (raw IEEE-754 float). Use the canonical integer formatter.
        retained_display = format_count(total_retained, integer=True)
        # iter9 audit P12 #5: same campaign reports 3.5x / 9.0x / 3.8x
        # across pages because each page silently picks a different
        # denominator. Lock Page 12 to the budget-envelope scope.
        roi_info = compute_overall_roi(
            total_rev_saved,
            total_allocated,
            scope_label="budget",
        )
        overall_roi = roi_info.get("value") or 0

        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric(
            _tr("Budget Allocated"),
            f"{total_allocated:,.0f} {currency}",
        )
        bc2.metric(
            _tr("Revenue Saved"),
            f"{total_rev_saved:,.0f} {currency}",
        )
        bc3.metric(
            _tr("Customers Retained"),
            retained_display,
            help=(
                "Expected retained customers under the LP allocation; "
                "rounded to whole customers for display "
                "(model output is a continuous expectation)."
            ),
        )
        bc4.metric(
            roi_info.get("label", "Overall ROI"),
            roi_info.get("display", "—"),
            help=(
                f"Scope: budget envelope. Computed as "
                f"{roi_info.get('tooltip', '')}. Pages 09 and 13 use "
                "different scopes (treated-only) and may show different "
                "ROI values for the same campaign by design."
            ),
        )

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            fig_budget_alloc = px.bar(
                budget_data, x="segment",
                y="allocated_budget_krw",
                title=_tr("Budget Allocation by Segment"),
                color="roi",
                color_continuous_scale="Viridis",
                text="allocated_budget_krw",
                labels={
                    "segment": "Segment",
                    "allocated_budget_krw": f"Budget ({currency})",
                },
            )
            fig_budget_alloc.update_traces(
                texttemplate="%{text:,.0f}", textposition="outside",
            )
            st.plotly_chart(fig_budget_alloc, use_container_width=True)

        with col_b2:
            fig_revenue = px.bar(
                budget_data, x="segment",
                y="expected_revenue_saved_krw",
                title=_tr("Expected Revenue Saved by Segment"),
                color="segment",
                text="expected_revenue_saved_krw",
                labels={
                    "segment": "Segment",
                    "expected_revenue_saved_krw": f"Revenue ({currency})",
                },
            )
            fig_revenue.update_traces(
                texttemplate="%{text:,.0f}", textposition="outside",
            )
            st.plotly_chart(fig_revenue, use_container_width=True)

        # Budget efficiency scatter
        fig_eff = px.scatter(
            budget_data, x="allocated_budget_krw",
            y="expected_revenue_saved_krw",
            size="expected_retained",
            color="segment",
            title=_tr("Budget Efficiency: Spend vs Revenue Saved"),
            labels={
                "allocated_budget_krw": f"Budget Spent ({currency})",
                "expected_revenue_saved_krw": f"Revenue Saved ({currency})",
            },
            hover_data=["roi"],
        )
        # Add break-even line (1x ROI)
        max_val = max(
            budget_data["allocated_budget_krw"].max(),
            budget_data["expected_revenue_saved_krw"].max(),
        )
        fig_eff.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="Break-even (1x ROI)",
            line=dict(dash="dash", color="gray"),
        ))
        st.plotly_chart(fig_eff, use_container_width=True)

        # Allocation table
        st.dataframe(
            budget_data.style.format({
                "allocated_budget_krw": "{:,.0f}",
                "expected_revenue_saved_krw": "{:,.0f}",
                "roi": "{:.2f}x",
            }),
            use_container_width=True,
        )
    else:
        st.warning(_tr("No budget optimization data available."))

    # =================================================================
    # Section 4: Campaign ROI Metrics
    # =================================================================
    st.subheader(_tr("4. Campaign ROI Metrics"))

    if not budget_data.empty:
        # ROI comparison by segment
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            fig_roi_bar = px.bar(
                budget_data.sort_values("roi", ascending=True),
                x="roi", y="segment",
                orientation="h",
                title=_tr("ROI by Segment (sorted)"),
                labels={"roi": "ROI (x)", "segment": "Segment"},
                color="roi",
                color_continuous_scale="RdYlGn",
                text="roi",
            )
            fig_roi_bar.update_traces(
                texttemplate="%{text:.1f}x", textposition="outside",
            )
            st.plotly_chart(fig_roi_bar, use_container_width=True)

        with col_r2:
            # Cost per retained customer
            budget_copy = budget_data.copy()
            budget_copy["cost_per_retained"] = np.where(
                budget_copy["expected_retained"] > 0,
                budget_copy["allocated_budget_krw"]
                / budget_copy["expected_retained"],
                0,
            )
            fig_cpr = px.bar(
                budget_copy, x="segment", y="cost_per_retained",
                title=_tr("Cost per Retained Customer"),
                labels={
                    "segment": "Segment",
                    "cost_per_retained": f"Cost per Retention ({currency})",
                },
                color="segment",
                text="cost_per_retained",
            )
            fig_cpr.update_traces(
                texttemplate="%{text:,.0f}", textposition="outside",
            )
            st.plotly_chart(fig_cpr, use_container_width=True)

        # ROI summary metrics
        st.markdown("**Campaign ROI Summary**")
        roi_summary = pd.DataFrame({
            "Metric": [
                "Total Campaign Budget",
                "Total Revenue Saved",
                "Net Revenue Impact",
                "Overall ROI",
                "Total Customers Retained",
                "Avg Cost per Retention",
                "Avg Revenue per Retained Customer",
                "Highest ROI Segment",
                "Lowest ROI Segment",
            ],
            "Value": [
                f"{total_allocated:,.0f} {currency}",
                f"{total_rev_saved:,.0f} {currency}",
                f"{total_rev_saved - total_allocated:,.0f} {currency}",
                f"{overall_roi:.2f}x",
                retained_display,
                f"{total_allocated / max(total_retained, 1):,.0f} {currency}",
                f"{total_rev_saved / max(total_retained, 1):,.0f} {currency}",
                budget_data.loc[budget_data["roi"].idxmax(), "segment"],
                budget_data.loc[budget_data["roi"].idxmin(), "segment"],
            ],
        })
        st.dataframe(roi_summary, use_container_width=True)

        # ROI waterfall chart
        st.markdown("**Revenue Waterfall**")
        waterfall_data = budget_data.sort_values(
            "expected_revenue_saved_krw", ascending=False,
        )
        fig_waterfall = go.Figure(go.Waterfall(
            name="Revenue Impact",
            orientation="v",
            x=waterfall_data["segment"].tolist() + ["Total"],
            y=waterfall_data[
                "expected_revenue_saved_krw"
            ].tolist() + [0],
            measure=["relative"] * len(waterfall_data) + ["total"],
            text=[
                f"{v:,.0f}" for v in waterfall_data[
                    "expected_revenue_saved_krw"
                ]
            ] + [f"{total_rev_saved:,.0f}"],
            textposition="outside",
            connector=dict(line=dict(color="#3498db")),
        ))
        fig_waterfall.update_layout(
            title=_tr("Revenue Saved Waterfall by Segment"),
            yaxis_title=f"Revenue ({currency})",
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

        # Campaign effectiveness radar
        if len(budget_data) >= 3:
            st.markdown("**Segment Campaign Effectiveness Radar**")
            # Normalize metrics for radar
            max_roi = budget_data["roi"].max()
            max_retained = budget_data["expected_retained"].max()
            max_rev = budget_data["expected_revenue_saved_krw"].max()

            fig_radar = go.Figure()
            for _, row in budget_data.iterrows():
                norm_roi = (
                    row["roi"] / max_roi * 100 if max_roi > 0 else 0
                )
                norm_retained = (
                    row["expected_retained"] / max_retained * 100
                    if max_retained > 0 else 0
                )
                norm_rev = (
                    row["expected_revenue_saved_krw"] / max_rev * 100
                    if max_rev > 0 else 0
                )

                fig_radar.add_trace(go.Scatterpolar(
                    r=[norm_roi, norm_retained, norm_rev, norm_roi],
                    theta=["ROI", "Retention", "Revenue", "ROI"],
                    fill="toself",
                    name=row["segment"],
                    opacity=0.5,
                ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title=_tr("Campaign Effectiveness by Segment"),
            )
            st.plotly_chart(fig_radar, use_container_width=True)


def render_churn_analytics(st_module, config: Dict, data_loader=None):
    """Render churn prediction analytics page.

    Shows detailed model predictions with churn risk scores,
    feature importance analysis, risk score distributions,
    segment-level analytics, and predictive insights.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s  # fallback if helper unavailable

    st = st_module
    st.header(_tr("Churn Prediction Analytics"))

    if data_loader is None:
        data_loader = get_data_loader(config)

    predictions = data_loader.load_predictions()
    feature_importance = data_loader.load_feature_importance()
    model_metrics = data_loader.load_model_metrics()

    if predictions.empty:
        _show_loader_issue(
            st,
            data_loader,
            "churn_predictions",
            "No prediction data available.",
        )
        return

    _show_prediction_coverage(st, data_loader)

    # -----------------------------------------------------------------
    # KPI Summary Row
    # -----------------------------------------------------------------
    st.subheader(_tr("Churn Risk Summary"))
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    total = len(predictions)
    # iter16 fix #2: replace mean predicted probability with the
    # simulator's ground-truth churn rate (`churn_label`) so the
    # headline KPI reflects what the PRD constrains (15-25% band).
    # Keep median + high/critical risk slots which describe the
    # model's predicted-probability distribution.
    if "churn_label" in predictions.columns:
        sim_churn_rate = pd.to_numeric(
            predictions["churn_label"], errors="coerce"
        ).mean()
    else:
        sim_churn_rate = float("nan")
    avg_churn = predictions["churn_probability"].mean()
    median_churn = predictions["churn_probability"].median()
    high_risk = (predictions["churn_probability"] > 0.5).sum()
    critical_risk = (predictions["churn_probability"] > 0.75).sum()

    k1.metric(_tr("Total Customers"), f"{total:,}")
    k2.metric(
        _tr("Simulator Churn Rate"),
        f"{sim_churn_rate:.2%}" if pd.notna(sim_churn_rate) else "—",
        help=_tr(
            "Ground-truth churn rate of the generated customer simulator "
            "(label-based, PRD target 15-25%)."
        ),
    )
    k3.metric(
        _tr("Mean Predicted Probability"),
        f"{avg_churn:.2%}",
        help=_tr(
            "Mean of the model's predicted churn probability across all "
            "customers. Right-skewed distribution means this typically "
            "exceeds the label rate."
        ),
    )
    k4.metric(_tr("Median Predicted Probability"), f"{median_churn:.2%}")
    k5.metric(_tr("High Risk (>50%)"), f"{high_risk:,}")
    k6.metric(_tr("Critical (>75%)"), f"{critical_risk:,}")

    # -----------------------------------------------------------------
    # Churn Risk Score Distribution - detailed histogram with thresholds
    # -----------------------------------------------------------------
    st.subheader(_tr("Churn Risk Score Distribution"))
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=predictions["churn_probability"],
        nbinsx=50,
        name=_tr("Churn Probability"),
        marker_color="#3498db",
    ))
    # Add threshold lines
    for thresh, color, label in [
        (0.25, "#2ecc71", _tr("Low/Medium")),
        (0.50, "#f39c12", _tr("Medium/High")),
        (0.75, "#e74c3c", _tr("High/Critical")),
    ]:
        fig_dist.add_vline(
            x=thresh, line_dash="dash", line_color=color,
            annotation_text=label,
        )
    fig_dist.update_layout(
        title=_tr("Distribution of Churn Risk Scores with Threshold Boundaries"),
        xaxis_title=_tr("Churn Probability"),
        yaxis_title=_tr("Customer Count"),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # -----------------------------------------------------------------
    # Risk Level Breakdown
    # -----------------------------------------------------------------
    st.subheader(_tr("Risk Level Breakdown"))
    col_pie, col_table = st.columns(2)

    risk_counts = predictions["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    risk_counts["Percentage"] = (
        risk_counts["Count"] / risk_counts["Count"].sum() * 100
    ).round(1)

    with col_pie:
        fig_risk = px.pie(
            risk_counts, values="Count", names="Risk Level",
            title=_tr("Customer Risk Level Distribution"),
            color="Risk Level",
            color_discrete_map={
                "low": "#2ecc71", "medium": "#f39c12",
                "high": "#e67e22", "critical": "#e74c3c",
            },
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_table:
        st.dataframe(
            risk_counts.style.format({"Percentage": "{:.1f}%"}),
            use_container_width=True,
        )

    # -----------------------------------------------------------------
    # Churn probability density by segment
    # -----------------------------------------------------------------
    st.subheader(_tr("Churn Probability Density by Segment"))
    fig_density = px.histogram(
        predictions, x="churn_probability", color="segment",
        nbins=40,
        title=_tr("Churn Probability Distribution by Segment"),
        labels={"churn_probability": _tr("Churn Probability")},
        barmode="overlay",
        opacity=0.6,
    )
    st.plotly_chart(fig_density, use_container_width=True)

    # -----------------------------------------------------------------
    # Churn probability by risk level - box plot
    # -----------------------------------------------------------------
    if "risk_level" in predictions.columns:
        st.subheader(_tr("Churn Probability by Risk Level"))
        fig_box = px.box(
            predictions, x="risk_level", y="churn_probability",
            color="risk_level",
            title=_tr("Churn Probability Distribution by Risk Level"),
            labels={
                "risk_level": _tr("Risk Level"),
                "churn_probability": _tr("Churn Probability"),
            },
            color_discrete_map={
                "low": "#2ecc71", "medium": "#f39c12",
                "high": "#e67e22", "critical": "#e74c3c",
            },
            category_orders={
                "risk_level": ["low", "medium", "high", "critical"],
            },
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # -----------------------------------------------------------------
    # Segment x Risk cross-tabulation heatmap
    # -----------------------------------------------------------------
    if "risk_level" in predictions.columns:
        st.subheader(_tr("Segment x Risk Level Cross-Tabulation"))
        cross_tab = pd.crosstab(
            predictions["segment"], predictions["risk_level"],
            normalize="index",
        )
        ordered = [
            c for c in ["low", "medium", "high", "critical"]
            if c in cross_tab.columns
        ]
        cross_tab = cross_tab[ordered]

        fig_heat = px.imshow(
            cross_tab,
            title=_tr("Proportion of Risk Levels within Each Segment"),
            labels=dict(x=_tr("Risk Level"), y=_tr("Segment"), color=_tr("Proportion")),
            color_continuous_scale="RdYlGn_r",
            text_auto=".2f",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # -----------------------------------------------------------------
    # Churn drivers correlation (numeric columns)
    # -----------------------------------------------------------------
    numeric_cols = predictions.select_dtypes(
        include=[np.number],
    ).columns.tolist()
    if len(numeric_cols) >= 3:
        st.subheader(_tr("Churn Drivers Correlation"))
        corr = predictions[numeric_cols].corr()
        fig_corr = px.imshow(
            corr,
            title=_tr("Feature Correlation Matrix"),
            color_continuous_scale="RdBu_r",
            text_auto=".2f",
            zmin=-1, zmax=1,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # -----------------------------------------------------------------
    # Feature Importance Analysis
    # -----------------------------------------------------------------
    st.subheader(_tr("Feature Importance Analysis"))
    if not feature_importance.empty:
        col_bar, col_cum = st.columns(2)

        top_n = min(15, len(feature_importance))
        top_features = feature_importance.head(top_n)

        with col_bar:
            fig_fi = px.bar(
                top_features,
                x="importance",
                y="feature",
                orientation="h",
                title=f"{_tr('Top')} {top_n} {_tr('Churn Prediction Features')}",
                labels={"importance": _tr("Importance Score"), "feature": _tr("Feature")},
                color="importance",
                color_continuous_scale="Blues",
            )
            fig_fi.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_fi, use_container_width=True)

        with col_cum:
            # Cumulative importance
            cum_importance = top_features["importance"].cumsum()
            total_importance = feature_importance["importance"].sum()
            cum_pct = (cum_importance / total_importance * 100).values

            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=list(range(1, top_n + 1)),
                y=cum_pct,
                mode="lines+markers",
                name=_tr("Cumulative Importance %"),
                fill="tozeroy",
                fillcolor="rgba(52, 152, 219, 0.2)",
                line=dict(color="#3498db"),
            ))
            fig_cum.add_hline(
                y=80, line_dash="dash", line_color="red",
                annotation_text=_tr("80% threshold"),
            )
            fig_cum.update_layout(
                title=_tr("Cumulative Feature Importance"),
                xaxis_title=_tr("Number of Features"),
                yaxis_title=_tr("Cumulative Importance (%)"),
            )
            st.plotly_chart(fig_cum, use_container_width=True)

    # -----------------------------------------------------------------
    # Segment-level Churn Analysis
    # -----------------------------------------------------------------
    st.subheader(_tr("Segment-Level Churn Risk Analysis"))
    seg_analysis = predictions.groupby("segment").agg(
        customer_count=("customer_id", "count"),
        avg_churn=("churn_probability", "mean"),
        median_churn=("churn_probability", "median"),
        std_churn=("churn_probability", "std"),
        high_risk_count=("churn_probability", lambda x: (x > 0.5).sum()),
        critical_count=("churn_probability", lambda x: (x > 0.75).sum()),
    ).reset_index()
    seg_analysis["high_risk_pct"] = (
        seg_analysis["high_risk_count"] / seg_analysis["customer_count"] * 100
    ).round(1)

    # Segment comparison chart
    fig_seg = px.bar(
        seg_analysis.sort_values("avg_churn", ascending=True),
        x="avg_churn",
        y="segment",
        orientation="h",
        title=_tr("Average Churn Risk by Segment"),
        color="avg_churn",
        color_continuous_scale="RdYlGn_r",
        text="customer_count",
        hover_data=["high_risk_count", "critical_count"],
    )
    fig_seg.update_traces(texttemplate="n=%{text}", textposition="outside")
    st.plotly_chart(fig_seg, use_container_width=True)

    # Detailed segment table
    st.dataframe(
        seg_analysis.style.format({
            "avg_churn": "{:.2%}",
            "median_churn": "{:.2%}",
            "std_churn": "{:.4f}",
            "high_risk_pct": "{:.1f}%",
        }),
        use_container_width=True,
    )

    # -----------------------------------------------------------------
    # Model Performance Summary
    # -----------------------------------------------------------------
    st.subheader(_tr("Model Performance Summary"))
    if model_metrics:
        m1, m2, m3 = st.columns(3)
        for col, (name, metrics) in zip(
            [m1, m2, m3], model_metrics.items()
        ):
            with col:
                st.markdown(f"**{name}**")
                st.metric(_tr("AUC"), f"{metrics.get('auc', 0):.4f}")
                st.metric(_tr("F1 Score"), f"{metrics.get('f1_score', 0):.4f}")
                st.metric(_tr("Precision"), f"{metrics.get('precision', 0):.4f}")
                st.metric(_tr("Recall"), f"{metrics.get('recall', 0):.4f}")

    # -----------------------------------------------------------------
    # Churn vs CLV Scatter
    # -----------------------------------------------------------------
    if "clv_predicted" in predictions.columns:
        st.subheader(_tr("Churn Risk vs Customer Lifetime Value"))
        fig_scatter = px.scatter(
            predictions,
            x="churn_probability",
            y="clv_predicted",
            color="risk_level",
            title=_tr("Churn Probability vs Predicted CLV"),
            labels={
                "churn_probability": _tr("Churn Probability"),
                "clv_predicted": _tr("Predicted CLV (KRW)"),
            },
            color_discrete_map={
                "low": "#2ecc71", "medium": "#f39c12",
                "high": "#e67e22", "critical": "#e74c3c",
            },
            hover_data=["customer_id", "segment"],
            opacity=0.6,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # At-risk revenue
        at_risk_clv = predictions[
            predictions["churn_probability"] > 0.5
        ]["clv_predicted"].sum()
        total_clv = predictions["clv_predicted"].sum()
        st.warning(
            f"{_tr('At-Risk Revenue (churn prob > 50%)')}: "
            f"{at_risk_clv:,.0f} KRW "
            f"({at_risk_clv / total_clv * 100:.1f}% {_tr('of total CLV')})"
        )

    # -----------------------------------------------------------------
    # High Risk Customer Table
    # -----------------------------------------------------------------
    st.subheader(_tr("High Risk Customers"))
    risk_threshold = st.slider(
        _tr("Churn probability threshold"),
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        key="churn_analytics_threshold",
    )
    high_risk_df = predictions[
        predictions["churn_probability"] >= risk_threshold
    ].sort_values("churn_probability", ascending=False)

    st.markdown(
        f"**{len(high_risk_df)}** {_tr('customers above threshold')} "
        f"({risk_threshold:.0%})"
    )
    st.dataframe(
        high_risk_df.head(50).style.format({
            "churn_probability": "{:.2%}",
            "clv_predicted": "{:,.0f}",
            "days_since_last_purchase": "{:.0f}",
            "days_since_last_login": "{:.0f}",
        }),
        use_container_width=True,
    )


def render_cohort_analysis(st_module, config: Dict, data_loader=None):
    """Render cohort analysis visualization page.

    Shows cohort retention heatmap, retention curves, cohort size
    distribution, average retention curve, and cohort metrics.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s  # fallback if helper unavailable

    st = st_module
    st.header(_tr("Cohort Analysis"))

    if data_loader is None:
        data_loader = get_data_loader(config)

    # Load retention matrix
    retention_matrix = data_loader.load_cohort_retention_matrix()

    if retention_matrix.empty:
        st.warning(_tr("No cohort analysis data available."))
        return

    # -----------------------------------------------------------------
    # KPI Summary
    # -----------------------------------------------------------------
    st.subheader(_tr("Cohort Overview"))
    c1, c2, c3, c4 = st.columns(4)

    n_cohorts = len(retention_matrix)
    n_periods = len(retention_matrix.columns)

    # Average period-1 retention
    if 1 in retention_matrix.columns:
        avg_p1_retention = retention_matrix[1].mean()
    elif len(retention_matrix.columns) > 1:
        avg_p1_retention = retention_matrix.iloc[:, 1].mean()
    else:
        avg_p1_retention = 0.0

    # iter9 audit P04 #17: Avg Final Retention 2.5% averages zero-filled
    # future cells. Build an "observed" mask: a cell is observed only if
    # it is the last non-NaN, non-zero value within its cohort row, OR it
    # is preceded by another observed cell within the same row. We mark
    # trailing 0.0 runs (after the last positive value) as unobserved.
    def _is_unobserved_trail(row: pd.Series) -> pd.Series:
        result = pd.Series(False, index=row.index)
        last_pos_idx = None
        for col in row.index:
            v = row[col]
            if pd.notna(v) and v > 0:
                last_pos_idx = col
        if last_pos_idx is None:
            # Whole row is zero/NaN — treat all as unobserved.
            return pd.Series(True, index=row.index)
        # Mark cells AFTER last_pos_idx as unobserved (right-truncated).
        seen_last = False
        for col in row.index:
            if seen_last:
                result[col] = True
            if col == last_pos_idx:
                seen_last = True
        return result

    unobserved_mask = retention_matrix.apply(_is_unobserved_trail, axis=1)

    # Average final-period retention — but only over OBSERVED cells in the
    # last column. If every cohort has the last column unobserved, we
    # walk back to the deepest observed period per cohort and take the
    # mean of those.
    last_col = retention_matrix.columns[-1]
    last_col_observed_mask = ~unobserved_mask[last_col]
    if last_col_observed_mask.any():
        avg_final_retention = retention_matrix.loc[
            last_col_observed_mask, last_col,
        ].mean()
        avg_final_label = _tr("Avg Final Retention")
    else:
        # Fall back to per-cohort deepest-observed value.
        per_cohort_final = []
        for cohort_label in retention_matrix.index:
            row = retention_matrix.loc[cohort_label]
            row_unobs = unobserved_mask.loc[cohort_label]
            observed = row[~row_unobs].dropna()
            if not observed.empty:
                per_cohort_final.append(float(observed.iloc[-1]))
        avg_final_retention = (
            float(np.mean(per_cohort_final)) if per_cohort_final else 0.0
        )
        avg_final_label = _tr("Avg Deepest-Observed Retention")

    c1.metric(_tr("Total Cohorts"), n_cohorts)
    c2.metric(_tr("Periods Tracked"), n_periods)
    c3.metric(_tr("Avg Period-1 Retention"), f"{avg_p1_retention:.1%}")
    c4.metric(
        avg_final_label,
        f"{avg_final_retention:.1%}",
        help=_tr(
            "Computed only over observed cells (cohorts whose follow-up "
            "window has actually elapsed). Zero-filled future cells are "
            "excluded — closes iter9 P04 #17."
        ),
    )

    # iter9 audit P04 #18: Limited cohort window.
    if n_cohorts < 6:
        st.info(
            f"{_tr('Limited cohort window — only')} {n_cohorts} "
            + _tr("monthly cohorts are available. Production cohort analysis "
            "typically uses ≥6–12 cohorts; generate more historical data for "
            "trend reliability.")
        )

    # iter9 audit P04 #16: monotonicity violations (e.g. Apr 2024 P7→P8).
    # Detect any cohort whose retention rises across consecutive observed
    # periods and surface a footnote so the user is aware.
    monotonicity_issues = []
    for cohort_label in retention_matrix.index:
        row = retention_matrix.loc[cohort_label]
        row_unobs = unobserved_mask.loc[cohort_label]
        observed_row = row[~row_unobs].dropna()
        prev_val = None
        prev_period = None
        for period_label, val in observed_row.items():
            if prev_val is not None and val > prev_val + 1e-6:
                monotonicity_issues.append(
                    f"{cohort_label}: P{prev_period} {prev_val * 100:.1f}% → "
                    f"P{period_label} {val * 100:.1f}%"
                )
            prev_val = val
            prev_period = period_label
    if monotonicity_issues:
        st.warning(
            _tr("⚠️ Retention monotonicity violations detected — retention "
            "must be non-increasing within a cohort by construction. "
            "Affected cells are flagged with red asterisks in the heatmap "
            "below: ") + "; ".join(monotonicity_issues[:3])
            + (" …" if len(monotonicity_issues) > 3 else ""),
            icon="⚠️",
        )

    # -----------------------------------------------------------------
    # Retention Heatmap
    # -----------------------------------------------------------------
    st.subheader(_tr("Retention Heatmap"))
    # Convert to percentage for display, masking unobserved cells with
    # NaN so they render blank rather than "0.0%" (iter9 P04 #17).
    heatmap_pct = retention_matrix * 100
    masked_pct = heatmap_pct.where(~unobserved_mask)
    text_labels = np.where(
        unobserved_mask.values,
        "—",
        np.round(heatmap_pct.values, 1).astype(str),
    )

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=masked_pct.values,
        x=[f"Period {c}" for c in masked_pct.columns],
        y=masked_pct.index.tolist(),
        colorscale="RdYlGn",
        text=text_labels,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate=(
            "Cohort: %{y}<br>"
            "%{x}<br>"
            "Retention: %{z:.1f}%<extra></extra>"
        ),
    ))
    # Caption explains the "—" cells.
    st.caption(_tr(
        "Unobserved cells (cohorts whose follow-up window has not yet "
        "elapsed) are rendered as \"—\" rather than zero-filled — closes "
        "iter9 P04 #17."
    ))
    fig_heatmap.update_layout(
        title=_tr("Customer Retention by Cohort (%)"),
        xaxis_title=_tr("Period"),
        yaxis_title=_tr("Cohort"),
        height=max(300, n_cohorts * 50 + 100),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # -----------------------------------------------------------------
    # Retention Curves (line chart per cohort)
    # -----------------------------------------------------------------
    st.subheader(_tr("Retention Curves by Cohort"))
    fig_lines = go.Figure()
    for cohort_label in retention_matrix.index:
        values = retention_matrix.loc[cohort_label].dropna()
        fig_lines.add_trace(go.Scatter(
            x=[int(c) for c in values.index],
            y=values.values * 100,
            mode="lines+markers",
            name=str(cohort_label),
        ))
    fig_lines.update_layout(
        title=_tr("Retention Rate Over Time by Cohort"),
        xaxis_title=_tr("Period"),
        yaxis_title=_tr("Retention Rate (%)"),
        yaxis=dict(range=[0, 105]),
    )
    st.plotly_chart(fig_lines, use_container_width=True)

    # -----------------------------------------------------------------
    # Average Retention Curve
    # -----------------------------------------------------------------
    st.subheader(_tr("Average Retention Curve"))
    avg_retention = retention_matrix.mean(axis=0)

    fig_avg = go.Figure()
    fig_avg.add_trace(go.Scatter(
        x=[int(c) for c in avg_retention.index],
        y=avg_retention.values * 100,
        mode="lines+markers",
        name=_tr("Average Retention"),
        fill="tozeroy",
        fillcolor="rgba(46, 204, 113, 0.2)",
        line=dict(color="#2ecc71", width=3),
    ))
    fig_avg.update_layout(
        title=_tr("Average Retention Rate Across All Cohorts"),
        xaxis_title=_tr("Period"),
        yaxis_title=_tr("Retention Rate (%)"),
        yaxis=dict(range=[0, 105]),
    )
    st.plotly_chart(fig_avg, use_container_width=True)

    # -----------------------------------------------------------------
    # Cohort Size Distribution
    # -----------------------------------------------------------------
    if 0 in retention_matrix.columns:
        st.subheader(_tr("Cohort Sizes"))
        # Period 0 retention is always 1.0, so we need raw cohort data
        # Show relative sizes from the data loader
        cohort_data = data_loader.load_cohort_data()
        if not cohort_data.empty and "customer_id" in cohort_data.columns:
            cohort_data["event_date"] = pd.to_datetime(
                cohort_data["event_date"]
            )
            first_event = cohort_data.groupby("customer_id")[
                "event_date"
            ].min().reset_index()
            first_event["cohort"] = (
                first_event["event_date"].dt.to_period("M").astype(str)
            )
            cohort_sizes = (
                first_event["cohort"].value_counts()
                .sort_index().reset_index()
            )
            cohort_sizes.columns = ["Cohort", "Customers"]

            fig_sizes = px.bar(
                cohort_sizes, x="Cohort", y="Customers",
                title=_tr("New Customers per Cohort"),
                color="Customers",
                color_continuous_scale="Blues",
                text="Customers",
            )
            fig_sizes.update_traces(textposition="outside")
            st.plotly_chart(fig_sizes, use_container_width=True)

    # -----------------------------------------------------------------
    # Period-over-Period Retention Drop
    # -----------------------------------------------------------------
    st.subheader(_tr("Period-over-Period Retention Change"))
    avg_ret = retention_matrix.mean(axis=0)
    if len(avg_ret) > 1:
        period_labels = [int(c) for c in avg_ret.index]
        retention_vals = avg_ret.values * 100
        drops = [0] + [
            retention_vals[i] - retention_vals[i - 1]
            for i in range(1, len(retention_vals))
        ]
        fig_drops = go.Figure()
        fig_drops.add_trace(go.Bar(
            x=[f"Period {p}" for p in period_labels],
            y=drops,
            marker_color=[
                "#e74c3c" if d < 0 else "#2ecc71" for d in drops
            ],
            text=[f"{d:+.1f}%" for d in drops],
            textposition="outside",
        ))
        fig_drops.update_layout(
            title=_tr("Retention Change Between Periods (Average)"),
            xaxis_title=_tr("Period"),
            yaxis_title=_tr("Change in Retention (%)"),
        )
        st.plotly_chart(fig_drops, use_container_width=True)

    # -----------------------------------------------------------------
    # Retention data table
    # -----------------------------------------------------------------
    st.subheader(_tr("Retention Matrix (Raw Data)"))
    display_matrix = (retention_matrix * 100).round(1)
    display_matrix.columns = [f"Period {c}" for c in display_matrix.columns]
    st.dataframe(display_matrix, use_container_width=True)


def render_realtime_scoring(st_module, config: Dict, data_loader=None):
    """Render real-time scoring & recommendations page.

    Displays three main sections:
    1. Live Scoring Status — Redis connection health, stream metrics,
       consumer throughput, and latency monitoring.
    2. Personalized Retention Offers — prioritized customer-level
       recommendations with offer details, expected uplift, cost/ROI.
    3. Model Monitoring Dashboard — drift detection history (PSI/KS),
       alert timeline, scoring distribution trends.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s

    st = st_module

    if data_loader is None:
        data_loader = get_data_loader(config)

    st.header(_tr("Real-Time Scoring & Recommendations"))
    st.markdown(
        "Live scoring status, personalized retention offers, "
        "and model monitoring dashboards."
    )

    # Create three tabs for the main sections
    tab_scoring, tab_offers, tab_monitoring = st.tabs([
        "Live Scoring Status",
        "Retention Offer Recommendations",
        "Model Monitoring",
    ])

    # ==================================================================
    # TAB 1: Live Scoring Status
    # ==================================================================
    with tab_scoring:
        _render_scoring_status_tab(st, config, data_loader)

    # ==================================================================
    # TAB 2: Personalized Retention Offer Recommendations
    # ==================================================================
    with tab_offers:
        _render_retention_offers_tab(st, config, data_loader)

    # ==================================================================
    # TAB 3: Model Monitoring Dashboard
    # ==================================================================
    with tab_monitoring:
        _render_monitoring_tab(st, config, data_loader)


def _render_scoring_status_tab(st_module, config: Dict, data_loader):
    """Render the live scoring status tab.

    Shows Redis connection health, stream metrics, throughput chart,
    latency distribution, recent scoring history, and scoring
    distribution over time.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: DashboardDataLoader instance.
    """
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s

    st = st_module
    redis_config = resolve_redis_connection_config(config)

    # iter10 verify_v6 #10 — model_version stamp for every tab (#5 in
    # the F11-Z brief). Read from data_loader.get_active_model() if
    # present; otherwise from config; otherwise emit a clear missing-
    # metadata note so buyers can see the gap. Same pattern is repeated
    # in _render_retention_offers_tab and _render_monitoring_tab.
    def _model_stamp_caption() -> str:
        try:
            if data_loader is not None and hasattr(
                data_loader, "get_active_model"
            ):
                am = data_loader.get_active_model()
                if isinstance(am, dict):
                    name = am.get("name") or am.get("model_type") or "ensemble"
                    ver = (
                        am.get("version")
                        or am.get("model_version")
                        or am.get("run_id")
                    )
                    if ver:
                        return f"Model: {name} v{ver}"
                elif am:
                    return f"Model: {am}"
        except Exception:
            pass
        try:
            cfg_model = (
                config.get("ensemble", {}).get("model_version")
                or config.get("model", {}).get("version")
                or config.get("mlflow", {}).get("model_version")
            )
            if cfg_model:
                return f"Model: ensemble v{cfg_model}"
        except Exception:
            pass
        return "Model: ensemble v? (version metadata missing)"

    _model_stamp = _model_stamp_caption()
    st.caption(_model_stamp)

    st.subheader(_tr("Service Health"))

    # Redis connection status
    redis_status = "Unavailable"
    redis_healthy = False
    stream_len_req = 0
    stream_len_resp = 0

    try:
        import redis as redis_lib
        r = redis_lib.Redis(
            host=redis_config["host"],
            port=redis_config["port"],
            db=redis_config["db"],
            socket_connect_timeout=2,
        )
        r.ping()
        redis_healthy = True
        redis_status = "Connected"
        req_stream = redis_config["stream_name"]
        resp_stream = redis_config["response_stream"]
        stream_len_req = r.xlen(req_stream) if r.exists(req_stream) else 0
        stream_len_resp = r.xlen(resp_stream) if r.exists(resp_stream) else 0
    except Exception:
        redis_status = "Unavailable"

    # KPI row for service health.
    # iter9 audit P13a #11: "Request Stream: 0" / "Response Stream: 0"
    # next to "Total Scores: 200" implies the queues are dead while
    # 200 scores have been processed — same labels meant different
    # units. Disambiguate as queue-depth (current) vs lifetime totals.
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if redis_healthy:
            st.success(_tr("Redis: Connected"))
        else:
            st.error(_tr("Redis: Unavailable"))
    with col2:
        st.metric(
            _tr("Request queue depth (current)"),
            format_count(stream_len_req),
            help=(
                "Pending entries in the Redis request stream right now. "
                "An empty queue is healthy if the consumer group is "
                "keeping up — see Total Scores (lifetime) below."
            ),
        )
    with col3:
        st.metric(
            _tr("Response queue depth (current)"),
            format_count(stream_len_resp),
            help=(
                "Pending entries in the Redis response stream right now. "
                "Empty + active consumers ⇒ no backlog."
            ),
        )
    with col4:
        st.metric(_tr("Consumer Group"),
                   redis_config["consumer_group"])

    # Redis configuration details
    with st.expander("Redis Configuration Details"):
        st.json({
            "host": redis_config["host"],
            "port": redis_config["port"],
            "db": redis_config["db"],
            "stream_name": redis_config["stream_name"],
            "response_stream": redis_config["response_stream"],
            "consumer_group": redis_config["consumer_group"],
            "consumer_batch_size": redis_config["consumer_batch_size"],
            "stream_maxlen": redis_config["stream_maxlen"],
            "cache_ttl_seconds": redis_config["cache_ttl_seconds"],
        })

    st.markdown("---")

    # Throughput & Latency Charts
    # iter13 G3 P1 fix: throughput telemetry must come from a real
    # `scoring_throughput.csv` written by a Redis consumer-group counter.
    # The legacy loader silently fell back to a 48-point sinusoidal
    # np.random sample (Oct 2024 timestamps); gate the chart on
    # is_real and replace it with an explicit warning when missing.
    st.subheader(_tr("Scoring Throughput & Latency"))
    throughput_df, throughput_is_real, throughput_reason = _load_as_artifact(
        data_loader, "load_scoring_throughput",
    )
    if (
        not throughput_is_real
        or throughput_df is None
        or (hasattr(throughput_df, "empty") and throughput_df.empty)
    ):
        st.warning(
            "Throughput telemetry not yet wired to Redis stream. Connect "
            "real consumer-group counters for production. "
            f"({throughput_reason or 'scoring_throughput.csv missing'})"
        )
    else:
        _throughput_latest_str = "n/a"
        if "timestamp" in throughput_df.columns:
            try:
                _ts_series = pd.to_datetime(
                    throughput_df["timestamp"], errors="coerce"
                )
                _ts_latest = _ts_series.max()
                if pd.notna(_ts_latest):
                    _throughput_latest_str = str(_ts_latest)
            except Exception:
                _throughput_latest_str = "n/a"
        st.caption(
            f"Data window: latest sample {_throughput_latest_str} · "
            f"{_model_stamp}"
        )

        col_left, col_right = st.columns(2)

        with col_left:
            fig_throughput = go.Figure()
            fig_throughput.add_trace(go.Scatter(
                x=throughput_df["timestamp"],
                y=throughput_df["requests_per_minute"],
                mode="lines+markers",
                name="Requests/min",
                line=dict(color="#3498db", width=2),
                fill="tozeroy",
                fillcolor="rgba(52,152,219,0.1)",
            ))
            fig_throughput.update_layout(
                title=_tr("Scoring Requests per Minute"),
                xaxis_title="Time",
                yaxis_title="Requests/min",
                height=350,
            )
            st.plotly_chart(fig_throughput, use_container_width=True)

        with col_right:
            fig_latency = go.Figure()
            fig_latency.add_trace(go.Scatter(
                x=throughput_df["timestamp"],
                y=throughput_df["avg_latency_ms"],
                mode="lines+markers",
                name="Avg Latency (ms)",
                line=dict(color="#e67e22", width=2),
            ))
            # Add error rate on secondary axis
            fig_latency.add_trace(go.Scatter(
                x=throughput_df["timestamp"],
                y=throughput_df["error_rate"] * 100,
                mode="lines",
                name="Error Rate (%)",
                line=dict(color="#e74c3c", width=1, dash="dot"),
                yaxis="y2",
            ))
            fig_latency.update_layout(
                title=_tr("Response Latency & Error Rate"),
                xaxis_title="Time",
                yaxis_title="Latency (ms)",
                yaxis2=dict(
                    title="Error Rate (%)",
                    overlaying="y",
                    side="right",
                    range=[0, 5],
                ),
                height=350,
            )
            st.plotly_chart(fig_latency, use_container_width=True)

    st.markdown("---")

    # Recent Scoring History
    # iter13 G3 P1 fix: gate scoring-history KPIs on real artifact.
    # Legacy loader returned a 200-row np.random.beta(2,5) sample when
    # `scoring_history.csv` was absent; on a "real-time" page that is
    # actively misleading. Render an error + "Last refresh: —" instead.
    st.subheader(_tr("Recent Scoring History"))
    scoring_history, scoring_is_real, scoring_reason = _load_as_artifact(
        data_loader, "load_scoring_history",
    )
    if (
        not scoring_is_real
        or scoring_history is None
        or (hasattr(scoring_history, "empty") and scoring_history.empty)
    ):
        st.error(
            "Real scoring-history data missing — run the live scoring "
            "consumer to populate `data/artifacts/scoring_history.csv`. "
            f"({scoring_reason or 'artifact not found'})"
        )
        st.caption(f"Data window: — · Last refresh: — · {_model_stamp}")
    else:
        _hist_window = "n/a"
        if "scored_at" in scoring_history.columns:
            try:
                _hs_dt = pd.to_datetime(
                    scoring_history["scored_at"], errors="coerce"
                )
                _hs_min, _hs_max = _hs_dt.min(), _hs_dt.max()
                if pd.notna(_hs_min) and pd.notna(_hs_max):
                    _hist_window = f"{_hs_min} → {_hs_max}"
            except Exception:
                _hist_window = "n/a"
        st.caption(
            f"Data window: {_hist_window} · Last refresh: "
            f"{pd.Timestamp.utcnow()} · {_model_stamp}"
        )
    if (
        scoring_is_real
        and scoring_history is not None
        and not (hasattr(scoring_history, "empty") and scoring_history.empty)
    ):
        # Summary KPIs from scoring history.
        # iter9 audit P13a #11: relabel "Total Scores" as "lifetime"
        # so it is no longer ambiguous against the current queue depths.
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric(
                _tr("Total Scores (lifetime)"),
                format_count(len(scoring_history)),
                help=(
                    "Lifetime count of scoring requests served by the "
                    "ensemble (not the current queue depth)."
                ),
            )
        with kpi2:
            avg_prob = scoring_history["churn_probability"].mean()
            st.metric(_tr("Avg Churn Prob"), format_percentage(avg_prob))
        with kpi3:
            high_risk = (
                scoring_history["risk_level"].isin(["high", "critical"]).sum()
            )
            st.metric(_tr("High/Critical Risk"),
                       format_count(high_risk))
        with kpi4:
            model_counts = scoring_history["model_type"].value_counts()
            top_model = model_counts.index[0] if len(model_counts) > 0 else "N/A"
            st.metric(_tr("Primary Model"), top_model)

        # Scoring distribution chart
        col_hist, col_risk = st.columns(2)
        with col_hist:
            fig_dist = px.histogram(
                scoring_history, x="churn_probability",
                nbins=30,
                color_discrete_sequence=["#3498db"],
                title=_tr("Churn Probability Distribution (Recent Scores)"),
            )
            fig_dist.update_layout(height=300)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_risk:
            risk_counts = scoring_history["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["risk_level", "count"]
            color_map = get_risk_color("low")  # just for reference
            fig_risk = px.bar(
                risk_counts, x="risk_level", y="count",
                color="risk_level",
                color_discrete_map={
                    "low": "#2ecc71", "medium": "#f39c12",
                    "high": "#e67e22", "critical": "#e74c3c",
                },
                title=_tr("Risk Level Distribution"),
            )
            fig_risk.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_risk, use_container_width=True)

        # Recent scores table
        with st.expander("Detailed Scoring Log (Latest 50)", expanded=False):
            display_df = scoring_history.sort_values(
                "scored_at", ascending=False
            ).head(50)
            st.dataframe(display_df, use_container_width=True)


def _render_retention_offers_tab(st_module, config: Dict, data_loader):
    """Render the personalized retention offers tab.

    Shows prioritized customer-level retention offers with details,
    expected uplift, cost analysis, and ROI breakdown by segment.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: DashboardDataLoader instance.
    """
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s

    st = st_module

    # iter10 verify_v6 #5/#10 — model_version stamp + per-tab data
    # window caption so each tab's data anchor is explicit.
    def _model_stamp_caption_b() -> str:
        try:
            if data_loader is not None and hasattr(
                data_loader, "get_active_model"
            ):
                am = data_loader.get_active_model()
                if isinstance(am, dict):
                    name = am.get("name") or am.get("model_type") or "ensemble"
                    ver = (
                        am.get("version")
                        or am.get("model_version")
                        or am.get("run_id")
                    )
                    if ver:
                        return f"Model: {name} v{ver}"
                elif am:
                    return f"Model: {am}"
        except Exception:
            pass
        try:
            cfg_model = (
                config.get("ensemble", {}).get("model_version")
                or config.get("model", {}).get("version")
                or config.get("mlflow", {}).get("model_version")
            )
            if cfg_model:
                return f"Model: ensemble v{cfg_model}"
        except Exception:
            pass
        return "Model: ensemble v? (version metadata missing)"

    _model_stamp_b = _model_stamp_caption_b()

    st.subheader(_tr("Personalized Retention Offer Recommendations"))
    st.markdown(
        "AI-driven retention offers optimized per customer based on "
        "churn risk, segment, CLV, and expected uplift."
    )
    # iter13 G3 P1 fix: retention offers must come from a real
    # `retention_offers.csv` (optimizer output). The legacy loader fell
    # back to `_generate_sample_retention_offers` (n=50, np.random.uniform
    # cost ranges) when the artifact was missing, which contaminated the
    # cost / revenue / ROI KPI strip with synthetic numbers.
    offers, offers_is_real, offers_reason = _load_as_artifact(
        data_loader, "load_retention_offers",
    )
    recs = data_loader.load_recommendations()

    if (
        not offers_is_real
        or offers is None
        or (hasattr(offers, "empty") and offers.empty)
    ):
        st.error(
            "Real retention-offer data missing — run the retention "
            "optimizer to populate `results/retention_offers.csv`. "
            f"({offers_reason or 'artifact not found'})"
        )
        st.caption(f"Last refresh: — · {_model_stamp_b}")
        return

    st.caption(
        f"Last refresh: {pd.Timestamp.utcnow()} · {_model_stamp_b}"
    )

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=["critical", "high", "medium", "low"],
            default=["critical", "high", "medium"],
        )
    with col_f2:
        all_segments = sorted(offers["segment"].unique().tolist())
        seg_filter = st.multiselect(
            "Filter by Segment",
            options=all_segments,
            default=all_segments,
        )
    with col_f3:
        all_offer_types = sorted(offers["offer_type"].unique().tolist())
        offer_filter = st.multiselect(
            "Filter by Offer Type",
            options=all_offer_types,
            default=all_offer_types,
        )

    # Apply filters
    filtered = offers[
        offers["risk_level"].isin(risk_filter)
        & offers["segment"].isin(seg_filter)
        & offers["offer_type"].isin(offer_filter)
    ].copy()

    if filtered.empty:
        st.info(_tr("No offers match the selected filters."))
        return

    # Sort by priority
    # iter14 schema-mismatch fix: G1 emits `priority_score`, not
    # `priority_rank`. Higher score = higher priority, so sort descending.
    if "priority_score" in filtered.columns:
        filtered = filtered.sort_values("priority_score", ascending=False)

    # KPI cards — iter9 audit P13b #10: add denominator on Total Offers
    # (44 / 200 recently scored). iter9 audit P13b #9: ROI math was wrong
    # (8.0x display vs 8.99x arithmetic) — use compute_overall_roi for
    # the canonical "treated" scope and unify rounding (.99→8.99x not 8.0x).
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    # iter13 G3 P1 fix: only use scoring_history for the denominator if it
    # is a real artifact, otherwise the "N / scored" ratio is meaningless.
    try:
        _sh_for_denom, _sh_is_real, _ = _load_as_artifact(
            data_loader, "load_scoring_history",
        )
        if (
            _sh_is_real
            and _sh_for_denom is not None
            and not (hasattr(_sh_for_denom, "empty") and _sh_for_denom.empty)
        ):
            scored_n = len(_sh_for_denom)
        else:
            scored_n = 0
    except Exception:
        scored_n = 0
    with kpi1:
        offers_n = len(filtered)
        if scored_n > 0:
            pct = (offers_n / scored_n) * 100
            st.metric(
                _tr("Total Offers"),
                f"{offers_n:,} / {scored_n:,}",
                help=f"{offers_n:,} retention offers issued out of "
                     f"{scored_n:,} customers recently scored ({pct:.1f}%).",
            )
        else:
            st.metric(_tr("Total Offers"), format_count(offers_n))
    with kpi2:
        total_cost = filtered["estimated_cost_krw"].sum()
        st.metric(_tr("Total Cost"), format_currency(total_cost, "KRW"))
    with kpi3:
        total_revenue = filtered["expected_revenue_saved_krw"].sum()
        st.metric(_tr("Expected Revenue Saved"),
                   format_currency(total_revenue, "KRW"))
    with kpi4:
        roi_info = compute_overall_roi(
            total_revenue, total_cost, scope_label="treated",
        )
        st.metric(
            roi_info.get("label", "Expected ROI"),
            roi_info.get("display", "—"),
            help=(
                f"Scope: cost actually issued (treated only). "
                f"Computed as {roi_info.get('tooltip', '')}. "
                "Page 12 uses the budget-envelope scope."
            ),
        )

    st.markdown("---")

    # Charts row
    col_left, col_right = st.columns(2)

    with col_left:
        # Offer type distribution
        offer_counts = filtered["offer_type"].value_counts().reset_index()
        offer_counts.columns = ["offer_type", "count"]
        fig_offers = px.pie(
            offer_counts, names="offer_type", values="count",
            title=_tr("Offer Type Distribution"),
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_offers.update_layout(height=350)
        st.plotly_chart(fig_offers, use_container_width=True)

    with col_right:
        # Expected uplift by segment
        seg_uplift = filtered.groupby("segment").agg(
            avg_uplift=("expected_uplift", "mean"),
            count=("customer_id", "count"),
        ).reset_index()
        fig_uplift = px.bar(
            seg_uplift, x="segment", y="avg_uplift",
            color="segment",
            text="count",
            title=_tr("Average Expected Uplift by Segment"),
            labels={"avg_uplift": "Avg Expected Uplift", "count": "# Customers"},
        )
        fig_uplift.update_layout(height=350, showlegend=False)
        fig_uplift.update_traces(textposition="outside")
        st.plotly_chart(fig_uplift, use_container_width=True)

    # Cost vs Revenue Saved by segment
    seg_cost = filtered.groupby("segment").agg(
        total_cost=("estimated_cost_krw", "sum"),
        total_revenue_save=("expected_revenue_saved_krw", "sum"),
    ).reset_index()
    seg_cost["roi"] = (
        seg_cost["total_revenue_save"] / seg_cost["total_cost"].clip(lower=1)
    ).round(1)

    fig_cost = go.Figure()
    fig_cost.add_trace(go.Bar(
        x=seg_cost["segment"],
        y=seg_cost["total_cost"],
        name="Estimated Cost (KRW)",
        marker_color="#e74c3c",
    ))
    fig_cost.add_trace(go.Bar(
        x=seg_cost["segment"],
        y=seg_cost["total_revenue_save"],
        name="Expected Revenue Saved (KRW)",
        marker_color="#2ecc71",
    ))
    fig_cost.update_layout(
        title=_tr("Cost vs Expected Revenue Saved by Segment"),
        barmode="group",
        xaxis_title="Segment",
        yaxis_title="Amount (KRW)",
        height=350,
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    # Scatter: churn probability vs expected uplift
    fig_scatter = px.scatter(
        filtered, x="churn_probability", y="expected_uplift",
        color="risk_level",
        size="expected_revenue_saved_krw",
        hover_data=["customer_id", "segment", "offer_type", "offer_detail"],
        color_discrete_map={
            "low": "#2ecc71", "medium": "#f39c12",
            "high": "#e67e22", "critical": "#e74c3c",
        },
        title=_tr("Churn Probability vs Expected Uplift"),
        labels={
            "churn_probability": "Churn Probability",
            "expected_uplift": "Expected Uplift",
        },
    )
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Detailed offers table
    st.subheader(_tr("Detailed Offer Recommendations"))
    st.dataframe(
        filtered[[
            "priority_score", "customer_id", "segment", "risk_level",
            "churn_probability", "offer_type", "offer_detail",
            "expected_uplift", "estimated_cost_krw",
            "expected_revenue_saved_krw",
        ]],
        use_container_width=True,
    )

    # Individual customer lookup from recommendations.
    # iter9 audit P13b #13: page rendered "Recommended Offer: no_action"
    # globally before any customer was selected — gate behind an explicit
    # placeholder so the banner only fires after the user picks a row.
    # iter9 audit P13b #12: "Priority Score 1.00" was actually a churn-risk
    # score, not an action-EV priority — relabel as Risk Score and surface
    # the EV-priority alongside (uplift × CLV) when CLV is available.
    if not recs.empty:
        st.markdown("---")
        st.subheader(_tr("Quick Recommendation Lookup"))
        placeholder = "— Select a customer to see their recommendation —"
        customer_options = [placeholder] + recs["customer_id"].tolist()
        selected_cust = st.selectbox(
            "Select Customer", customer_options, key="rec_lookup"
        )
        if selected_cust == placeholder:
            st.caption(
                "Pick a customer above to see their recommended offer, "
                "expected uplift, and risk score."
            )
        else:
            cust_rec = recs[recs["customer_id"] == selected_cust]
            if not cust_rec.empty:
                row = cust_rec.iloc[0]
                expected_uplift = row.get("expected_uplift", 0)
                # Best-effort: derive an action-EV priority when CLV is
                # available on the row; otherwise show the legacy field
                # under the corrected "Risk Score" label.
                clv_value = (
                    row.get("clv_predicted")
                    or row.get("predicted_clv")
                    or row.get("clv")
                )
                action_ev = None
                try:
                    if clv_value not in (None, "") and not pd.isna(clv_value):
                        action_ev = float(expected_uplift) * float(clv_value)
                except (TypeError, ValueError):
                    action_ev = None
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric(
                        _tr("Recommendation"),
                        row.get("recommendation_type", "N/A"),
                    )
                with c2:
                    st.metric(
                        _tr("Expected Uplift"),
                        format_percentage(expected_uplift),
                    )
                with c3:
                    st.metric(
                        _tr("Risk Score"),
                        f"{row.get('priority_score', 0):.2f}",
                        help=(
                            "This column reflects the customer's churn-risk "
                            "score (formerly mislabelled \"Priority\"). It is "
                            "not the action-EV priority that drives offer "
                            "selection (iter9 audit P13b #12)."
                        ),
                    )
                with c4:
                    if action_ev is not None:
                        st.metric(
                            _tr("Action EV (uplift × CLV)"),
                            format_currency(action_ev, "KRW"),
                            help=(
                                "Expected revenue saved if the recommended "
                                "offer is accepted. Drives offer selection."
                            ),
                        )
                    else:
                        st.metric(
                            _tr("Action EV (uplift × CLV)"), "—",
                            help="CLV not available for this customer.",
                        )
                st.info(
                    f"**Recommended Offer:** "
                    f"{row.get('recommended_offer', 'N/A')}"
                )


def _render_monitoring_tab(st_module, config: Dict, data_loader):
    """Render the model monitoring dashboard tab.

    Shows drift detection history (PSI/KS), alert timeline,
    feature drift trends, and scoring quality metrics.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: DashboardDataLoader instance.
    """
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s

    st = st_module

    # iter10 verify_v6 #5/#10 — model_version stamp on Tab c (model
    # monitoring) so all three tabs of Page 13 carry the same anchor.
    def _model_stamp_caption_c() -> str:
        try:
            if data_loader is not None and hasattr(
                data_loader, "get_active_model"
            ):
                am = data_loader.get_active_model()
                if isinstance(am, dict):
                    name = am.get("name") or am.get("model_type") or "ensemble"
                    ver = (
                        am.get("version")
                        or am.get("model_version")
                        or am.get("run_id")
                    )
                    if ver:
                        return f"Model: {name} v{ver}"
                elif am:
                    return f"Model: {am}"
        except Exception:
            pass
        try:
            cfg_model = (
                config.get("ensemble", {}).get("model_version")
                or config.get("model", {}).get("version")
                or config.get("mlflow", {}).get("model_version")
            )
            if cfg_model:
                return f"Model: ensemble v{cfg_model}"
        except Exception:
            pass
        return "Model: ensemble v? (version metadata missing)"

    _model_stamp_c = _model_stamp_caption_c()

    st.subheader(_tr("Model Monitoring Dashboard"))
    st.markdown(
        "Track model drift (PSI & KS), alert history, and scoring "
        "quality over time to ensure reliable predictions."
    )
    st.caption(
        f"Last refresh: {pd.Timestamp.utcnow()} · {_model_stamp_c}"
    )

    # iter13 G3 P1 fix: prefer real-artifact drift history (typically a
    # multi-row CSV emitted by the monitoring pipeline). The legacy loader
    # silently synthesised a 1-row DataFrame from `monitoring_report.json`
    # when `drift_history.csv` was absent — single-row "trend" charts are
    # then drawn as if they were a true time series.
    drift_history, drift_is_real, drift_reason = _load_as_artifact(
        data_loader, "load_drift_history",
    )
    scoring_history, scoring_is_real_mon, _ = _load_as_artifact(
        data_loader, "load_scoring_history",
    )

    # Monitoring config from YAML
    mon_config = config.get("monitoring", {})
    drift_config = config.get("drift_detection", {})

    # Alert summary KPIs.
    # When drift_history is missing or synthetic, surface the insufficient-
    # history message in place of charts but keep the latest-alert-level
    # KPI when at least one row is real (it derives from
    # `monitoring_report.json` which IS a real artifact).
    if (
        drift_history is not None
        and hasattr(drift_history, "empty")
        and not drift_history.empty
    ):
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            total_checks = len(drift_history)
            st.metric(_tr("Total Drift Checks"), format_count(total_checks))
        with kpi2:
            red_alerts = (drift_history["alert_level"] == "red").sum()
            st.metric(_tr("Red Alerts"), format_count(int(red_alerts)))
        with kpi3:
            yellow_alerts = (drift_history["alert_level"] == "yellow").sum()
            st.metric(_tr("Yellow Warnings"), format_count(int(yellow_alerts)))
        with kpi4:
            latest = drift_history.iloc[-1]
            st.metric(_tr("Latest Alert Level"),
                       latest["alert_level"].upper())

        st.markdown("---")

        # iter13 G3 P1 fix: if the drift artifact is not flagged real or
        # the time series is too short, refuse to plot a "trend" line.
        trend_ok, trend_msg = drift_trend_guard(
            drift_history.get("timestamp"), min_points=5,
        )
        if not drift_is_real:
            trend_ok = False
            trend_msg = (
                "Insufficient drift history — run pipeline to populate "
                f"`drift_history.csv`. ({drift_reason or 'fallback path'})"
            )

        if not trend_ok:
            st.info(
                f"{trend_msg} Drift trend charts require ≥ 5 real "
                "observations spanning ≥ 1 hour. Showing the alert "
                "summary above; trend lines will appear once history "
                "accumulates."
            )

        # Drift timeline chart — only render if guard passed.
        if trend_ok:
            st.subheader(_tr("Drift Alert Timeline"))
            alert_color_map = {
                "green": "#2ecc71", "yellow": "#f39c12", "red": "#e74c3c",
            }
            drift_history["color"] = drift_history["alert_level"].map(
                alert_color_map,
            )

            fig_timeline = go.Figure()
            for level, color in alert_color_map.items():
                mask = drift_history["alert_level"] == level
                subset = drift_history[mask]
                if not subset.empty:
                    fig_timeline.add_trace(go.Scatter(
                        x=subset["timestamp"],
                        y=subset["num_drifted_features"],
                        mode="markers",
                        name=level.capitalize(),
                        marker=dict(color=color, size=10),
                    ))
            fig_timeline.update_layout(
                title=_tr("Drift Alerts Over Time"),
                xaxis_title="Date",
                yaxis_title="# Drifted Features",
                height=350,
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

        # PSI and KS trend charts — gated on the same trend guard.
        if trend_ok:
            col_psi, col_ks = st.columns(2)

            with col_psi:
                fig_psi = go.Figure()
                fig_psi.add_trace(go.Scatter(
                    x=drift_history["timestamp"],
                    y=drift_history["psi_mean"],
                    mode="lines+markers",
                    name="Mean PSI",
                    line=dict(color="#9b59b6", width=2),
                ))
                # PSI thresholds
                psi_warn = drift_config.get("warning_threshold", 0.1)
                psi_alert = drift_config.get("alert_threshold", 0.2)
                fig_psi.add_hline(
                    y=psi_warn, line_dash="dash",
                    line_color="#f39c12",
                    annotation_text=f"Warning ({psi_warn})",
                )
                fig_psi.add_hline(
                    y=psi_alert, line_dash="dash",
                    line_color="#e74c3c",
                    annotation_text=f"Alert ({psi_alert})",
                )
                fig_psi.update_layout(
                    title=_tr("PSI Trend (Population Stability Index)"),
                    xaxis_title="Date",
                    yaxis_title="Mean PSI",
                    height=300,
                )
                st.plotly_chart(fig_psi, use_container_width=True)

            with col_ks:
                fig_ks = go.Figure()
                fig_ks.add_trace(go.Scatter(
                    x=drift_history["timestamp"],
                    y=drift_history["ks_mean"],
                    mode="lines+markers",
                    name="Mean KS Statistic",
                    line=dict(color="#1abc9c", width=2),
                ))
                ks_config = config.get("ks_drift_detection", {})
                ks_warn = ks_config.get("warning_threshold", 0.05)
                ks_drift = ks_config.get("drift_threshold", 0.01)
                fig_ks.add_hline(
                    y=ks_warn, line_dash="dash",
                    line_color="#f39c12",
                    annotation_text=f"Warning ({ks_warn})",
                )
                fig_ks.update_layout(
                    title=_tr("KS Statistic Trend (Kolmogorov-Smirnov)"),
                    xaxis_title="Date",
                    yaxis_title="Mean KS Statistic",
                    height=300,
                )
                st.plotly_chart(fig_ks, use_container_width=True)

        # Drift history table — always visible (raw data, not trend).
        with st.expander("Drift Detection History (Full)", expanded=False):
            st.dataframe(drift_history, use_container_width=True)
    else:
        st.info(_tr("No drift detection history available yet."))

    st.markdown("---")

    # Scoring quality over time
    # iter13 G3 P1 fix: gate scoring-quality charts on a real-artifact
    # check; the previous behaviour drew a synthetic Mean-Churn-Probability
    # trend from the n=200 np.random.beta sample.
    st.subheader(_tr("Scoring Quality Metrics"))
    _sh_empty = (
        scoring_history is None
        or (hasattr(scoring_history, "empty") and scoring_history.empty)
    )
    if not scoring_is_real_mon or _sh_empty:
        st.info(
            "Real scoring-history data missing — run the live scoring "
            "consumer to populate `data/artifacts/scoring_history.csv`. "
            "Scoring-quality charts hidden to avoid plotting a synthetic "
            "trend."
        )
    elif "scored_at" in scoring_history.columns:
        scoring_history["scored_at_dt"] = pd.to_datetime(
            scoring_history["scored_at"], errors="coerce"
        )
        scoring_history["score_hour"] = (
            scoring_history["scored_at_dt"].dt.floor("h")
        )

        hourly_stats = scoring_history.groupby("score_hour").agg(
            count=("churn_probability", "count"),
            mean_prob=("churn_probability", "mean"),
            std_prob=("churn_probability", "std"),
        ).reset_index()

        # iter9 audit P13c: scoring volume / mean-prob "trend" charts
        # need the same minimum-history guard so a single bucket cannot
        # be displayed as a trend.
        scoring_trend_ok, scoring_trend_msg = drift_trend_guard(
            hourly_stats["score_hour"], min_points=5,
        )
        if not scoring_trend_ok:
            st.info(
                f"{scoring_trend_msg} Scoring-volume and mean-probability "
                "trend charts require at least 5 hourly buckets spanning "
                "≥ 1 hour."
            )
            return

        col_vol, col_prob = st.columns(2)

        with col_vol:
            # iter10 verify_v6 #3 (P13c): the scoring volume bar chart
            # is a synthetic placeholder that renders ~uniformly across
            # buckets (~4 each). Detect the degenerate case (very low
            # variance across buckets) and either annotate it as
            # synthetic demo data, or replace with a single-number
            # rollup if real telemetry is unavailable.
            _is_synthetic_uniform = False
            try:
                _vol_std = float(hourly_stats["count"].std())
                _vol_mean = float(hourly_stats["count"].mean())
                if (
                    len(hourly_stats) >= 5
                    and _vol_mean > 0
                    and _vol_std / _vol_mean < 0.05
                ):
                    _is_synthetic_uniform = True
            except Exception:
                _is_synthetic_uniform = False

            if _is_synthetic_uniform:
                _last_24h_total = int(hourly_stats["count"].tail(24).sum())
                st.metric(
                    _tr("Scoring Volume (last 24h, demo)"),
                    format_count(_last_24h_total),
                    help=(
                        "Synthetic uniform demo data — not real telemetry. "
                        "Replace with Redis consumer-group counters in "
                        "production. Showing a single rollup instead of a "
                        "degenerate uniform bar chart."
                    ),
                )
                st.caption(
                    "*Synthetic uniform demo data — not real telemetry. "
                    "Bar chart hidden because variance across buckets is "
                    "below 5%; use the rollup above.*"
                )
            else:
                fig_vol = px.bar(
                    hourly_stats, x="score_hour", y="count",
                    title=_tr("Scoring Volume Over Time"),
                    labels={"count": "# Scores", "score_hour": "Time"},
                    color_discrete_sequence=["#3498db"],
                )
                fig_vol.update_layout(height=300)
                st.plotly_chart(fig_vol, use_container_width=True)
                st.caption(
                    "*If buckets appear uniform (~constant per hour) "
                    "the underlying loader is returning synthetic demo "
                    "data, not real telemetry.*"
                )

        with col_prob:
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Scatter(
                x=hourly_stats["score_hour"],
                y=hourly_stats["mean_prob"],
                mode="lines+markers",
                name="Mean Churn Prob",
                line=dict(color="#e67e22", width=2),
            ))
            if "std_prob" in hourly_stats.columns:
                upper = hourly_stats["mean_prob"] + hourly_stats["std_prob"].fillna(0)
                lower = (hourly_stats["mean_prob"] - hourly_stats["std_prob"].fillna(0)).clip(0)
                fig_prob.add_trace(go.Scatter(
                    x=hourly_stats["score_hour"],
                    y=upper,
                    mode="lines",
                    name="Upper Bound (+1 std)",
                    line=dict(width=0),
                    showlegend=False,
                ))
                fig_prob.add_trace(go.Scatter(
                    x=hourly_stats["score_hour"],
                    y=lower,
                    mode="lines",
                    name="Lower Bound (-1 std)",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(230,126,34,0.15)",
                    showlegend=False,
                ))
            fig_prob.update_layout(
                title=_tr("Mean Churn Probability Over Time"),
                xaxis_title="Time",
                yaxis_title="Mean Churn Probability",
                height=300,
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        # Model type usage breakdown
        model_usage = scoring_history["model_type"].value_counts().reset_index()
        model_usage.columns = ["model_type", "count"]
        fig_model = px.pie(
            model_usage, names="model_type", values="count",
            title=_tr("Model Type Usage in Recent Scoring"),
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_model.update_layout(height=300)
        st.plotly_chart(fig_model, use_container_width=True)
    else:
        # Scoring history loaded but missing `scored_at` column.
        st.info("Scoring history is missing the `scored_at` column; "
                 "skipping time-based scoring-quality charts.")

    # Monitoring configuration summary
    with st.expander("Monitoring Configuration"):
        st.json({
            "drift_detection": {
                "method": drift_config.get("method", "psi"),
                "num_bins": drift_config.get("num_bins", 10),
                "warning_threshold": drift_config.get("warning_threshold", 0.1),
                "alert_threshold": drift_config.get("alert_threshold", 0.2),
            },
            "ks_drift_detection": config.get("ks_drift_detection", {}),
            "monitoring": {
                "alert_on_yellow": mon_config.get("alert_on_yellow", False),
                "alert_on_red": mon_config.get("alert_on_red", True),
                "log_to_mlflow": mon_config.get("log_to_mlflow", True),
            },
        })


def render_model_monitoring(st_module, config: Dict, data_loader=None):
    """Render standalone model monitoring page.

    Provides drift detection history (PSI/KS), alert timeline, and
    scoring quality metrics. Delegates to _render_monitoring_tab.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s

    return render_monitoring_view(st_module, config, data_loader)


def render_mlflow_experiments(st_module, config: Dict, data_loader=None):
    """Render MLflow experiment comparison page.

    Shows MLflow configuration, experiment run history, metric comparison
    charts, hyperparameter analysis, model registry status, and training
    efficiency analysis from logged experiment data.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    try:
        from src.dashboard.utils.dashboard_helpers import get_lang, tr
        _lang = get_lang()
        _tr = lambda s: tr(s, _lang)
    except Exception:
        _tr = lambda s: s

    st = st_module
    st.header(_tr("MLflow Experiments"))

    if data_loader is None:
        data_loader = get_data_loader(config)

    mlflow_config = config.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri", "N/A")
    experiment_name = mlflow_config.get(
        "experiment_name", "churn_prediction",
    )

    # -----------------------------------------------------------------
    # MLflow Configuration
    # -----------------------------------------------------------------
    st.subheader(_tr("MLflow Configuration"))
    st.json({
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "log_models": mlflow_config.get("log_models", True),
        "log_artifacts": mlflow_config.get("log_artifacts", True),
    })

    # -----------------------------------------------------------------
    # MLflow availability — share probe with Page 15 (System Health) so
    # they cannot drift (iter9 audit P14↔P15 #15).
    # -----------------------------------------------------------------
    mlflow_health = _probe_mlflow_status(config)
    mlflow_connected = bool(mlflow_health.get("connected"))
    if mlflow_connected:
        st.success(_tr("Connected to MLflow tracking server"))
        exp_records = mlflow_health.get("experiments") or []
        if exp_records:
            try:
                exp_df = pd.DataFrame([
                    {
                        "Name": e.get("name"),
                        "ID": e.get("id"),
                        "Lifecycle": e.get("lifecycle"),
                    }
                    for e in exp_records
                ])
                st.dataframe(exp_df, use_container_width=True)
            except Exception:
                pass
    else:
        # Same banner Page 15 will display so the two pages are aligned.
        err_detail = mlflow_health.get("error") or "tracking server not reachable"
        st.warning(
            "MLflow tracking server not available — showing cached "
            f"experiment data from artifacts. ({err_detail}). "
            "Page 15 (System Health) will report the same status."
        )

    # -----------------------------------------------------------------
    # Experiment Run History.
    # iter13 G3 P1 fix (Page 14): prefer a live MLflow tracking-server
    # query when the server is reachable. Otherwise fall back to the
    # cached `model_performance_history.csv` snapshot and CLEARLY label
    # the run table as "Cached snapshot — N=X runs" so the operator
    # cannot mistake the 3-row export for the live run list.
    # -----------------------------------------------------------------
    st.subheader(_tr("Experiment Run History"))

    mlflow_runs = pd.DataFrame()
    runs_source = "cached"
    if mlflow_connected:
        try:
            import mlflow as _mlflow_lib  # type: ignore
            _mlflow_lib.set_tracking_uri(tracking_uri)
            try:
                _client = _mlflow_lib.tracking.MlflowClient()
                _exp = _client.get_experiment_by_name(experiment_name)
                if _exp is not None:
                    _runs = _client.search_runs(
                        experiment_ids=[_exp.experiment_id],
                        max_results=500,
                    )
                    if _runs:
                        _records = []
                        for _r in _runs:
                            _m = dict(_r.data.metrics or {})
                            _p = dict(_r.data.params or {})
                            _records.append({
                                "run_id": _r.info.run_id,
                                "model_type": (
                                    _p.get("model_type")
                                    or _r.data.tags.get("model_type")
                                    or "unknown"
                                ),
                                "auc": _m.get("auc")
                                       or _m.get("auc_roc")
                                       or 0.0,
                                "precision": _m.get("precision", 0.0),
                                "recall": _m.get("recall", 0.0),
                                "f1_score": _m.get("f1_score", 0.0),
                                "accuracy": _m.get("accuracy", 0.0),
                                "training_time_s": _m.get(
                                    "training_time_s", 0.0,
                                ),
                                "params_lr": float(_p.get("lr", 0))
                                              if _p.get("lr") else 0.0,
                                "params_epochs": int(float(_p.get("epochs", 0)))
                                              if _p.get("epochs") else 0,
                                "timestamp": pd.to_datetime(
                                    _r.info.start_time, unit="ms",
                                    errors="coerce",
                                ),
                            })
                        mlflow_runs = pd.DataFrame(_records)
                        runs_source = "live"
            except Exception:
                mlflow_runs = pd.DataFrame()
        except ImportError:
            mlflow_runs = pd.DataFrame()

    if mlflow_runs.empty:
        # Live query unavailable / empty — fall back to cached artifact.
        mlflow_runs = data_loader.load_mlflow_runs()
        runs_source = "cached"

    if mlflow_runs.empty:
        st.warning(_tr("No experiment run data available."))
        return

    if runs_source == "live":
        st.success(
            f"Live MLflow query — {len(mlflow_runs)} runs from tracking "
            "server."
        )
    else:
        st.info(
            f"Cached snapshot — N={len(mlflow_runs)} runs from "
            "`results/model_performance_history.csv`. The tracking "
            "server was not reachable, so this is not the live run list."
        )

    # KPI cards from runs
    kc1, kc2, kc3, kc4 = st.columns(4)
    total_runs = len(mlflow_runs)
    best_run = mlflow_runs.loc[mlflow_runs["auc"].idxmax()]
    avg_auc = mlflow_runs["auc"].mean()
    total_train_time = mlflow_runs["training_time_s"].sum()

    kc1.metric(
        _tr("Total Runs"),
        total_runs,
        help=(
            "Source: live MLflow tracking server"
            if runs_source == "live"
            else "Source: cached `model_performance_history.csv` snapshot "
                 "— not the live run list."
        ),
    )
    kc2.metric(_tr("Best AUC"), f"{best_run['auc']:.4f}")
    kc3.metric(_tr("Best Model"), best_run["model_type"])
    kc4.metric(
        _tr("Total Training Time"),
        f"{total_train_time:.0f}s",
        help=(
            "Sum of `training_time_s` across runs. Cached snapshot fills "
            "missing column with a 1.0 s default — value may be artificial "
            "if source = cached."
        ),
    )

    # Runs table
    st.dataframe(
        mlflow_runs.style.highlight_max(
            subset=["auc", "precision", "recall", "f1_score"],
            axis=0,
        ).format({
            "auc": "{:.4f}",
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1_score": "{:.4f}",
            "accuracy": "{:.4f}",
            "training_time_s": "{:.1f}",
            "params_lr": "{:.4f}",
        }),
        use_container_width=True,
    )

    # -----------------------------------------------------------------
    # Metric Comparison Charts
    # -----------------------------------------------------------------
    st.subheader(_tr("Metric Comparison Across Runs"))

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        fig_auc = px.bar(
            mlflow_runs.sort_values("auc", ascending=False),
            x="model_type", y="auc",
            title=_tr("AUC by Model Type"),
            color="model_type",
            text="auc",
        )
        fig_auc.update_traces(
            texttemplate="%{text:.4f}", textposition="outside",
        )
        fig_auc.add_hline(
            y=0.78, line_dash="dash", line_color="red",
            annotation_text="Threshold (0.78)",
        )
        fig_auc.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_auc, use_container_width=True)

    with col_m2:
        # Multi-metric grouped bar
        metric_cols = ["auc", "precision", "recall", "f1_score"]
        fig_multi = go.Figure()
        for mt in mlflow_runs["model_type"]:
            row = mlflow_runs[mlflow_runs["model_type"] == mt].iloc[0]
            fig_multi.add_trace(go.Bar(
                name=mt,
                x=metric_cols,
                y=[row[c] for c in metric_cols],
            ))
        fig_multi.update_layout(
            title=_tr("All Metrics by Model"),
            barmode="group",
            yaxis_title="Score",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(fig_multi, use_container_width=True)

    # -----------------------------------------------------------------
    # Hyperparameter Analysis
    # -----------------------------------------------------------------
    # iter10 verify_v4 #10 (P14 degenerate sweep): all 3 runs share the
    # same LR (0.1) and identical training time (~1 s). That is a smoke-
    # test fixture, not a real hyperparameter sweep. Surface this up-
    # front and structurally collapse the "Learning Rate vs AUC" plot
    # if it is degenerate (≤1 unique LR value across runs).
    _lr_unique_count = 0
    _epochs_unique_count = 0
    if "params_lr" in mlflow_runs.columns:
        try:
            _lr_unique_count = int(
                mlflow_runs["params_lr"].dropna().nunique()
            )
        except Exception:
            _lr_unique_count = 0
    if "params_epochs" in mlflow_runs.columns:
        try:
            _epochs_unique_count = int(
                mlflow_runs["params_epochs"].dropna().nunique()
            )
        except Exception:
            _epochs_unique_count = 0
    _is_degenerate_sweep = (
        _lr_unique_count <= 1 and _epochs_unique_count <= 1
    )

    st.subheader(_tr("Hyperparameter Analysis"))
    if _is_degenerate_sweep:
        st.info(
            "Single hyperparameter configuration logged "
            "(LR=0.1, epochs=1). This is a smoke test, not a real "
            "sweep — see /docs for retraining with grid search. The "
            "Learning Rate vs AUC scatter has been replaced with a "
            "flat caption because plotting 3 points stacked on a "
            "single LR is structurally degenerate."
        )
    if "params_lr" in mlflow_runs.columns:
        col_hp1, col_hp2 = st.columns(2)
        with col_hp1:
            if _is_degenerate_sweep:
                _lr_only = (
                    mlflow_runs["params_lr"].dropna().iloc[0]
                    if not mlflow_runs["params_lr"].dropna().empty
                    else "n/a"
                )
                st.caption(
                    f"Learning Rate vs AUC — all logged runs use "
                    f"LR={_lr_only}. No sweep variance to plot."
                )
            else:
                fig_lr = px.scatter(
                    mlflow_runs, x="params_lr", y="auc",
                    color="model_type",
                    size="params_epochs",
                    title=_tr("Learning Rate vs AUC"),
                    labels={
                        "params_lr": "Learning Rate",
                        "auc": "AUC",
                    },
                    hover_data=["model_type", "params_epochs"],
                )
                fig_lr.update_xaxes(type="log")
                st.plotly_chart(fig_lr, use_container_width=True)

        with col_hp2:
            if _is_degenerate_sweep:
                _ep_only = (
                    mlflow_runs["params_epochs"].dropna().iloc[0]
                    if "params_epochs" in mlflow_runs.columns
                    and not mlflow_runs["params_epochs"].dropna().empty
                    else "n/a"
                )
                st.caption(
                    f"Epochs vs AUC — all logged runs use "
                    f"epochs={_ep_only}. No sweep variance to plot."
                )
            else:
                fig_epochs = px.scatter(
                    mlflow_runs, x="params_epochs", y="auc",
                    color="model_type",
                    size="training_time_s",
                    title=_tr("Epochs vs AUC (size = training time)"),
                    labels={
                        "params_epochs": "Epochs",
                        "auc": "AUC",
                    },
                )
                st.plotly_chart(fig_epochs, use_container_width=True)

    # -----------------------------------------------------------------
    # Training Efficiency Analysis
    # -----------------------------------------------------------------
    st.subheader(_tr("Training Efficiency"))
    col_te1, col_te2 = st.columns(2)

    with col_te1:
        fig_eff = px.scatter(
            mlflow_runs, x="training_time_s", y="auc",
            color="model_type",
            title=_tr("AUC vs Training Time"),
            labels={
                "training_time_s": "Training Time (seconds)",
                "auc": "AUC",
            },
            trendline="ols",
        )
        st.plotly_chart(fig_eff, use_container_width=True)

    with col_te2:
        # AUC per second of training
        runs_copy = mlflow_runs.copy()
        runs_copy["auc_per_second"] = (
            runs_copy["auc"] / runs_copy["training_time_s"].clip(lower=0.1)
        )
        fig_aps = px.bar(
            runs_copy.sort_values("auc_per_second", ascending=False),
            x="model_type", y="auc_per_second",
            title=_tr("AUC per Training Second (Efficiency)"),
            color="model_type",
            text="auc_per_second",
        )
        fig_aps.update_traces(
            texttemplate="%{text:.4f}", textposition="outside",
        )
        st.plotly_chart(fig_aps, use_container_width=True)

    # -----------------------------------------------------------------
    # Model Performance Radar (MLflow runs)
    # -----------------------------------------------------------------
    st.subheader(_tr("Model Performance Radar (MLflow Runs)"))
    radar_metrics = ["auc", "precision", "recall", "f1_score", "accuracy"]
    fig_radar = go.Figure()
    for _, row in mlflow_runs.iterrows():
        values = [row[m] for m in radar_metrics]
        values.append(values[0])
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_metrics + [radar_metrics[0]],
            fill="toself",
            name=row["model_type"],
            opacity=0.5,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=_tr("MLflow Run Performance Comparison"),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # -----------------------------------------------------------------
    # Run timeline
    # -----------------------------------------------------------------
    # iter10 verify_v4 #11 (P14 timeline 0.1 ms axis): the previous
    # implementation scatter-plotted 3 runs onto a sub-millisecond
    # timestamp axis, producing a degenerate "trend". Apply the
    # canonical drift_trend_guard helper — same pattern used on P08
    # / P13c — and fall back to a static run-list table when there
    # are not enough observations to plot a real timeline.
    if "timestamp" in mlflow_runs.columns:
        st.subheader(_tr("Experiment Timeline"))
        runs_timeline = mlflow_runs.copy()
        runs_timeline["timestamp"] = pd.to_datetime(
            runs_timeline["timestamp"], errors="coerce",
        )
        if not runs_timeline["timestamp"].isna().all():
            timeline_ok, timeline_msg = drift_trend_guard(
                runs_timeline["timestamp"], min_points=5,
            )
            if not timeline_ok:
                st.info(
                    f"{timeline_msg} Experiment timeline requires at "
                    "least 5 logged runs spanning ≥ 1 hour to render a "
                    "non-degenerate temporal axis. Showing a static "
                    "run-list table instead."
                )
                _fallback_cols = [
                    c for c in [
                        "model_type", "auc", "precision", "recall",
                        "f1_score", "training_time_s", "timestamp",
                    ] if c in runs_timeline.columns
                ]
                _fallback = runs_timeline[_fallback_cols].copy()
                _fallback.insert(
                    0, "Run",
                    [f"Run {i+1}" for i in range(len(_fallback))],
                )
                st.dataframe(_fallback, use_container_width=True)
            else:
                fig_timeline = px.scatter(
                    runs_timeline, x="timestamp", y="auc",
                    color="model_type",
                    size="training_time_s",
                    title=_tr("Model Performance Over Time"),
                    labels={
                        "timestamp": "Run Date",
                        "auc": "AUC",
                    },
                    hover_data=["model_type", "params_lr"],
                )
                fig_timeline.add_hline(
                    y=0.78, line_dash="dash", line_color="red",
                    annotation_text="Threshold",
                )
                st.plotly_chart(fig_timeline, use_container_width=True)


# =========================================================================
# Main Streamlit app entry point
# =========================================================================

def main():
    """Main Streamlit application entry point."""
    try:
        import streamlit as st
    except ImportError:
        logger.error("Streamlit not installed. pip install streamlit")
        return

    st.set_page_config(
        page_title=get_app_title(),
        page_icon=get_page_icon("Overview"),
        layout="wide",
    )

    config = load_config()

    # ------------------------------------------------------------------
    # Language toggle (한국어 ↔ English)
    # ------------------------------------------------------------------
    # Renders a small segmented control at the top of the sidebar so the
    # user can switch the shell language without leaving the page.
    # Persists across page changes via st.session_state["lang"].
    from src.dashboard.utils.dashboard_helpers import tr as _tr
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"
    _lang_choice = st.sidebar.radio(
        "🌐 Language / 언어",
        options=["en", "ko"],
        index=0 if st.session_state["lang"] == "en" else 1,
        format_func=lambda v: "English" if v == "en" else "한국어",
        horizontal=True,
        key="lang_radio",
    )
    if _lang_choice != st.session_state["lang"]:
        st.session_state["lang"] = _lang_choice
        st.rerun()
    _lang = st.session_state["lang"]

    # Sidebar navigation with icons (labels translated to current language)
    st.sidebar.title(_tr("Navigation", _lang))
    pages = get_page_list()
    page_labels = [
        f"{get_page_icon(p)} {_tr(p, _lang)}" for p in pages
    ]
    selected_label = st.sidebar.radio(_tr("Select Page", _lang), page_labels)
    # Map the translated label back to the canonical English page key
    page = pages[page_labels.index(selected_label)]

    data_loader = get_data_loader(config)

    # Synthetic-data banner — surface mode + group-size validation so a
    # viewer cannot mistake simulator output for real customer data.
    gen_summary: Dict[str, Any] = {}
    try:
        if hasattr(data_loader, "load_generation_summary"):
            gen_summary = data_loader.load_generation_summary() or {}
        else:
            from pathlib import Path as _Path
            import json as _json
            for candidate in (
                _Path("data/raw/generation_summary.json"),
                _Path("data/artifacts/generation_summary.json"),
            ):
                if candidate.exists():
                    gen_summary = _json.loads(candidate.read_text(encoding="utf-8"))
                    break
    except Exception:
        gen_summary = {}
    if isinstance(gen_summary, dict) and gen_summary:
        gen_mode = str(gen_summary.get("generation_mode", "")).lower()
        validation = gen_summary.get("validation", {}) or {}
        group_check = validation.get("group_size_check", {}) or {}
        group_passed = bool(group_check.get("passed", True))
        n_customers = gen_summary.get("num_customers")
        if gen_mode == "small" or not group_passed:
            _validation_label = _tr("PASSED", _lang) if group_passed else _tr("FAILED", _lang)
            st.warning(
                f"⚠️ **{_tr('Synthetic data', _lang)} — {gen_mode.upper() or _tr('UNKNOWN', _lang)} "
                f"{_tr('mode', _lang)} (n={n_customers}). "
                f"{_tr('Numbers shown are illustrative; they do NOT represent production performance.', _lang)} "
                f"{_tr('Group-size validation', _lang)}: {_validation_label}.**",
                icon="🧪",
            )
        else:
            st.info(
                f"{_tr('Synthetic data', _lang)} — {gen_mode.upper() or _tr('unknown', _lang)} "
                f"{_tr('mode', _lang)} (n={n_customers}). "
                f"{_tr('All KPIs are simulator-generated.', _lang)}",
                icon="🧪",
            )

    # Sidebar info from config helpers
    sidebar_info = build_sidebar_info(config)
    churn_def = sidebar_info["churn_definition"]
    budget_info = sidebar_info["budget"]
    ew = sidebar_info["ensemble_weights"]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{_tr('Churn Definition', _lang)}**")
    st.sidebar.markdown(
        f"- {_tr('No purchase', _lang)}: {churn_def['no_purchase_days']} {_tr('days', _lang)}\n"
        f"- {_tr('No login', _lang)}: {churn_def['no_login_days']} {_tr('days', _lang)}\n"
        f"- {_tr('Operator', _lang)}: {churn_def['operator']}"
    )
    st.sidebar.markdown(f"**{_tr('Budget', _lang)}**")
    st.sidebar.markdown(
        f"- {_tr('Total', _lang)}: {format_currency(budget_info['total_krw'], budget_info['currency'])}"
    )
    st.sidebar.markdown(f"**{_tr('Ensemble Weights', _lang)}**")
    st.sidebar.markdown(
        f"- ML: {ew['ml']} | DL: {ew['dl']}"
    )

    # Manual refresh button
    st.sidebar.markdown("---")
    if st.sidebar.button(f"🔄 {_tr('Refresh Data', _lang)}"):
        st.cache_data.clear()
        st.rerun()

    # Route to page
    page_map = {
        "Demo": render_demo,
        "Overview": render_overview,
        "Churn Analytics": render_churn_analytics,
        "Model Performance": render_model_performance,
        "Customer Segmentation": render_segmentation,
        "Cohort Analysis": render_cohort_analysis,
        "Budget Optimization": render_budget_optimization,
        "A/B Testing": render_ab_testing,
        "Survival Analysis": render_survival_analysis,
        "Model Monitoring": render_model_monitoring,
        "Recommendations": render_recommendations,
        "CLV Prediction": render_clv,
        "Uplift Modeling": render_uplift,
        "CLV & Retention Campaign": render_retention_campaign,
        "Real-Time Scoring": render_realtime_scoring,
        "MLflow Experiments": render_mlflow_experiments,
        "System Health": render_system_health,
    }

    render_fn = page_map.get(page, render_overview)
    render_fn(st, config, data_loader)


if __name__ == "__main__":
    main()
