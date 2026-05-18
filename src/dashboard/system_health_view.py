"""
System Overview & Health Dashboard View.

Provides a unified health dashboard showing:
- System service status (Redis, MLflow, Pipeline)
- Streaming pipeline status with Redis stream metrics
- MLflow experiment tracking integration and run history
- Overall system health indicators and alerts

All configurable parameters are sourced from config/simulator_config.yaml.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Defensive import — F1 may add helpers concurrently.
try:  # pragma: no cover - import guard
    from src.dashboard.utils.dashboard_helpers import format_count  # type: ignore
except ImportError:  # pragma: no cover
    def format_count(value: int) -> str:  # type: ignore[no-redef]
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return str(value)

try:  # pragma: no cover - import guard
    from src.dashboard.utils.dashboard_helpers import drift_trend_guard  # type: ignore
except ImportError:  # pragma: no cover
    def drift_trend_guard(*_args, **_kwargs):  # type: ignore[no-redef]
        # Default fallback: no veto (parent helper not yet committed).
        return True, None

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


# Maximum age (hours) before "Scoring Throughput (24h)" data is considered
# stale relative to wall-clock; suppresses Oct-2024-vs-May-2026 drift split.
THROUGHPUT_FRESHNESS_HOURS = 48

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


def _artifact_reason(artifact: Any) -> Optional[str]:
    """Return ``artifact.reason`` when present, else None."""
    if artifact is None:
        return None
    reason = getattr(artifact, "reason", None)
    return str(reason) if reason else None

# Service status constants
STATUS_HEALTHY = "healthy"
STATUS_DEGRADED = "degraded"
STATUS_DOWN = "down"

STATUS_COLORS = {
    STATUS_HEALTHY: "#2ecc71",
    STATUS_DEGRADED: "#f39c12",
    STATUS_DOWN: "#e74c3c",
}

STATUS_ICONS = {
    STATUS_HEALTHY: "✅",
    STATUS_DEGRADED: "⚠️",
    STATUS_DOWN: "❌",
}


def resolve_redis_connection_config(config: Dict) -> Dict[str, Any]:
    """Resolve Redis connection settings with Docker env overrides."""
    redis_config = config.get("redis", {})
    return {
        "host": os.environ.get("REDIS_HOST", redis_config.get("host", "localhost")),
        "port": int(os.environ.get("REDIS_PORT", redis_config.get("port", 6379))),
        "db": int(os.environ.get("REDIS_DB", redis_config.get("db", 0))),
        "stream_name": redis_config.get("stream_name", "scoring_requests"),
        "response_stream": redis_config.get("response_stream", "scoring_responses"),
        "consumer_group": redis_config.get("consumer_group", "scoring_consumers"),
        "consumer_batch_size": redis_config.get("consumer_batch_size", 10),
        "stream_maxlen": redis_config.get("stream_maxlen", 10000),
        "cache_ttl_seconds": redis_config.get("cache_ttl_seconds", 3600),
    }


def check_redis_health(config: Dict) -> Dict[str, Any]:
    """Check Redis streaming pipeline health.

    Args:
        config: Configuration dictionary with redis section.

    Returns:
        Dict with status, connected, stream_lengths, consumer_groups,
        memory_used, uptime, and error details.
    """
    redis_config = resolve_redis_connection_config(config)
    result = {
        "status": STATUS_DOWN,
        "connected": False,
        "host": redis_config["host"],
        "port": redis_config["port"],
        "stream_lengths": {},
        "consumer_groups": {},
        "memory_used_mb": 0,
        "uptime_seconds": 0,
        "error": None,
    }

    try:
        import redis as redis_lib
        r = redis_lib.Redis(
            host=result["host"],
            port=result["port"],
            db=redis_config["db"],
            socket_connect_timeout=2,
        )
        r.ping()
        result["connected"] = True
        result["status"] = STATUS_HEALTHY

        # Stream lengths
        req_stream = redis_config["stream_name"]
        resp_stream = redis_config["response_stream"]
        for stream_name in [req_stream, resp_stream]:
            try:
                if r.exists(stream_name):
                    result["stream_lengths"][stream_name] = r.xlen(stream_name)
                else:
                    result["stream_lengths"][stream_name] = 0
            except Exception:
                result["stream_lengths"][stream_name] = 0

        # Server info
        try:
            info = r.info()
            result["memory_used_mb"] = round(
                info.get("used_memory", 0) / (1024 * 1024), 2
            )
            result["uptime_seconds"] = info.get("uptime_in_seconds", 0)
        except Exception:
            pass

        # Consumer group info
        for stream_name in [req_stream, resp_stream]:
            try:
                if r.exists(stream_name):
                    groups = r.xinfo_groups(stream_name)
                    result["consumer_groups"][stream_name] = [
                        {
                            "name": g.get("name", b"").decode()
                            if isinstance(g.get("name", b""), bytes)
                            else str(g.get("name", "")),
                            "consumers": g.get("consumers", 0),
                            "pending": g.get("pending", 0),
                        }
                        for g in groups
                    ]
            except Exception:
                result["consumer_groups"][stream_name] = []

    except ImportError:
        result["error"] = "redis package not installed"
    except Exception as e:
        result["error"] = str(e)

    return result


def check_mlflow_health(config: Dict) -> Dict[str, Any]:
    """Check MLflow tracking server health.

    Single source of truth used by both Page 15 KPI cards and the Page 14
    banner copy (rendered via :func:`mlflow_status_banner`). "Connected"
    requires an actual MLflow API round-trip that returns at least one
    experiment — opening a sqlite file is not sufficient (defect a4 P14↔P15).

    Args:
        config: Configuration dictionary with mlflow section.

    Returns:
        Dict with status, connected, tracking_uri, experiments,
        recent_runs, and error details.
    """
    mlflow_config = config.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri", "sqlite:///mlflow/mlflow.db")

    result = {
        "status": STATUS_DOWN,
        "connected": False,
        "tracking_uri": tracking_uri,
        "experiment_name": mlflow_config.get("experiment_name", "churn_prediction"),
        "experiments": [],
        "total_runs": 0,
        "recent_runs": [],
        "best_run": None,
        "error": None,
    }

    parsed_uri = urlparse(tracking_uri)

    if parsed_uri.scheme in {"http", "https"}:
        health_url = tracking_uri.rstrip("/") + "/health"
        try:
            import requests

            response = requests.get(health_url, timeout=1)
            if not response.ok:
                result["status"] = STATUS_DEGRADED
                result["error"] = (
                    f"MLflow health endpoint returned {response.status_code}"
                )
                return result
        except ImportError:
            result["status"] = STATUS_DEGRADED
            result["error"] = "requests package not installed"
            return result
        except Exception as e:
            result["status"] = STATUS_DEGRADED
            result["error"] = f"MLflow server not reachable: {e}"
            return result
        # HTTP /health passed — fall through to API probe for canonical answer.

    if parsed_uri.scheme == "sqlite":
        db_path = Path(parsed_uri.path)
        if not db_path.exists():
            result["status"] = STATUS_DOWN
            result["error"] = f"MLflow tracking DB not found: {db_path}"
            return result
        # File exists, but we still require a successful API probe below.

    # Canonical API probe — same call path Page 14 uses to decide whether
    # to show "MLflow tracking server not available". Connected ⇔ at least
    # one experiment is returned via the mlflow client.
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        experiments = mlflow.search_experiments()

        if experiments:
            result["connected"] = True
            result["status"] = STATUS_HEALTHY
            result["experiments"] = [
                {
                    "name": e.name,
                    "id": e.experiment_id,
                    "lifecycle": e.lifecycle_stage,
                }
                for e in experiments
            ]

            # Get recent runs
            try:
                runs = mlflow.search_runs(
                    experiment_names=[result["experiment_name"]],
                    max_results=10,
                    order_by=["start_time DESC"],
                )
                if not runs.empty:
                    result["total_runs"] = len(runs)
                    result["recent_runs"] = runs.head(5).to_dict("records")

                    # Find best run by AUC
                    auc_col = None
                    for col in runs.columns:
                        if "auc" in col.lower():
                            auc_col = col
                            break
                    if auc_col and not runs[auc_col].isna().all():
                        best_idx = runs[auc_col].idxmax()
                        result["best_run"] = {
                            "run_id": runs.loc[best_idx, "run_id"],
                            "auc": float(runs.loc[best_idx, auc_col]),
                        }
            except Exception:
                pass
        else:
            # API reachable but registry empty — degraded, NOT connected.
            # Page 14 will display the "tracking server not available /
            # showing cached data" banner; Page 15 must agree (defect a4).
            result["connected"] = False
            result["status"] = STATUS_DEGRADED
            result["error"] = "No experiments found on MLflow server"

    except ImportError as e:
        # The previous implementation reported every ImportError as
        # "mlflow package not installed", which masked transitive-dep gaps
        # (e.g. `alembic`, required by MLflow's SQLAlchemy backend during
        # the first `search_experiments()` call on a sqlite tracking URI).
        # Distinguish: only claim mlflow itself is missing when its name
        # appears in the error; otherwise surface the actual missing module
        # and mark the service degraded, not down.
        missing = getattr(e, "name", "") or ""
        message = str(e)
        if missing == "mlflow" or "mlflow" in message.lower().split("'"):
            result["status"] = STATUS_DOWN
            result["error"] = "mlflow package not installed"
        else:
            result["status"] = STATUS_DEGRADED
            actual = missing or message
            result["error"] = (
                f"MLflow dependency missing: {actual} "
                "(install the missing transitive dep on the dashboard image)"
            )
    except ModuleNotFoundError as e:  # pragma: no cover - defensive
        # Some MLflow code paths re-raise ModuleNotFoundError from inside
        # the SQLAlchemy backend; treat the same as ImportError above.
        missing = getattr(e, "name", "") or str(e)
        if missing == "mlflow":
            result["status"] = STATUS_DOWN
            result["error"] = "mlflow package not installed"
        else:
            result["status"] = STATUS_DEGRADED
            result["error"] = (
                f"MLflow dependency missing: {missing} "
                "(install the missing transitive dep on the dashboard image)"
            )
    except Exception as e:
        result["status"] = STATUS_DOWN
        result["error"] = f"MLflow server not reachable: {e}"

    return result


def mlflow_status_banner(mlflow_health: Dict[str, Any]) -> Tuple[str, str]:
    """Return canonical (level, message) for the MLflow status banner.

    Centralised so Page 14 and Page 15 cannot drift. ``level`` is one of
    ``"success" | "warning" | "error"`` and matches Streamlit's banner API.
    """
    if mlflow_health.get("connected"):
        return "success", "Connected to MLflow tracking server"
    err = mlflow_health.get("error") or "tracking server not reachable"
    return (
        "warning",
        f"MLflow tracking server not available ({err}) — "
        "showing cached experiment data from artifacts.",
    )


def check_pipeline_health(config: Dict) -> Dict[str, Any]:
    """Check ML pipeline health based on artifacts and state.

    Args:
        config: Configuration dictionary.

    Returns:
        Dict with status, models_available, last_training,
        and artifact details.
    """
    from pathlib import Path

    configured_artifacts_dir = Path(
        config.get("dashboard", {}).get("artifacts_dir", "data/artifacts")
    )
    artifact_candidates = [configured_artifacts_dir, Path("results")]
    artifacts_dir = next(
        (candidate for candidate in artifact_candidates if candidate.exists()),
        configured_artifacts_dir,
    )
    models_dir = Path("models")

    result = {
        "status": STATUS_DOWN,
        "artifacts_dir": str(artifacts_dir),
        "artifact_sources": [str(path) for path in artifact_candidates],
        "artifacts_exist": artifacts_dir.exists(),
        "artifact_count": 0,
        "models_available": [],
        "last_modified": None,
        "error": None,
    }

    try:
        if artifacts_dir.exists():
            artifact_files = list(artifacts_dir.glob("*"))
            result["artifact_count"] = len(artifact_files)

            if artifact_files:
                latest = max(artifact_files, key=lambda f: f.stat().st_mtime)
                result["last_modified"] = datetime.fromtimestamp(
                    latest.stat().st_mtime
                ).isoformat()

        if models_dir.exists():
            model_files = list(models_dir.glob("*.joblib")) + list(
                models_dir.glob("*.pt")
            )
            result["models_available"] = [f.name for f in model_files]

        if result["artifact_count"] > 0 or result["models_available"]:
            result["status"] = STATUS_HEALTHY
        elif result["artifacts_exist"]:
            result["status"] = STATUS_DEGRADED
            result["error"] = "Artifacts directory exists but is empty"
        else:
            result["error"] = "No artifacts or models found"

    except Exception as e:
        result["error"] = str(e)

    return result


def _drift_alert_to_status(alert_level: Optional[str]) -> str:
    """Map a drift alert level (green/yellow/red) onto a service status."""
    if not alert_level:
        return STATUS_HEALTHY
    level = str(alert_level).lower()
    if level == "green":
        return STATUS_HEALTHY
    if level == "yellow":
        return STATUS_DEGRADED
    if level == "red":
        return STATUS_DOWN
    return STATUS_DEGRADED


def get_system_health_summary(
    config: Dict,
    data_loader: Any = None,
) -> Dict[str, Any]:
    """Get comprehensive system health summary.

    Aggregate health propagates the WORST child state, including the
    model-drift subsystem (defect a4 P15: header was green while
    Drift=RED). Pass ``data_loader`` so drift history can be folded in;
    when omitted the rollup falls back to infrastructure services only.

    Args:
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader for drift status.

    Returns:
        Dict with service statuses, overall health, and timestamp.
    """
    redis_health = check_redis_health(config)
    mlflow_health = check_mlflow_health(config)
    pipeline_health = check_pipeline_health(config)

    drift_status = STATUS_HEALTHY
    drift_alert_level: Optional[str] = None
    drift_error: Optional[str] = None
    if data_loader is not None:
        try:
            drift_history = data_loader.load_drift_history()
            if drift_history is not None and not drift_history.empty:
                # Only the ``__overall__`` summary row should drive the
                # aggregate rollup — per-feature rows are surfaced inside
                # the drift page, not the system banner.
                #
                # Additionally, the first pipeline run writes its synthetic
                # ``__overall__`` row with ``is_initial_check=True`` and a
                # conservative ``alert_level="red"``, even when the actual
                # PSI is well below the yellow threshold. Filtering those
                # initial-check rows out prevents the System Health page
                # from flashing "System Issues Detected" on day one of a
                # fresh deployment (and on any later baseline reset).
                drift_df = drift_history
                if "feature_name" in drift_df.columns:
                    drift_df = drift_df[
                        drift_df["feature_name"] == "__overall__"
                    ]
                if (
                    not drift_df.empty
                    and "is_initial_check" in drift_df.columns
                ):
                    is_initial = drift_df["is_initial_check"].fillna(False)
                    # Accept both bool and string representations ("True"/"False").
                    is_initial = is_initial.apply(
                        lambda v: str(v).strip().lower() == "true"
                        if not isinstance(v, bool) else v
                    )
                    drift_df = drift_df[~is_initial]

                if drift_df.empty:
                    # No informative overall row yet — treat as healthy and
                    # let the drift page show the baseline-pending caption.
                    drift_status = STATUS_HEALTHY
                    drift_alert_level = None
                    drift_error = "drift baseline not yet established"
                else:
                    latest_alert = drift_df.iloc[-1].get("alert_level")
                    drift_alert_level = (
                        str(latest_alert).lower() if latest_alert else None
                    )
                    drift_status = _drift_alert_to_status(drift_alert_level)
                    if drift_status != STATUS_HEALTHY:
                        drift_error = (
                            f"Latest drift alert: {drift_alert_level}"
                        )
        except Exception as exc:  # pragma: no cover - defensive
            drift_error = f"drift history unavailable: {exc}"

    drift_health = {
        "status": drift_status,
        "alert_level": drift_alert_level,
        "error": drift_error,
    }

    statuses = [
        redis_health["status"],
        mlflow_health["status"],
        pipeline_health["status"],
        drift_status,
    ]

    # Worst-child propagation: any DOWN ⇒ DOWN, any DEGRADED ⇒ DEGRADED.
    if any(s == STATUS_DOWN for s in statuses):
        overall = STATUS_DOWN
    elif any(s == STATUS_DEGRADED for s in statuses):
        overall = STATUS_DEGRADED
    else:
        overall = STATUS_HEALTHY

    return {
        "overall_status": overall,
        "timestamp": datetime.now().isoformat(),
        "services": {
            "redis": redis_health,
            "mlflow": mlflow_health,
            "pipeline": pipeline_health,
            "drift": drift_health,
        },
    }


def render_system_health(st_module, config: Dict, data_loader=None):
    """Render system overview and health dashboard.

    Shows:
    - Overall system health status
    - Service health indicators (Redis, MLflow, Pipeline)
    - Streaming pipeline status with stream metrics
    - MLflow experiment tracking integration
    - Drift detection summary
    - Recent activity log

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module
    st.header(_tr("System Overview & Health"))
    st.markdown(_tr(
        "Real-time health monitoring for all system components: "
        "streaming pipeline, ML tracking, and model serving."
    ))

    if data_loader is None:
        from src.dashboard.app import get_data_loader
        data_loader = get_data_loader(config)

    # Gather health data — pass data_loader so drift status is folded into
    # the aggregate rollup (defect a4: header was green while Drift=RED).
    health = get_system_health_summary(config, data_loader=data_loader)

    # Pre-load mlflow runs once so the service card and the run-history
    # section share a single source of truth (defect a4 P15: Experiments=0
    # vs Total Runs=3 came from two independent code paths). iter13 G4:
    # probe with as_artifact=True so we can detect the "cached fallback"
    # case and surface it on the MLflow service card.
    mlflow_runs_df, mlflow_runs_artifact = _load_artifact_safely(
        getattr(data_loader, "load_mlflow_runs", None),
    )

    # ==================================================================
    # Section 1: Overall Health Status
    # ==================================================================
    _render_overall_health(st, health)

    # ==================================================================
    # Section 2: Service Status Cards
    # ==================================================================
    st.markdown("---")
    _render_service_cards(
        st,
        health,
        mlflow_runs_df=mlflow_runs_df,
        mlflow_runs_artifact=mlflow_runs_artifact,
    )

    # ==================================================================
    # Section 3: Streaming Pipeline Status
    # ==================================================================
    st.markdown("---")
    _render_streaming_status(st, config, health, data_loader)

    # ==================================================================
    # Section 4: MLflow Experiment Tracking
    # ==================================================================
    st.markdown("---")
    _render_mlflow_tracking(
        st, config, health, data_loader, mlflow_runs_df=mlflow_runs_df,
    )

    # ==================================================================
    # Section 5: Model Health & Drift Summary
    # ==================================================================
    st.markdown("---")
    _render_model_health(st, config, data_loader)

    # ==================================================================
    # Section 6: System Configuration
    # ==================================================================
    st.markdown("---")
    _render_system_config(st, config)


# =========================================================================
# Internal rendering helpers
# =========================================================================


def _throughput_freshness(
    throughput: pd.DataFrame,
) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
    """Return (latest_timestamp, age_in_hours) for a throughput frame.

    Falls back to (None, None) when the frame has no usable timestamp
    column. Age is measured against ``datetime.now()`` and is naive on
    purpose to match the un-tz'd fixture timestamps.
    """
    if throughput is None or throughput.empty:
        return None, None
    if "timestamp" not in throughput.columns:
        return None, None
    try:
        ts = pd.to_datetime(throughput["timestamp"], errors="coerce")
        ts = ts.dropna()
        if ts.empty:
            return None, None
        latest = ts.max()
        # Strip timezone for a uniform comparison.
        if getattr(latest, "tzinfo", None) is not None:
            latest = latest.tz_convert(None)
        now = pd.Timestamp(datetime.now())
        age = (now - latest).total_seconds() / 3600.0
        return latest, age
    except Exception:  # pragma: no cover - defensive
        return None, None


def _render_overall_health(st, health: Dict):
    """Render overall system health banner.

    Worst-child propagation: header turns yellow (Degraded) when any
    subsystem is degraded, red (Issues Detected) when any subsystem is
    down — including the drift child folded in by
    :func:`get_system_health_summary` (defect a4 P15).
    """
    overall = health["overall_status"]
    icon = STATUS_ICONS.get(overall, "❓")
    timestamp = health.get("timestamp", "N/A")

    status_text = {
        STATUS_HEALTHY: _tr("All Systems Operational"),
        STATUS_DEGRADED: _tr("Degraded — Investigate Subsystems"),
        STATUS_DOWN: _tr("System Issues Detected"),
    }

    headline = (
        f"### {icon} {_tr('System Status')}: "
        f"**{status_text.get(overall, _tr('Unknown'))}**"
    )
    if overall == STATUS_HEALTHY:
        st.markdown(headline)
    elif overall == STATUS_DEGRADED:
        st.warning(headline)
    else:
        st.error(headline)

    # Surface which subsystem(s) dragged the rollup down so the headline
    # cannot silently disagree with a child card (defect a4 P15).
    services = health.get("services", {})
    bad_children = [
        name for name, svc in services.items()
        if svc.get("status") in (STATUS_DEGRADED, STATUS_DOWN)
    ]
    if bad_children:
        st.caption(
            f"{_tr('Non-healthy subsystems')}: "
            + ", ".join(sorted(bad_children))
        )

    st.caption(f"{_tr('Last checked')}: {timestamp}")

    # Service count summary
    healthy_count = sum(
        1 for s in services.values() if s.get("status") == STATUS_HEALTHY
    )
    total_count = len(services)

    st.progress(
        healthy_count / max(total_count, 1),
        text=(
            f"{format_count(healthy_count)}/{format_count(total_count)} "
            + _tr("subsystems healthy")
        ),
    )


def _render_service_cards(
    st,
    health: Dict,
    mlflow_runs_df=None,
    mlflow_runs_artifact: Any = None,
):
    """Render individual service health cards.

    ``mlflow_runs_df`` is the canonical run table loaded by the page; it
    is used to reconcile the Experiments KPI with the Run-History KPI on
    the same page (defect a4 P15 contradiction A: Experiments=0 vs
    Total Runs=3 from two different code paths).

    ``mlflow_runs_artifact`` (iter13 G4) is the optional DashboardArtifact
    wrapper from G2's loader. When ``is_real=False`` it tells the card
    "this run table is cached — not live MLflow API output" so the card
    can append an explicit "Experiments cached: N rows" line.
    """
    st.subheader(_tr("Service Health"))

    services = health.get("services", {})
    cols = st.columns(3)

    service_display = [
        ("redis", _tr("Redis Streaming"), _tr("Handles real-time scoring requests")),
        ("mlflow", _tr("MLflow Tracking"), _tr("Experiment tracking & model registry")),
        ("pipeline", _tr("ML Pipeline"), _tr("Model training & artifact storage")),
    ]

    for i, (key, name, desc) in enumerate(service_display):
        svc = services.get(key, {})
        status = svc.get("status", STATUS_DOWN)
        icon = STATUS_ICONS.get(status, "❓")
        error = svc.get("error")

        with cols[i]:
            if status == STATUS_HEALTHY:
                st.success(f"{icon} **{name}**")
            elif status == STATUS_DEGRADED:
                st.warning(f"{icon} **{name}**")
            else:
                st.error(f"{icon} **{name}**")

            st.caption(desc)

            if error:
                st.caption(f"⚠ {error}")

            # Service-specific details
            if key == "redis":
                connected = bool(svc.get("connected"))
                streams = svc.get("stream_lengths", {})
                total_msgs = sum(streams.values()) if streams else 0
                # Hollow-health guard (defect a4): "Connected: Yes" with
                # zero traffic should read as idle, not healthy.
                if connected and total_msgs == 0:
                    conn_label = _tr("Yes (idle — no traffic)")
                else:
                    conn_label = _tr("Yes") if connected else _tr("No")
                st.metric(_tr("Connected"), conn_label)
                for stream_name, length in streams.items():
                    short_name = stream_name.split("_")[-1]
                    st.metric(
                        f"{_tr('Stream')} ({short_name})",
                        format_count(int(length)),
                    )

            elif key == "mlflow":
                # Reconcile with Run-History KPI: prefer the canonical
                # mlflow_runs DataFrame (same source as Total Runs) when
                # the live API call returned an empty experiments list.
                live_experiments = svc.get("experiments", []) or []
                exp_from_health = len(live_experiments)

                runs_experiment_count = 0
                runs_total = 0
                if mlflow_runs_df is not None and not mlflow_runs_df.empty:
                    runs_total = len(mlflow_runs_df)
                    if "experiment_id" in mlflow_runs_df.columns:
                        runs_experiment_count = int(
                            mlflow_runs_df["experiment_id"].nunique()
                        )
                    elif "experiment_name" in mlflow_runs_df.columns:
                        runs_experiment_count = int(
                            mlflow_runs_df["experiment_name"].nunique()
                        )
                    else:
                        # Cached snapshot exposes runs but not the
                        # experiment id; treat as a single experiment so
                        # "0 experiments / 3 runs" can never happen.
                        runs_experiment_count = 1

                exp_count = max(exp_from_health, runs_experiment_count)
                connected = bool(svc.get("connected")) and exp_count > 0
                if connected and runs_total == 0:
                    conn_label = _tr("Yes (idle — no runs logged)")
                elif not connected and runs_total > 0:
                    conn_label = _tr("No (showing cached runs)")
                else:
                    conn_label = _tr("Yes") if connected else _tr("No")
                st.metric(_tr("Connected"), conn_label)
                st.metric(_tr("Experiments"), format_count(exp_count))
                st.metric(_tr("Total Runs"), format_count(runs_total))

                # iter13 G4: when the loader explicitly tells us the
                # run table is a cached fallback (is_real=False), show
                # how many rows we are displaying so the operator can
                # see that "Connected: No" is paired with cached data.
                if (
                    _artifact_marked_unreal(mlflow_runs_artifact)
                    and runs_total > 0
                ):
                    st.caption(
                        f"{_tr('Experiments cached')}: "
                        f"{format_count(runs_total)} {_tr('rows')}"
                    )
                    cached_reason = _artifact_reason(mlflow_runs_artifact)
                    if cached_reason:
                        st.caption(f"{_tr('Reason')}: {cached_reason}")

            elif key == "pipeline":
                artifacts = int(svc.get("artifact_count", 0))
                models = svc.get("models_available", [])
                st.metric(_tr("Artifacts"), format_count(artifacts))
                st.metric(_tr("Models"), format_count(len(models)))
                if artifacts == 0 and not models:
                    st.caption(_tr("Pipeline idle — no artifacts or models."))


def _render_streaming_status(
    st, config: Dict, health: Dict, data_loader,
):
    """Render streaming pipeline status section."""
    st.subheader(_tr("Streaming Pipeline Status"))

    redis_health = health.get("services", {}).get("redis", {})
    redis_config = resolve_redis_connection_config(config)

    # Stream configuration
    col_config, col_metrics = st.columns(2)

    with col_config:
        st.markdown(f"#### {_tr('Configuration')}")
        st.json({
            "host": redis_config["host"],
            "port": redis_config["port"],
            "request_stream": redis_config["stream_name"],
            "response_stream": redis_config["response_stream"],
            "consumer_group": redis_config["consumer_group"],
            "batch_size": redis_config["consumer_batch_size"],
            "max_stream_length": redis_config["stream_maxlen"],
            "cache_ttl_seconds": redis_config["cache_ttl_seconds"],
        })

    with col_metrics:
        st.markdown(f"#### {_tr('Stream Metrics')}")
        streams = redis_health.get("stream_lengths", {})

        if streams:
            stream_df = pd.DataFrame([
                {_tr("Stream"): name, _tr("Length"): length}
                for name, length in streams.items()
            ])
            fig_streams = px.bar(
                stream_df, x=_tr("Stream"), y=_tr("Length"),
                title=_tr("Stream Lengths"),
                color=_tr("Stream"),
                text=_tr("Length"),
            )
            fig_streams.update_traces(textposition="outside")
            st.plotly_chart(fig_streams, use_container_width=True)
        else:
            st.info(_tr(
                "Redis not connected. Stream metrics unavailable. "
                "Start Redis with `docker-compose up redis`."
            ))

    # Consumer group details
    consumer_groups = redis_health.get("consumer_groups", {})
    if any(consumer_groups.values()):
        st.markdown(f"#### {_tr('Consumer Groups')}")
        group_rows = []
        for stream, groups in consumer_groups.items():
            for g in groups:
                group_rows.append({
                    _tr("Stream"): stream,
                    _tr("Group"): g.get("name", "N/A"),
                    _tr("Consumers"): g.get("consumers", 0),
                    _tr("Pending Messages"): g.get("pending", 0),
                })
        if group_rows:
            st.dataframe(pd.DataFrame(group_rows), use_container_width=True)

    # Throughput chart from data loader. The cached fixture is dated
    # Oct 2024 while the drift charts are dated today; rather than render
    # an Oct-2024 24h line beside a "May 2026" drift trend (defect a4
    # time-anchor split), we surface the freshness gap explicitly and
    # only show the chart when the data is recent.
    throughput, throughput_artifact = _load_artifact_safely(
        getattr(data_loader, "load_scoring_throughput", None),
    )
    if throughput is None:
        throughput = pd.DataFrame()

    # iter13 G4: skip the chart entirely and surface a hard error when
    # the loader reports the underlying artifact is NOT real (closes the
    # iter12 audit finding for Page 15 throughput tile).
    if _artifact_marked_unreal(throughput_artifact):
        st.markdown(f"#### {_tr('Scoring Throughput')}")
        st.error(_tr(
            "Real scoring throughput missing — run pipeline to populate "
            "`results/scoring_throughput.csv`."
        ))
        reason = _artifact_reason(throughput_artifact)
        if reason:
            st.caption(f"{_tr('Reason')}: {reason}")
    elif not throughput.empty:
        last_ts, age_hours = _throughput_freshness(throughput)
        last_label = (
            last_ts.isoformat() if last_ts is not None else _tr("unknown")
        )
        is_stale = age_hours is None or age_hours > THROUGHPUT_FRESHNESS_HOURS

        if is_stale:
            st.markdown(f"#### {_tr('Scoring Throughput')}")
            st.warning(
                f"{_tr('Throughput telemetry is stale')} "
                f"({_tr('last sample')}: {last_label}). "
                f"{_tr('Chart suppressed to avoid a time-anchor split with the drift trend below. Restart the scoring pipeline to refresh.')}"
            )
        else:
            st.markdown(f"#### {_tr('Scoring Throughput (last 24h)')}")
            st.caption(f"{_tr('Last refresh')}: {last_label}")
            fig_tp = go.Figure()
            fig_tp.add_trace(go.Scatter(
                x=throughput["timestamp"],
                y=throughput["requests_per_minute"],
                mode="lines",
                name=_tr("Requests/min"),
                line=dict(color="#2ecc71", width=2),
                fill="tozeroy",
                fillcolor="rgba(46,204,113,0.1)",
            ))
            fig_tp.update_layout(
                xaxis_title=_tr("Time"),
                yaxis_title=_tr("Requests per Minute"),
                height=300,
            )
            st.plotly_chart(fig_tp, use_container_width=True)

            # Throughput summary
            tp1, tp2, tp3 = st.columns(3)
            tp1.metric(
                _tr("Avg Throughput"),
                f"{throughput['requests_per_minute'].mean():.1f} req/min",
            )
            tp2.metric(
                _tr("Avg Latency"),
                f"{throughput['avg_latency_ms'].mean():.1f} ms",
            )
            if "error_rate" in throughput.columns:
                tp3.metric(
                    _tr("Avg Error Rate"),
                    f"{throughput['error_rate'].mean():.4f}",
                )


def _render_mlflow_tracking(
    st, config: Dict, health: Dict, data_loader, mlflow_runs_df=None,
):
    """Render MLflow experiment tracking integration section.

    Banner copy is sourced from :func:`mlflow_status_banner` so Page 15
    cannot disagree with Page 14 (defect a4 cross-page contradiction).
    """
    st.subheader(_tr("MLflow Experiment Tracking"))

    mlflow_health = health.get("services", {}).get("mlflow", {})
    mlflow_config = config.get("mlflow", {})

    # Connection status and config
    col_status, col_info = st.columns(2)

    with col_status:
        level, message = mlflow_status_banner(mlflow_health)
        if level == "success":
            st.success(_tr(message))
        elif level == "error":
            st.error(_tr(message))
        else:
            st.warning(_tr(message))
        st.json({
            "tracking_uri": mlflow_health.get("tracking_uri", "N/A"),
            "experiment_name": mlflow_health.get("experiment_name", "N/A"),
            "log_models": mlflow_config.get("log_models", True),
            "log_artifacts": mlflow_config.get("log_artifacts", True),
        })

    with col_info:
        # Experiment list
        experiments = mlflow_health.get("experiments", [])
        if experiments:
            st.markdown(f"#### {_tr('Registered Experiments')}")
            st.dataframe(
                pd.DataFrame(experiments),
                use_container_width=True,
            )
        else:
            st.info(_tr(
                "No experiments returned by the MLflow server. "
                "Run history below is loaded from cached artifacts."
            ))

    # Reuse the cached DataFrame so this section and the service card
    # show the same Total Runs (defect a4 contradiction A).
    if mlflow_runs_df is not None:
        mlflow_runs = mlflow_runs_df
    else:
        mlflow_runs = data_loader.load_mlflow_runs()

    if not mlflow_runs.empty:
        st.markdown(f"#### {_tr('Experiment Run History')}")

        # KPI cards from runs
        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric(_tr("Total Runs"), len(mlflow_runs))

        if "auc" in mlflow_runs.columns:
            best_auc = mlflow_runs["auc"].max()
            kc2.metric(_tr("Best AUC"), f"{best_auc:.4f}")

        if "model_type" in mlflow_runs.columns:
            best_model = mlflow_runs.loc[
                mlflow_runs["auc"].idxmax(), "model_type"
            ] if "auc" in mlflow_runs.columns else "N/A"
            kc3.metric(_tr("Best Model"), best_model)

        if "training_time_s" in mlflow_runs.columns:
            total_time = mlflow_runs["training_time_s"].sum()
            kc4.metric(_tr("Total Train Time"), f"{total_time:.0f}s")

        # Performance comparison chart
        col_perf, col_time = st.columns(2)

        with col_perf:
            if "model_type" in mlflow_runs.columns and "auc" in mlflow_runs.columns:
                fig_perf = px.bar(
                    mlflow_runs.sort_values("auc", ascending=False),
                    x="model_type",
                    y="auc",
                    color="model_type",
                    title=_tr("AUC by Model Type"),
                    text=mlflow_runs.sort_values("auc", ascending=False)["auc"].apply(
                        lambda v: f"{v:.4f}"
                    ),
                )
                fig_perf.update_traces(textposition="outside")
                fig_perf.update_layout(
                    yaxis_range=[0, 1],
                    showlegend=False,
                )
                fig_perf.add_hline(
                    y=0.78, line_dash="dash", line_color="red",
                    annotation_text=f"{_tr('Threshold')} (0.78)",
                )
                st.plotly_chart(fig_perf, use_container_width=True)

        with col_time:
            if "timestamp" in mlflow_runs.columns and "auc" in mlflow_runs.columns:
                fig_timeline = px.scatter(
                    mlflow_runs,
                    x="timestamp",
                    y="auc",
                    color="model_type" if "model_type" in mlflow_runs.columns else None,
                    size="training_time_s" if "training_time_s" in mlflow_runs.columns else None,
                    render_mode="svg",
                    title=_tr("Model Performance Over Time"),
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

        # Run details table
        st.markdown(f"#### {_tr('Run Details')}")
        display_cols = [
            c for c in [
                "run_id", "model_type", "auc", "precision", "recall",
                "f1_score", "training_time_s", "timestamp",
            ]
            if c in mlflow_runs.columns
        ]
        st.dataframe(mlflow_runs[display_cols], use_container_width=True)


def _render_model_health(st, config: Dict, data_loader):
    """Render model health and drift detection summary."""
    st.subheader(_tr("Model Health & Drift Detection"))

    drift_history = data_loader.load_drift_history()
    model_metrics = data_loader.load_model_metrics()

    col_drift, col_perf = st.columns(2)

    with col_drift:
        if not drift_history.empty:
            latest = drift_history.iloc[-1]
            alert_level = latest.get("alert_level", "unknown")

            if alert_level == "green":
                st.success(
                    f"{_tr('Current Drift Status')}: **{alert_level.upper()}**"
                )
            elif alert_level == "yellow":
                st.warning(
                    f"{_tr('Current Drift Status')}: **{alert_level.upper()}**"
                )
            else:
                st.error(
                    f"{_tr('Current Drift Status')}: **{alert_level.upper()}**"
                )

            # Recent drift trend
            fig_drift = go.Figure()
            fig_drift.add_trace(go.Scatter(
                x=drift_history["timestamp"],
                y=drift_history["psi_mean"],
                mode="lines+markers",
                name=_tr("Mean PSI"),
                line=dict(color="#3498db", width=2),
            ))

            drift_cfg = config.get("drift_detection", {})
            fig_drift.add_hline(
                y=drift_cfg.get("yellow_threshold", 0.10),
                line_dash="dash", line_color="#f39c12",
            )
            fig_drift.add_hline(
                y=drift_cfg.get("red_threshold", 0.25),
                line_dash="dash", line_color="#e74c3c",
            )
            fig_drift.update_layout(
                title=_tr("PSI Drift Trend"),
                xaxis_title=_tr("Date"),
                yaxis_title=_tr("Mean PSI"),
                height=300,
            )
            st.plotly_chart(fig_drift, use_container_width=True)
        else:
            st.info(_tr("No drift detection history available."))

    with col_perf:
        if model_metrics:
            st.markdown(f"#### {_tr('Current Model Performance')}")
            metrics_rows = []
            for model_name, metrics in model_metrics.items():
                metrics_rows.append({
                    _tr("Model"): model_name,
                    _tr("AUC"): metrics.get("auc", 0),
                    _tr("Precision"): metrics.get("precision", 0),
                    _tr("Recall"): metrics.get("recall", 0),
                    _tr("F1"): metrics.get("f1_score", 0),
                })
            metrics_df = pd.DataFrame(metrics_rows)
            st.dataframe(metrics_df, use_container_width=True)

            # Best model highlight
            best = max(
                model_metrics.keys(),
                key=lambda m: model_metrics[m].get("auc", 0),
            )
            st.info(
                f"{_tr('Best model')}: **{best}** "
                f"(AUC = {model_metrics[best].get('auc', 0):.4f})"
            )
        else:
            st.info(_tr("No model performance metrics available."))


def _render_system_config(st, config: Dict):
    """Render system configuration summary."""
    st.subheader(_tr("System Configuration"))

    with st.expander(_tr("Full Configuration"), expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{_tr('Simulation')}**")
            sim = config.get("simulation", {})
            st.json({
                "n_customers": sim.get("n_customers", 20000),
                "horizon_months": sim.get("horizon_months", 12),
                "random_seed": sim.get("random_seed", 42),
            })

            st.markdown(f"**{_tr('Budget')}**")
            budget = config.get("budget", {})
            st.json({
                "total_krw": budget.get("total_krw", 50000000),
                "currency": budget.get("currency", "KRW"),
            })

        with col2:
            st.markdown(f"**{_tr('ML Model')}**")
            ml = config.get("ml_model", {})
            st.json({
                "n_splits": ml.get("n_splits", 5),
                "early_stopping_rounds": ml.get("early_stopping_rounds", 10),
            })

            st.markdown(f"**{_tr('DL Model')}**")
            dl = config.get("dl_model", {})
            st.json({
                "architecture": dl.get("architecture", "transformer"),
                "hidden_size": dl.get("hidden_size", 64),
                "num_layers": dl.get("num_layers", 2),
                "sequence_window": dl.get("sequence_window", 6),
            })
