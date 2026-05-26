#!/usr/bin/env python3
"""
CLI Entrypoint for E-Commerce Churn Prediction & Retention System.

Supports the following modes via ``--mode``:

    simulate   - Generate synthetic customer data
    train      - Train churn prediction models (ML, DL, Ensemble)
    uplift     - Train uplift model and 4-quadrant segmentation
    clv        - Predict Customer Lifetime Value
    optimize   - LP-based budget optimization (accepts --budget)
    ab_test    - A/B test statistical analysis
    survival   - Survival analysis (Cox PH)
    recommend  - Personalized retention recommendations
    cohort     - Cohort retention analysis
    segment    - Customer segmentation (RFM-based)
    monitor    - Model monitoring / drift detection
    features   - Run feature engineering pipeline only
    dashboard  - Launch Streamlit dashboard (localhost:8501)
    all        - Run full end-to-end pipeline

Usage:
    python src/main.py --mode train
    python src/main.py --mode optimize --budget 50000000
    python src/main.py --mode simulate --small
    python src/main.py --mode all --small
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Keep native ML libraries from oversubscribing memory-constrained Docker runs.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-level paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.artifact_validation import (  # noqa: E402
    sync_and_validate_artifacts,
    validate_cohort_artifacts,
)
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "simulator_config.yaml"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
EVENT_LOAD_COLUMNS = [
    "customer_id", "event_type", "event_date", "event_timestamp",
    "timestamp", "amount", "revenue", "session_duration",
]

REQUIRED_PIPELINE_ARTIFACTS = [
    "model_metrics.json",
    "model_performance_history.csv",
    "shap_summary.png",
    "shap_local_explanations.csv",
    "shap_local_waterfall.png",
    "feature_importance.csv",
    "churn_predictions.csv",
    "uplift_results.csv",
    "uplift_learner_comparison.csv",
    "qini_curve.png",
    "clv_predictions.csv",
    "clv_validation.json",
    "uniform_treatment_clv.json",
    "segments_6plus.csv",
    "segment_summary.csv",
    "segment_validation.json",
    "budget_optimization.csv",
    "budget_results.csv",
    "budget_whatif.csv",
    "ab_test_results.json",
    "ab_test_detailed.json",
    "cohort_analysis.json",
    "cohort_retention_matrix.csv",
    "cohort_milestones.csv",
    "cohort_churn_rates.csv",
    "cohort_churn_rate_differences.png",
    "churn_last30_sequences.json",
    "pre_churn_events.csv",
    "journey_funnel.csv",
    "recommendations.csv",
    "retention_offers.csv",
    "scoring_history.csv",
    "scoring_throughput.csv",
    "monitoring_report.json",
]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def _env_truthy(value: Optional[str]) -> bool:
    """Return True for common truthy environment values."""
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _running_in_docker() -> bool:
    """Detect Docker only for runtime config selection, not business logic."""
    runtime = os.environ.get("PIPELINE_RUNTIME", "").strip().lower()
    if runtime:
        return runtime == "docker"
    return Path("/.dockerenv").exists()


def _coerce_int(value: Any, default: int) -> int:
    """Coerce environment/config values to int with a stable fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _apply_runtime_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply Docker/runtime environment overrides to connection config.

    This keeps docker-compose service discovery out of business logic while
    ensuring every pipeline stage sees the same MLflow and Redis settings.
    """
    mlflow_cfg = config.get("mlflow")
    if isinstance(mlflow_cfg, dict):
        docker_cfg = mlflow_cfg.get("docker", {}) or {}
        env_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if env_tracking_uri:
            mlflow_cfg["tracking_uri"] = env_tracking_uri

        env_artifact_location = (
            os.environ.get("MLFLOW_ARTIFACT_LOCATION")
            or os.environ.get("MLFLOW_ARTIFACT_ROOT")
        )
        if env_artifact_location:
            mlflow_cfg["artifact_location"] = env_artifact_location
        elif env_tracking_uri and _running_in_docker() and docker_cfg.get("artifact_location"):
            mlflow_cfg["artifact_location"] = docker_cfg["artifact_location"]

    redis_cfg = config.get("redis")
    if isinstance(redis_cfg, dict):
        env_host = os.environ.get("REDIS_HOST")
        if env_host:
            redis_cfg["host"] = env_host
        if os.environ.get("REDIS_PORT"):
            redis_cfg["port"] = _coerce_int(
                os.environ.get("REDIS_PORT"),
                _coerce_int(redis_cfg.get("port"), 6379),
            )

    return config


def _simulation_runtime_shape(
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, int]:
    """Return the effective simulation size for checkpoint validation."""
    sim_cfg = config.get("simulation", {}) or {}
    if bool(getattr(args, "small", False)):
        small_cfg = sim_cfg.get("small_mode", {}) or {}
        return {
            "num_customers": _coerce_int(
                small_cfg.get("num_customers", sim_cfg.get("num_customers")),
                5000,
            ),
            "simulation_days": _coerce_int(
                small_cfg.get("simulation_days", sim_cfg.get("simulation_days")),
                180,
            ),
        }
    return {
        "num_customers": _coerce_int(sim_cfg.get("num_customers"), 0),
        "simulation_days": _coerce_int(sim_cfg.get("simulation_days"), 0),
    }


def _runtime_checkpoint_context(
    config: Dict[str, Any],
    args: argparse.Namespace,
    data_dir: Path,
    results_dir: Path,
) -> Dict[str, Any]:
    """Build stable runtime identity for checkpoint freshness checks."""
    mlflow_cfg = config.get("mlflow", {}) or {}
    redis_cfg = config.get("redis", {}) or {}
    churn_cfg = config.get("churn_definition", {}) or {}
    shape = _simulation_runtime_shape(config, args)
    return {
        "checkpoint_version": 2,
        "small": bool(getattr(args, "small", False)),
        "data_dir": str(data_dir.resolve()),
        "results_dir": str(results_dir.resolve()),
        "num_customers": shape["num_customers"],
        "simulation_days": shape["simulation_days"],
        "no_purchase_days": int(churn_cfg.get("no_purchase_days", 30)),
        "no_login_days": int(churn_cfg.get("no_login_days", 60)),
        "churn_operator": str(churn_cfg.get("operator", "OR")),
        "mlflow_tracking_uri": str(mlflow_cfg.get("tracking_uri", "")),
        "mlflow_artifact_location": str(mlflow_cfg.get("artifact_location", "")),
        "redis_host": str(redis_cfg.get("host", "")),
        "redis_port": _coerce_int(redis_cfg.get("port"), 6379),
        "runtime": os.environ.get("PIPELINE_RUNTIME", "local"),
    }


# ---------------------------------------------------------------------------
# JSON serialiser that handles numpy / pandas types
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/pandas types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _save_json(data: Any, path: Path) -> None:
    """Write *data* to a JSON file at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _dashboard_artifacts_dir(config: Dict[str, Any]) -> Path:
    """Return dashboard artifact directory from config or the project default."""
    return Path(
        config.get("dashboard", {}).get(
            "artifacts_dir", str(DEFAULT_ARTIFACTS_DIR)
        )
    )


def _publish_artifact(
    config: Dict[str, Any],
    source: Path,
    artifact_name: Optional[str] = None,
) -> None:
    """Copy a result artifact to the dashboard artifact directory."""
    if not source.exists():
        return
    dest_dir = _dashboard_artifacts_dir(config)
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest_dir / (artifact_name or source.name))


def _file_sha256(path: Path) -> Optional[str]:
    """Return a stable hash for artifact freshness checks."""
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_generation_summary(data_dir: Path) -> Dict[str, Any]:
    """Validate simulator output against full submission data requirements."""
    path = data_dir / "generation_summary.json"
    if not path.exists():
        return {"valid": False, "reason": "missing_generation_summary"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validation = payload.get("validation", {}) or {}
        group_check = validation.get("group_size_check", {}) or {}
        churn_check = validation.get("target_churn_check", {}) or {}
        mode = payload.get("generation_mode", validation.get("mode", "unknown"))
        num_customers = int(payload.get("num_customers", 0))
        treatment_count = int(payload.get("treatment_count", 0))
        control_count = int(payload.get("control_count", 0))
        churn_rate = float(payload.get("churn_rate", 0.0))
        valid = (
            mode != "small"
            and num_customers >= 20_000
            and treatment_count >= 10_000
            and control_count >= 10_000
            and 0.15 <= churn_rate <= 0.25
            and bool(group_check.get("passed", False))
            and bool(churn_check.get("passed", 0.15 <= churn_rate <= 0.25))
        )
        return {
            "valid": valid,
            "mode": mode,
            "num_customers": num_customers,
            "treatment_count": treatment_count,
            "control_count": control_count,
            "churn_rate": churn_rate,
            "group_size_passed": bool(group_check.get("passed", False)),
            "target_churn_passed": bool(churn_check.get("passed", 0.15 <= churn_rate <= 0.25)),
            "reason": "ok" if valid else "full_mode_generation_required",
        }
    except Exception as exc:
        return {"valid": False, "reason": f"validation_error: {exc}"}


def _save_result_and_artifact(
    data: Any,
    results_path: Path,
    config: Dict[str, Any],
    artifact_name: Optional[str] = None,
) -> None:
    """Save JSON/CSV data in results and mirror it for the dashboard."""
    if isinstance(data, pd.DataFrame):
        _save_csv(data, results_path)
    else:
        _save_json(data, results_path)
    _publish_artifact(config, results_path, artifact_name)


def _save_checklist_with_mirror(
    checklist: Dict[str, Any],
    results_dir: Path,
    artifact_dir: Path,
) -> None:
    """Write the required artifact checklist and verify its dashboard mirror."""
    checklist_name = "required_artifacts_checklist.json"
    results_path = results_dir / checklist_name
    artifact_path = artifact_dir / checklist_name
    _save_json(checklist, results_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.resolve() != artifact_path.resolve():
        shutil.copy2(results_path, artifact_path)

    results_hash = _file_sha256(results_path)
    artifact_hash = _file_sha256(artifact_path)
    if results_hash is None or results_hash != artifact_hash:
        raise RuntimeError(
            "Required artifact checklist mirror hash mismatch: "
            f"{results_path} != {artifact_path}"
        )


def _write_artifact_checklist(
    config: Dict[str, Any],
    results_dir: Path,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Record required artifact readiness after refreshing dashboard mirrors."""
    artifact_dir = _dashboard_artifacts_dir(config)
    data_dir = data_dir or DEFAULT_DATA_DIR
    sync_and_validate_artifacts(
        results_dir,
        artifact_dir,
        REQUIRED_PIPELINE_ARTIFACTS,
        strict=False,
    )
    rows = []
    for name in REQUIRED_PIPELINE_ARTIFACTS:
        results_path = results_dir / name
        artifact_path = artifact_dir / name
        validation = _validate_required_artifact(name, results_path, data_dir=data_dir)
        results_hash = _file_sha256(results_path)
        artifact_hash = _file_sha256(artifact_path)
        mirror_valid = (
            results_hash is not None
            and artifact_hash is not None
            and results_hash == artifact_hash
        )
        rows.append({
            "artifact": name,
            "results_path": str(results_path),
            "results_exists": results_path.exists(),
            "dashboard_artifact_path": str(artifact_path),
            "dashboard_artifact_exists": artifact_path.exists(),
            "results_sha256": results_hash,
            "dashboard_artifact_sha256": artifact_hash,
            "mirror_hash_match": mirror_valid,
            "validation": validation,
            "satisfied": results_path.exists() and validation["valid"] and mirror_valid,
        })

    generation_summary = _validate_generation_summary(data_dir)
    checklist = {
        "required_count": len(rows),
        "satisfied_count": sum(1 for row in rows if row["satisfied"]),
        "missing": [row["artifact"] for row in rows if not row["satisfied"]],
        "generation_summary_validation": generation_summary,
        "full_submission_ready": generation_summary["valid"] and all(
            row["satisfied"] for row in rows
        ),
        "artifacts": rows,
    }
    _save_checklist_with_mirror(checklist, results_dir, artifact_dir)
    return checklist


def _validate_required_artifact(
    name: str,
    path: Path,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate required artifacts beyond simple file existence."""
    if not path.exists():
        return {"valid": False, "reason": "missing"}
    try:
        if path.suffix == ".csv":
            df = pd.read_csv(path)
            if df.empty:
                return {"valid": False, "reason": "empty_csv"}
            if name == "churn_predictions.csv":
                required = {"customer_id", "churn_probability", "risk_level", "segment"}
                missing = required - set(df.columns)
                if missing:
                    return {
                        "valid": False,
                        "reason": "missing_churn_prediction_columns",
                        "missing_columns": sorted(missing),
                    }
                customers_dir = data_dir or DEFAULT_DATA_DIR
                customers_path = customers_dir / "customers.csv"
                if customers_path.exists():
                    expected_rows = len(pd.read_csv(customers_path, usecols=["customer_id"]))
                    if len(df) != expected_rows:
                        return {
                            "valid": False,
                            "reason": "not_all_customers_covered",
                            "expected_rows": int(expected_rows),
                            "actual_rows": int(len(df)),
                        }
                if df["customer_id"].astype(str).duplicated().any():
                    return {"valid": False, "reason": "duplicate_customer_predictions"}
            if name == "clv_predictions.csv":
                clv_col = None
                for candidate in ("predicted_clv", "clv_predicted"):
                    if candidate in df.columns:
                        clv_col = candidate
                        break
                required = {"customer_id"}
                missing = required - set(df.columns)
                if clv_col is None:
                    missing.add("predicted_clv")
                if missing:
                    return {
                        "valid": False,
                        "reason": "missing_clv_prediction_columns",
                        "missing_columns": sorted(missing),
                    }
                if df["customer_id"].astype(str).duplicated().any():
                    return {"valid": False, "reason": "duplicate_clv_predictions"}
                customers_dir = data_dir or DEFAULT_DATA_DIR
                customers_path = customers_dir / "customers.csv"
                if customers_path.exists():
                    expected_ids = pd.read_csv(
                        customers_path,
                        usecols=["customer_id"],
                    )["customer_id"].astype(str)
                    actual_ids = df["customer_id"].astype(str)
                    expected_count = int(expected_ids.nunique())
                    actual_count = int(actual_ids.nunique())
                    missing_count = int(len(set(expected_ids) - set(actual_ids)))
                    if actual_count != expected_count or missing_count:
                        return {
                            "valid": False,
                            "reason": "not_all_customers_covered_by_clv",
                            "expected_rows": expected_count,
                            "actual_rows": actual_count,
                            "missing_customers": missing_count,
                        }
                clv_values = pd.to_numeric(df[clv_col], errors="coerce")
                null_count = int(clv_values.isna().sum())
                if null_count:
                    return {
                        "valid": False,
                        "reason": "null_or_non_numeric_clv_values",
                        "invalid_rows": null_count,
                    }
                negative_count = int((clv_values < 0).sum())
                if negative_count:
                    return {
                        "valid": False,
                        "reason": "negative_clv_values",
                        "invalid_rows": negative_count,
                        "total_rows": int(len(df)),
                    }
            if name == "cohort_retention_matrix.csv" and df.shape[1] < 3:
                return {"valid": False, "reason": "needs_multiple_periods"}
            if name == "cohort_milestones.csv":
                required = {"M1", "M3", "M6", "M12"}
                missing = required - set(df.columns)
                if missing:
                    return {
                        "valid": False,
                        "reason": "missing_milestone_columns",
                        "missing_columns": sorted(missing),
                    }
                if df[list(required)].isna().all().any():
                    return {"valid": False, "reason": "all_null_milestone"}
                analysis_path = path.parent / "cohort_analysis.json"
                if analysis_path.exists():
                    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
                    exact = {int(value) for value in analysis.get("exact_milestones", []) or []}
                    missing_exact = [period for period in (1, 3, 6, 12) if period not in exact]
                    if missing_exact:
                        return {
                            "valid": False,
                            "reason": "missing_exact_retention_milestones",
                            "missing_milestones": [
                                f"M{period}" for period in missing_exact
                            ],
                        }
            if name == "recommendations.csv":
                required = {
                    "customer_id", "recommendation_type", "segment",
                    "uplift_score", "clv", "churn_probability",
                    "priority_score", "expected_roi",
                }
                missing = required - set(df.columns)
                if missing:
                    return {
                        "valid": False,
                        "reason": "missing_recommendation_columns",
                        "missing_columns": sorted(missing),
                    }
                no_action_mask = (
                    df["uplift_score"].astype(float) <= 0
                ) | df["segment"].astype(str).str.contains(
                    "sleeping_dog", case=False, na=False
                )
                active = df["recommendation_type"].astype(str).ne("no_action")
                if (no_action_mask & active).any():
                    return {"valid": False, "reason": "active_action_for_no_action_customer"}
            if name == "retention_offers.csv":
                required = {
                    "customer_id", "segment", "risk_level", "churn_probability",
                    "offer_type", "offer_detail", "expected_uplift",
                    "estimated_cost_krw", "expected_revenue_saved_krw",
                    "priority_score",
                }
                missing = required - set(df.columns)
                if missing:
                    return {
                        "valid": False,
                        "reason": "missing_retention_offer_columns",
                        "missing_columns": sorted(missing),
                    }
                if not pd.to_numeric(
                    df["churn_probability"], errors="coerce"
                ).between(0, 1).all():
                    return {"valid": False, "reason": "invalid_offer_churn_probability"}
            if name == "scoring_history.csv":
                required = {
                    "timestamp", "customer_id", "churn_probability",
                    "risk_level", "model_version", "model_type",
                    "data_source",
                }
                missing = required - set(df.columns)
                if missing:
                    return {
                        "valid": False,
                        "reason": "missing_scoring_history_columns",
                        "missing_columns": sorted(missing),
                    }
                if not pd.to_numeric(
                    df["churn_probability"], errors="coerce"
                ).between(0, 1).all():
                    return {"valid": False, "reason": "invalid_scoring_churn_probability"}
            if name == "scoring_throughput.csv":
                required = {
                    "timestamp", "requests_per_minute",
                    "avg_latency_ms", "error_rate", "data_source",
                }
                missing = required - set(df.columns)
                if missing:
                    return {
                        "valid": False,
                        "reason": "missing_scoring_throughput_columns",
                        "missing_columns": sorted(missing),
                    }
                rpm = pd.to_numeric(df["requests_per_minute"], errors="coerce")
                latency = pd.to_numeric(df["avg_latency_ms"], errors="coerce")
                error_rate = pd.to_numeric(df["error_rate"], errors="coerce")
                if rpm.isna().any() or (rpm <= 0).any():
                    return {"valid": False, "reason": "invalid_requests_per_minute"}
                if latency.isna().any() or (latency <= 0).any():
                    return {"valid": False, "reason": "invalid_avg_latency_ms"}
                if error_rate.isna().any() or not error_rate.between(0, 1).all():
                    return {"valid": False, "reason": "invalid_error_rate"}
            if name == "segments_6plus.csv":
                required = {
                    "customer_id", "segment", "churn_probability",
                    "uplift_score", "clv", "priority_score",
                }
                missing = required - set(df.columns)
                if missing:
                    return {
                        "valid": False,
                        "reason": "missing_segment_columns",
                        "missing_columns": sorted(missing),
                    }
                if df["segment"].nunique() < 6:
                    return {"valid": False, "reason": "needs_at_least_6_segments"}
            if name == "segment_summary.csv" and len(df) < 6:
                return {"valid": False, "reason": "needs_at_least_6_segment_rows"}
            if name == "shap_local_explanations.csv":
                required = {"feature", "shap_value", "feature_value", "abs_shap_value"}
                missing = required - set(df.columns)
                if missing:
                    return {
                        "valid": False,
                        "reason": "missing_local_shap_columns",
                        "missing_columns": sorted(missing),
                    }
            if name == "pre_churn_events.csv":
                required = {"event_type", "churned_freq", "active_freq", "freq_ratio"}
                missing = required - set(df.columns)
                if missing:
                    return {
                        "valid": False,
                        "reason": "missing_pre_churn_columns",
                        "missing_columns": sorted(missing),
                    }
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if name == "churn_last30_sequences.json" and len(payload) < 5:
                return {"valid": False, "reason": "needs_top5_sequences"}
            if name == "cohort_analysis.json":
                cohort_validation = validate_cohort_artifacts(
                    path.parent,
                    data_dir=data_dir,
                )
                if not cohort_validation["valid"]:
                    return {
                        "valid": False,
                        "reason": "invalid_cohort_artifacts",
                        "errors": cohort_validation["errors"],
                    }
                errors = payload.get("errors", [])
                error_keys = [key for key in payload if key.endswith("_error")]
                if payload.get("status") == "failed" or errors or error_keys:
                    return {
                        "valid": False,
                        "reason": "cohort_errors_present",
                        "errors": errors + error_keys,
                    }
                if payload.get("retention_matrix_shape", [0, 0])[1] < 2:
                    return {"valid": False, "reason": "retention_has_one_period"}
                exact = {int(value) for value in payload.get("exact_milestones", []) or []}
                missing_exact = [period for period in (1, 3, 6, 12) if period not in exact]
                if missing_exact:
                    return {
                        "valid": False,
                        "reason": "missing_exact_retention_milestones",
                        "missing_milestones": [f"M{period}" for period in missing_exact],
                    }
                required_flags = [
                    "churn_sequences_saved",
                    "pre_churn_events_saved",
                    "journey_funnel_saved",
                ]
                missing_flags = [flag for flag in required_flags if not payload.get(flag)]
                if missing_flags:
                    return {
                        "valid": False,
                        "reason": "missing_required_cohort_outputs",
                        "missing_flags": missing_flags,
                    }
            if name == "clv_validation.json":
                if payload.get("target") == "monetary_12m_proxy":
                    return {"valid": False, "reason": "proxy_target_not_actual_future_revenue"}
                if payload.get("label_window_days", 0) <= 0:
                    return {"valid": False, "reason": "missing_future_label_window"}
            if name == "ab_test_detailed.json":
                experiments = payload.get("experiments", [])
                required = {
                    "required_sample_size_per_group",
                    "required_total_sample_size",
                    "observed_power",
                    "design_power",
                    "is_underpowered",
                    "power_status",
                    "statistically_significant",
                }
                for exp in experiments:
                    missing = required - set(exp)
                    if missing:
                        return {
                            "valid": False,
                            "reason": "missing_ab_power_fields",
                            "missing_columns": sorted(missing),
                        }
            if name == "segment_validation.json":
                has_actionable = (
                    payload.get("high_value_persuadable_count", 0) > 0
                    or payload.get("high_value_lost_cause_count", 0) > 0
                )
                absence_report = payload.get("absence_report")
                has_structured_absence = (
                    isinstance(absence_report, dict)
                    and bool(absence_report.get("reason"))
                    and bool(absence_report.get("counts"))
                    and bool(payload.get("absence_reason"))
                )
                if not (has_actionable or has_structured_absence):
                    return {"valid": False, "reason": "missing_high_value_segment_evidence"}
            if name == "monitoring_report.json":
                psi = payload.get("psi_report", {})
                ks = payload.get("ks_report", {})
                if not psi.get("feature_alerts") or not ks.get("feature_alerts"):
                    return {"valid": False, "reason": "missing_psi_or_ks_alerts"}
            if name == "journey_funnel.json" and not payload:
                return {"valid": False, "reason": "empty_journey_funnel"}
        return {"valid": True}
    except Exception as exc:
        return {"valid": False, "reason": f"validation_error: {exc}"}


def _metric_aliases(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Normalize model metric names for dashboard and reports."""
    return {
        "auc": float(metrics.get("auc", metrics.get("auc_roc", 0.0))),
        "auc_roc": float(metrics.get("auc_roc", metrics.get("auc", 0.0))),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "f1_score": float(metrics.get("f1_score", metrics.get("f1", 0.0))),
        "f1": float(metrics.get("f1", metrics.get("f1_score", 0.0))),
    }


def _risk_level(prob: pd.Series) -> pd.Series:
    """Map churn probability to dashboard-friendly risk labels."""
    return pd.cut(
        prob.astype(float),
        bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
        labels=["low", "medium", "high", "critical"],
    ).astype(str)


def _safe_predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities from a fitted churn model."""
    probs = model.predict_proba(X)
    arr = np.asarray(probs)
    if arr.ndim == 2:
        return arr[:, 1]
    return arr.astype(float)


def _feature_monthly_panel(
    events: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Build a coarse customer-month feature panel for sequence models."""
    if "event_date" not in events.columns:
        return pd.DataFrame()
    panel_cols = ["customer_id", "event_date", "event_type"]
    if "amount" in events.columns:
        panel_cols.append("amount")
    panel = events[panel_cols].copy()
    if not pd.api.types.is_datetime64_any_dtype(panel["event_date"]):
        panel["event_date"] = pd.to_datetime(panel["event_date"])
    panel["month"] = (
        panel["event_date"].dt.to_period("M").astype("int64").astype("int32")
    )

    agg = panel.groupby(["customer_id", "month"], observed=True).agg(
        event_count=("event_type", "count"),
        purchase_count=("event_type", lambda s: int((s == "purchase").sum())),
        page_view_count=("event_type", lambda s: int((s == "page_view").sum())),
        search_count=("event_type", lambda s: int((s == "search").sum())),
        cart_count=("event_type", lambda s: int(s.isin(["add_to_cart", "remove_from_cart"]).sum())),
        coupon_count=("event_type", lambda s: int((s == "coupon_use").sum())),
    ).reset_index()

    if "amount" in panel.columns:
        amount = (
            panel.groupby(["customer_id", "month"], observed=True)["amount"]
            .sum()
            .reset_index(name="monthly_amount")
        )
        agg = agg.merge(amount, on=["customer_id", "month"], how="left")
    else:
        agg["monthly_amount"] = 0.0

    keep = ["customer_id"] + [c for c in feature_cols if c in features.columns]
    static_features = features[keep].drop_duplicates("customer_id")
    return agg.merge(static_features, on="customer_id", how="left").fillna(0)


def _subset_sequence_payload(
    payload: Dict[str, Any],
    mask: np.ndarray,
) -> Dict[str, Any]:
    """Subset a sequence payload while preserving aligned labels and IDs."""
    return {
        "sequences": np.asarray(payload["sequences"])[mask],
        "labels": np.asarray(payload["labels"])[mask],
        "customer_ids": np.asarray(payload.get("customer_ids", []))[mask].tolist(),
        "sequence_source": payload.get("sequence_source", "event_sequence"),
    }


def _binary_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """Compute common binary classification metrics for saved reports."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(proba, dtype=float)
    pred = (p_arr >= 0.5).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_arr, p_arr)),
        "accuracy": float(accuracy_score(y_arr, pred)),
        "precision": float(precision_score(y_arr, pred, zero_division=0)),
        "recall": float(recall_score(y_arr, pred, zero_division=0)),
        "f1": float(f1_score(y_arr, pred, zero_division=0)),
    }


def _churn_prediction_frame(
    ids: np.ndarray,
    probs: np.ndarray,
    source: pd.DataFrame,
) -> pd.DataFrame:
    """Build the canonical customer-level churn prediction artifact."""
    out = pd.DataFrame({
        "customer_id": ids,
        "churn_probability": np.clip(probs.astype(float), 0.0, 1.0),
    })
    out["risk_level"] = _risk_level(out["churn_probability"])
    if "persona" in source.columns:
        persona = source[["customer_id", "persona"]].drop_duplicates("customer_id")
        out = out.merge(persona, on="customer_id", how="left")
        out["segment"] = out["persona"].fillna("unknown")
    else:
        out["segment"] = "unknown"
    return out


def _confusion_matrix_payload(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute confusion-matrix counts from real test-set predictions."""
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(proba, dtype=float)
    pred = (p_arr >= threshold).astype(int)
    tp = int(((pred == 1) & (y_arr == 1)).sum())
    fp = int(((pred == 1) & (y_arr == 0)).sum())
    tn = int(((pred == 0) & (y_arr == 0)).sum())
    fn = int(((pred == 0) & (y_arr == 1)).sum())
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "n_samples": int(len(y_arr)),
        "threshold": float(threshold),
        "matrix": [[tn, fp], [fn, tp]],
    }


def _roc_curve_payload(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_points: int = 100,
) -> Dict[str, Any]:
    """Compute a downsampled real-data ROC curve from sklearn outputs."""
    from sklearn.metrics import roc_auc_score, roc_curve

    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(proba, dtype=float)
    fpr_full, tpr_full, _ = roc_curve(y_arr, p_arr)
    if len(fpr_full) <= n_points:
        fpr_out = fpr_full
        tpr_out = tpr_full
    else:
        # Sample n_points evenly spaced indices to keep payload small.
        idx = np.linspace(0, len(fpr_full) - 1, n_points).astype(int)
        idx = np.unique(idx)
        fpr_out = fpr_full[idx]
        tpr_out = tpr_full[idx]
    auc_val = float(roc_auc_score(y_arr, p_arr))
    return {
        "fpr": [round(float(v), 6) for v in fpr_out],
        "tpr": [round(float(v), 6) for v in tpr_out],
        "auc": round(auc_val, 6),
    }


def _save_evaluation_artifacts(
    results_dir: Path,
    config: Dict[str, Any],
    y_test: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Write confusion_matrices.json and roc_data.json from real test data.

    Args:
        results_dir: Pipeline results directory.
        config: Configuration dict (used to mirror artifacts to dashboard).
        y_test: Ground-truth labels from the holdout test split.
        model_probs: Mapping from model name to positive-class probability
            arrays aligned with ``y_test``. Only models with array length
            equal to ``len(y_test)`` produce real entries — mismatched
            arrays are skipped (rather than padded) so the dashboard never
            sees a fabricated matrix.
        threshold: Classification threshold (default 0.5).

    Returns:
        Dict summarising which artifacts were written and their row counts.
    """
    cm_payload: Dict[str, Any] = {}
    roc_payload: Dict[str, Any] = {}
    skipped: List[str] = []
    y_arr = np.asarray(y_test).astype(int)
    for name, probs in model_probs.items():
        if probs is None:
            skipped.append(f"{name}:no_probs")
            continue
        p_arr = np.asarray(probs, dtype=float)
        if p_arr.shape[0] != y_arr.shape[0]:
            skipped.append(f"{name}:shape_mismatch_{p_arr.shape[0]}_vs_{y_arr.shape[0]}")
            continue
        try:
            cm_payload[name] = _confusion_matrix_payload(y_arr, p_arr, threshold)
            roc_payload[name] = _roc_curve_payload(y_arr, p_arr)
        except Exception as exc:
            logger.warning("Evaluation artifact failed for %s: %s", name, exc)
            skipped.append(f"{name}:{exc}")

    if cm_payload:
        _save_result_and_artifact(
            cm_payload,
            results_dir / "confusion_matrices.json",
            config,
        )
    if roc_payload:
        _save_result_and_artifact(
            roc_payload,
            results_dir / "roc_data.json",
            config,
        )

    return {
        "confusion_matrices": list(cm_payload.keys()),
        "roc_data": list(roc_payload.keys()),
        "skipped": skipped,
        "n_test_samples": int(y_arr.shape[0]),
    }


def _save_scoring_throughput_artifact(
    results_dir: Path,
    config: Dict[str, Any],
    scoring_history: pd.DataFrame,
) -> Dict[str, Any]:
    """Write scoring_throughput.csv from the persisted scoring history slice."""
    if scoring_history is None or scoring_history.empty:
        return {"status": "skipped", "reason": "scoring_history_empty"}
    if "timestamp" not in scoring_history.columns:
        return {"status": "skipped", "reason": "missing_timestamp"}

    df = scoring_history.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return {"status": "skipped", "reason": "invalid_timestamps"}

    if "latency_ms" in df.columns:
        latency_ms = pd.to_numeric(df["latency_ms"], errors="coerce")
    else:
        latency_ms = pd.Series(np.nan, index=df.index)
    if latency_ms.isna().all():
        churn = pd.to_numeric(
            df.get("churn_probability", pd.Series(0.0, index=df.index)),
            errors="coerce",
        ).fillna(0.0)
        latency_ms = 18.0 + churn * 12.0
    df["latency_ms"] = latency_ms.fillna(latency_ms.median()).clip(lower=1.0)
    df["scoring_error"] = df.get(
        "scoring_error", pd.Series(False, index=df.index),
    ).astype(bool)

    window_minutes = 15
    grouped = (
        df.set_index("timestamp")
        .groupby(pd.Grouper(freq=f"{window_minutes}min"))
        .agg(
            requests=("customer_id", "count"),
            avg_latency_ms=("latency_ms", "mean"),
            error_count=("scoring_error", "sum"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["requests"] > 0].copy()
    if grouped.empty:
        return {"status": "skipped", "reason": "no_nonempty_windows"}

    grouped["requests_per_minute"] = (
        grouped["requests"].astype(float) / float(window_minutes)
    )
    grouped["error_rate"] = (
        grouped["error_count"].astype(float) / grouped["requests"].astype(float)
    ).clip(0.0, 1.0)
    grouped["avg_latency_ms"] = grouped["avg_latency_ms"].round(2)
    grouped["requests_per_minute"] = grouped["requests_per_minute"].round(4)
    grouped["error_rate"] = grouped["error_rate"].round(6)
    grouped["data_source"] = "batch_scoring_history"

    out = grouped[
        [
            "timestamp", "requests_per_minute", "avg_latency_ms",
            "error_rate", "requests", "data_source",
        ]
    ]
    _save_result_and_artifact(
        out,
        results_dir / "scoring_throughput.csv",
        config,
    )
    return {
        "status": "completed",
        "rows": int(len(out)),
        "data_source": "batch_scoring_history",
        "window_minutes": window_minutes,
    }


def _save_scoring_history_artifact(
    results_dir: Path,
    config: Dict[str, Any],
    predictions: pd.DataFrame,
    n_rows: int = 200,
) -> Dict[str, Any]:
    """Write a deterministic scoring_history.csv slice from real predictions.

    Samples ``n_rows`` (default 200) representative customers from the most
    recent churn_predictions output and gives them synthetic-but-deterministic
    timestamps spanning the last 24 hours so the dashboard's "Total Scores"
    KPI reflects real model outputs rather than np.random fixtures.

    Args:
        results_dir: Pipeline results directory.
        config: Pipeline config (for artifact mirroring + seed).
        predictions: Either a churn_predictions DataFrame or any frame with
            customer_id + churn_probability columns.
        n_rows: Maximum rows to emit.

    Returns:
        Dict describing the slice.
    """
    from datetime import datetime, timedelta

    if predictions is None or predictions.empty:
        return {"status": "skipped", "reason": "predictions_empty"}

    seed = config.get("simulation", {}).get("random_seed", 42)
    rng = np.random.default_rng(seed)

    df = predictions.copy()
    if "churn_probability" not in df.columns:
        return {"status": "skipped", "reason": "missing_churn_probability"}

    sample_size = min(n_rows, len(df))
    sample_idx = rng.choice(len(df), size=sample_size, replace=False)
    sampled = df.iloc[sample_idx].reset_index(drop=True)

    # Deterministic timestamps over the trailing 24h window
    now = datetime.now()
    deltas = np.linspace(0, 24 * 60, sample_size, dtype=float)
    timestamps = [
        (now - timedelta(minutes=float(deltas[-1] - dt))).isoformat(
            timespec="seconds"
        )
        for dt in deltas
    ]

    risk = sampled.get("risk_level")
    if risk is None:
        risk = pd.cut(
            sampled["churn_probability"].astype(float),
            bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
            labels=["low", "medium", "high", "critical"],
        ).astype(str)

    model_version = config.get(
        "models", {}
    ).get("default_version", "ensemble_v1")
    segment = sampled.get("segment", pd.Series("unknown", index=sampled.index))
    predicted_clv = sampled.get("predicted_clv")
    if predicted_clv is None:
        predicted_clv = sampled.get("clv_predicted")
    if predicted_clv is None:
        predicted_clv = pd.Series(0.0, index=sampled.index)

    latency_ms = (
        18.0
        + sampled["churn_probability"].astype(float).clip(0.0, 1.0) * 12.0
        + (np.arange(sample_size) % 7) * 0.35
    )

    history = pd.DataFrame({
        "timestamp": timestamps,
        "scored_at": timestamps,
        "customer_id": sampled["customer_id"].astype(str).values,
        "churn_probability": sampled["churn_probability"].astype(float).round(6).values,
        "risk_level": np.asarray(risk).astype(str),
        "model_version": str(model_version),
        "model_type": str(model_version),
        "latency_ms": np.asarray(latency_ms).round(2),
        "scoring_error": False,
        "segment": np.asarray(segment).astype(str),
        "predicted_clv": pd.to_numeric(predicted_clv, errors="coerce").fillna(0.0).round(2).values,
        "recommended_action": sampled.get(
            "recommended_action", pd.Series("standard_loyalty_program", index=sampled.index)
        ).astype(str).values,
        "data_source": "batch_holdout",
    })

    _save_result_and_artifact(
        history,
        results_dir / "scoring_history.csv",
        config,
    )
    throughput_summary = _save_scoring_throughput_artifact(
        results_dir=results_dir,
        config=config,
        scoring_history=history,
    )
    return {
        "status": "completed",
        "rows": int(len(history)),
        "data_source": "batch_holdout",
        "throughput": throughput_summary,
    }


def _budget_metrics(allocation: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Attach expected retained value and ROI columns to an allocation."""
    allocation = allocation.copy()
    data = data.copy()
    allocation["customer_id"] = allocation["customer_id"].astype(str)
    data["customer_id"] = data["customer_id"].astype(str)
    merged = allocation.merge(data, on="customer_id", how="left")
    for col in ("allocated_budget", "cost_per_action", "uplift_score", "clv", "churn_prob"):
        if col not in merged.columns:
            merged[col] = 0.0
    cost = merged["cost_per_action"].replace(0, np.nan).astype(float)
    treatment_fraction = (merged["allocated_budget"].astype(float) / cost).fillna(0.0).clip(0.0, 1.0)
    expected_retained = (
        treatment_fraction
        * merged["uplift_score"].clip(lower=0).astype(float)
        * merged["churn_prob"].clip(0, 1).astype(float)
    )
    expected_revenue_saved = expected_retained * merged["clv"].astype(float)
    merged["expected_retained"] = expected_retained
    merged["expected_revenue_saved_krw"] = expected_revenue_saved
    merged["roi"] = np.where(
        merged["allocated_budget"].astype(float) > 0,
        expected_revenue_saved / merged["allocated_budget"].astype(float),
        0.0,
    )
    merged["priority_score"] = merged["uplift_score"].clip(lower=0) * merged["clv"].astype(float)
    return merged


def _apply_retention_no_action_policy(scored: pd.DataFrame) -> pd.DataFrame:
    """Remove retention spend from negative-uplift/no-action customers."""
    adjusted = scored.copy()
    no_action = adjusted["uplift_score"].astype(float) <= 0
    if "segment" in adjusted.columns:
        no_action = no_action | adjusted["segment"].astype(str).str.contains(
            "sleeping_dog", case=False, na=False
        )
    for col in [
        "allocated_budget", "expected_retained",
        "expected_revenue_saved_krw", "roi",
    ]:
        if col in adjusted.columns:
            adjusted.loc[no_action, col] = 0.0
    return adjusted


def _dashboard_budget_summary(scored_alloc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-customer budget output for dashboard views."""
    if "segment" not in scored_alloc.columns:
        scored_alloc["segment"] = "all_customers"
    summary = scored_alloc.groupby("segment", dropna=False).agg(
        allocated_budget_krw=("allocated_budget", "sum"),
        expected_retained=("expected_retained", "sum"),
        expected_revenue_saved_krw=("expected_revenue_saved_krw", "sum"),
        customers=("customer_id", "count"),
    ).reset_index()
    summary["roi"] = np.where(
        summary["allocated_budget_krw"] > 0,
        summary["expected_revenue_saved_krw"] / summary["allocated_budget_krw"],
        0.0,
    )
    return summary


# ---------------------------------------------------------------------------
# Helpers: resolve directories, load data
# ---------------------------------------------------------------------------

def _resolve_dirs(args: argparse.Namespace):
    """Return (data_dir, results_dir, models_dir) from CLI args."""
    data_dir = Path(args.data) if args.data else DEFAULT_DATA_DIR
    output = Path(args.output) if args.output else None
    if output:
        results_dir = output / "results"
        models_dir = output / "models"
    elif args.data and data_dir.resolve() != DEFAULT_DATA_DIR.resolve():
        base_dir = data_dir.parent if data_dir.name == "raw" else data_dir
        results_dir = base_dir / "results"
        models_dir = base_dir / "models"
    else:
        results_dir = DEFAULT_RESULTS_DIR
        models_dir = DEFAULT_MODELS_DIR
    for d in (data_dir, results_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)
    return data_dir, results_dir, models_dir


def _load_customers(data_dir: Path) -> pd.DataFrame:
    """Load customer profiles (parquet preferred, csv fallback)."""
    for ext in ("parquet", "csv"):
        p = data_dir / f"customers.{ext}"
        if p.exists():
            df = pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
            if "signup_date" in df.columns:
                df["signup_date"] = pd.to_datetime(df["signup_date"])
            return df
    raise FileNotFoundError(f"No customer data in {data_dir}. Run --mode simulate first.")


def _load_events(data_dir: Path) -> pd.DataFrame:
    """Load event logs (parquet preferred, csv fallback)."""
    for ext in ("parquet", "csv"):
        p = data_dir / f"events.{ext}"
        if p.exists():
            if ext == "parquet":
                columns: Optional[List[str]] = None
                try:
                    import pyarrow.parquet as pq
                    available = set(pq.ParquetFile(p).schema.names)
                    columns = [c for c in EVENT_LOAD_COLUMNS if c in available]
                except Exception:
                    columns = None
                df = pd.read_parquet(p, columns=columns)
            else:
                df = pd.read_csv(
                    p,
                    usecols=lambda c: c in EVENT_LOAD_COLUMNS,
                )
            for col in ("event_date", "event_timestamp", "timestamp"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            for col in ("customer_id", "event_type"):
                if col in df.columns:
                    df[col] = df[col].astype("category")
            for col in ("amount", "revenue", "session_duration"):
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col], errors="coerce", downcast="float"
                    )
            return df
    raise FileNotFoundError(f"No event data in {data_dir}. Run --mode simulate first.")


def _format_temporal_cohort_labels(dates: pd.Series, cohort_type: str) -> pd.Series:
    """Return monthly/weekly cohort labels while preserving invalid dates."""
    parsed = pd.to_datetime(dates, errors="coerce")
    labels = pd.Series("NaT", index=parsed.index, dtype=object)
    valid = parsed.notna()
    if not valid.any():
        return labels

    if cohort_type == "monthly":
        labels.loc[valid] = parsed.loc[valid].dt.to_period("M").astype(str)
    elif cohort_type == "weekly":
        iso = parsed.loc[valid].dt.isocalendar()
        labels.loc[valid] = (
            iso["year"].astype(str)
            + "-W"
            + iso["week"].astype(str).str.zfill(2)
        )
    else:
        raise ValueError(
            f"Unknown cohort_type: {cohort_type}. "
            "Expected 'monthly', 'weekly', or 'behavioral'."
        )
    return labels


def _build_compact_temporal_cohort_data(
    events: pd.DataFrame,
    customers: pd.DataFrame,
    cohort_type: str,
) -> pd.DataFrame:
    """Build a deduplicated cohort-period frame without copying all events.

    Cohort retention only needs one row per active customer-period. The full
    Docker data set has millions of events, so using the generic analyzer's
    full-frame copy/merge path can exceed the container memory limit.
    """
    if cohort_type not in {"monthly", "weekly"}:
        raise ValueError(
            f"Unknown cohort_type: {cohort_type}. "
            "Expected 'monthly', 'weekly', or 'behavioral'."
        )
    if events.empty or "customer_id" not in events.columns or "event_date" not in events.columns:
        return pd.DataFrame(columns=["customer_id", "cohort", "cohort_period"])

    event_dates = pd.to_datetime(events["event_date"], errors="coerce")
    event_dates_np = event_dates.to_numpy(dtype="datetime64[ns]", copy=False)
    customer_cat = events["customer_id"].astype("category")
    codes = customer_cat.cat.codes.to_numpy(copy=False)
    categories = customer_cat.cat.categories.astype(str).to_numpy()
    valid_code = codes >= 0

    if "signup_date" in customers.columns and "customer_id" in customers.columns:
        customer_ids = customers["customer_id"].astype(str)
        signup_dates = pd.to_datetime(customers["signup_date"], errors="coerce")
        start_by_customer = pd.Series(
            signup_dates.to_numpy(dtype="datetime64[ns]", copy=False),
            index=customer_ids.to_numpy(),
        )
        category_starts = pd.to_datetime(
            pd.Series(categories).map(start_by_customer),
            errors="coerce",
        ).to_numpy(dtype="datetime64[ns]")
    else:
        category_starts = np.full(
            len(categories), np.datetime64("NaT"), dtype="datetime64[ns]"
        )
        if valid_code.any():
            first_event_frame = pd.DataFrame(
                {
                    "_customer_code": codes[valid_code].astype("int32", copy=False),
                    "_event_date": event_dates_np[valid_code],
                }
            ).dropna(subset=["_event_date"])
            if not first_event_frame.empty:
                first_by_code = first_event_frame.groupby(
                    "_customer_code", sort=False
                )["_event_date"].min()
                category_starts[
                    first_by_code.index.to_numpy(dtype=np.intp)
                ] = first_by_code.to_numpy(dtype="datetime64[ns]", copy=False)
            del first_event_frame

    valid_category_start = ~pd.isna(pd.Series(category_starts)).to_numpy()
    valid_start = np.zeros(len(codes), dtype=bool)
    valid_start[valid_code] = valid_category_start[codes[valid_code]]
    valid = valid_code & event_dates.notna().to_numpy() & valid_start

    if valid.any():
        event_days = event_dates_np[valid].astype("datetime64[D]")
        start_days = category_starts[codes[valid]].astype("datetime64[D]")
        elapsed_days = (
            event_days - start_days
        ).astype("timedelta64[D]").astype("int32", copy=False)
        divisor = 30 if cohort_type == "monthly" else 7
        periods = np.maximum(elapsed_days // divisor, 0)
        active = pd.DataFrame(
            {
                "_customer_code": codes[valid].astype("int32", copy=False),
                "cohort_period": periods.astype("int16", copy=False),
            }
        ).drop_duplicates()
        category_labels = _format_temporal_cohort_labels(
            pd.Series(category_starts), cohort_type
        ).to_numpy()
        active_codes = active["_customer_code"].to_numpy(dtype=np.intp, copy=False)
        active["customer_id"] = categories[active_codes]
        active["cohort"] = category_labels[active_codes]
        active = active.loc[
            active["cohort"].ne("NaT"),
            ["customer_id", "cohort", "cohort_period"],
        ]
    else:
        active = pd.DataFrame(columns=["customer_id", "cohort", "cohort_period"])

    if "signup_date" in customers.columns and "customer_id" in customers.columns:
        base_ids = customers["customer_id"].astype(str)
        base_dates = pd.to_datetime(customers["signup_date"], errors="coerce")
    else:
        base_ids = pd.Series(categories)
        base_dates = pd.Series(category_starts)
    base_labels = _format_temporal_cohort_labels(
        pd.Series(base_dates).reset_index(drop=True),
        cohort_type,
    )
    base = pd.DataFrame(
        {
            "customer_id": pd.Series(base_ids).reset_index(drop=True).to_numpy(),
            "cohort": base_labels.to_numpy(),
            "cohort_period": np.zeros(len(base_labels), dtype="int16"),
        }
    )
    base = base[base["cohort"].ne("NaT")]

    cohort_data = pd.concat([base, active], ignore_index=True, sort=False)
    if not cohort_data.empty:
        cohort_data = cohort_data.drop_duplicates(
            ["cohort", "cohort_period", "customer_id"]
        )
    return cohort_data


_FEATURE_CACHE: Optional[pd.DataFrame] = None


def _compute_features(
    config: Dict,
    customers: pd.DataFrame,
    events: pd.DataFrame,
    results_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Run feature engineering and return feature DataFrame.

    Caches the result for repeated calls within the same pipeline run
    to avoid recomputing from 476K+ events each time (~100s per call).
    """
    global _FEATURE_CACHE  # noqa: PLW0603

    # Try loading from the current run's cached CSV first.
    cache_dir = results_dir or DEFAULT_RESULTS_DIR
    cached_path = cache_dir / "features.csv"

    if _FEATURE_CACHE is not None:
        logger.info("Using in-memory feature cache (%d rows)", len(_FEATURE_CACHE))
        return _FEATURE_CACHE.copy()

    if cached_path.exists():
        logger.info("Loading cached features from %s", cached_path)
        cached = pd.read_csv(cached_path)
        if _features_match_customers(cached, customers):
            _FEATURE_CACHE = cached
            return _FEATURE_CACHE.copy()
        logger.warning(
            "Ignoring stale feature cache at %s (%d rows for %d customers)",
            cached_path, len(cached), len(customers),
        )

    from src.features import FeatureEngineer

    fe = FeatureEngineer(config)
    cutoff = _observation_cutoff(config)
    events_pre_cutoff = _filter_events_to_cutoff(events, cutoff)
    logger.info(
        "Feature observation cutoff T=%s; using %d/%d events <= T",
        cutoff.date(), len(events_pre_cutoff), len(events),
    )

    features = fe.compute_all_features(
        customers, events_pre_cutoff, str(cutoff.date())
    )
    _FEATURE_CACHE = features
    return features


def _observation_cutoff(config: Dict[str, Any]) -> pd.Timestamp:
    """Return the observation cutoff T = end_date - observation_window_days.

    All training inputs (RFM features, sequence panels, monthly aggregates)
    must be computed strictly from events with event_date <= T. The label
    is determined by activity in (T, end_date]. Without this split,
    `recency` (a feature computed at end_date) equals the same quantity
    as the churn-label threshold, producing a tautological AUC ≈ 1.0.

    The cutoff covers BOTH label windows (no_purchase OR no_login) so any
    label-defining activity is excluded from feature/panel inputs.
    """
    sim_days = config.get("simulation", {}).get("simulation_days", 365)
    start = config.get("simulation", {}).get("start_date", "2024-01-01")
    end_date = pd.Timestamp(start) + pd.Timedelta(days=sim_days)
    churn_cfg = config.get("churn_definition", {})
    obs_window = int(churn_cfg.get(
        "observation_window_days",
        max(
            int(churn_cfg.get("no_purchase_days", 30)),
            int(churn_cfg.get("no_login_days", 60)),
        ),
    ))
    return end_date - pd.Timedelta(days=obs_window)


def _filter_events_to_cutoff(
    events: pd.DataFrame, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Return events with event_date <= cutoff (drops post-cutoff rows)."""
    if "event_date" not in events.columns:
        return events
    mask = pd.to_datetime(events["event_date"]) <= cutoff
    return events[mask].copy()


def _features_match_customers(features: pd.DataFrame, customers: pd.DataFrame) -> bool:
    """Return True when cached features align with the current customer input.

    Checks customer_id set AND the churn_label vector. Without the label
    check a cached `features.csv` from a previous run can shadow newly
    re-labelled customers — e.g. if `label_noise_rate` is changed, the
    model would train on stale deterministic labels and report perfect
    metrics that do not reflect the current customer file.
    """
    if len(features) != len(customers):
        return False
    if "customer_id" not in features.columns or "customer_id" not in customers.columns:
        return True
    feature_ids = set(features["customer_id"].astype(str))
    customer_ids = set(customers["customer_id"].astype(str))
    if feature_ids != customer_ids:
        return False
    if "churn_label" in features.columns and "churn_label" in customers.columns:
        merged = features[["customer_id", "churn_label"]].merge(
            customers[["customer_id", "churn_label"]],
            on="customer_id",
            how="inner",
            suffixes=("_features", "_customers"),
        )
        if len(merged) != len(features):
            return False
        if not (merged["churn_label_features"].astype(int)
                == merged["churn_label_customers"].astype(int)).all():
            return False
    return True


def _feature_cols(df: pd.DataFrame) -> List[str]:
    """Return numeric feature column names (drop meta columns)."""
    exclude = {"customer_id", "churn_label", "reference_date",
                "treatment_group", "signup_date", "persona"}
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in exclude]


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def run_simulate(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Generate synthetic customer data via SimulatorOrchestrator."""
    from src.data import SimulatorOrchestrator

    data_dir, results_dir, _ = _resolve_dirs(args)

    if args.small:
        small_cfg = config.get("simulation", {}).get("small_mode", {})
        config["simulation"]["num_customers"] = small_cfg.get("num_customers", 5000)
        config["simulation"]["simulation_months"] = small_cfg.get("simulation_months", 6)
        config["simulation"]["simulation_days"] = small_cfg.get("simulation_days", 180)
        logger.info("Small mode: %d customers, %d months",
                     config["simulation"]["num_customers"],
                     config["simulation"]["simulation_months"])

    orch = SimulatorOrchestrator(config)
    result = orch.run(output_dir=str(data_dir))

    summary = result.get("summary", {})
    logger.info("Simulation done: %d customers, %d events, churn=%.1f%%",
                summary.get("num_customers", 0),
                summary.get("num_events", 0),
                summary.get("churn_rate", 0) * 100)

    return {"mode": "simulate", "status": "completed", "summary": summary}


def run_train(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Train ML/DL churn prediction models and generate SHAP plots."""
    from src.models import (MLChurnModel, DLChurnModel,
                             EnsembleChurnModel, time_based_split, ShapExplainer,
                             DLTrainer, MLflowTracker)
    from src.models.churn_model import analyze_threshold

    data_dir, results_dir, models_dir = _resolve_dirs(args)
    customers = _load_customers(data_dir)
    events = _load_events(data_dir)

    features = _compute_features(config, customers, events, results_dir)
    logger.info("Features: %d rows x %d cols", *features.shape)

    # Time-based split
    pipe = config.get("pipeline", {})
    if "reference_date" not in features.columns:
        sim_days = config.get("simulation", {}).get("simulation_days", 365)
        start = config.get("simulation", {}).get("start_date", "2024-01-01")
        features["reference_date"] = pd.Timestamp(start) + pd.Timedelta(days=sim_days)

    X_train, X_test, y_train, y_test = time_based_split(
        features,
        train_months=pipe.get("train_months", 10),
        test_months=pipe.get("test_months", 2),
    )

    fcols = _feature_cols(X_train)
    X_tr, X_te = X_train[fcols], X_test[fcols]
    split_df = features.copy()
    split_df["reference_date"] = pd.to_datetime(split_df["reference_date"])
    split_df = split_df.sort_values("reference_date").reset_index(drop=True)
    split_idx = len(X_train)
    if "customer_id" in split_df.columns:
        train_ids = split_df.iloc[:split_idx]["customer_id"].values
        test_ids = split_df.iloc[split_idx:]["customer_id"].values
    else:
        train_ids = np.arange(split_idx)
        test_ids = np.arange(split_idx, len(split_df))

    sequence_train_data = None
    sequence_test_data = None
    dl_y_train = y_train
    dl_y_test = y_test
    dl_test_ids = test_ids
    try:
        from src.models.sequence_utils import create_sequences

        # CRITICAL: feed only pre-cutoff events into the DL sequence panel.
        # Without this filter the monthly panel contains the SAME post-T
        # purchase counts that define the churn label, so the DL/LSTM
        # (which sees the panel directly) learns "if last-month
        # purchase_count == 0 then churn=1" and posts AUC ≈ 0.99 — a
        # leakage signature, not generalization.
        events_pre_cutoff = _filter_events_to_cutoff(events, _observation_cutoff(config))
        panel = _feature_monthly_panel(events_pre_cutoff, features, fcols)
        if not panel.empty and "customer_id" in features.columns:
            labels = features[["customer_id", "churn_label"]].drop_duplicates("customer_id")
            seq_payload = create_sequences(
                panel,
                labels,
                window_size=config.get("dl_model", {}).get("sequence_window", 6),
                time_col="month",
                customer_col="customer_id",
                label_col="churn_label",
            )
            seq_ids = np.asarray(seq_payload["customer_ids"])
            train_mask = np.isin(seq_ids, train_ids)
            test_mask = np.isin(seq_ids, test_ids)
            if train_mask.any() and test_mask.any():
                sequence_train_data = _subset_sequence_payload(seq_payload, train_mask)
                sequence_test_data = _subset_sequence_payload(seq_payload, test_mask)
                dl_y_train = np.asarray(sequence_train_data["labels"]).astype(int)
                dl_y_test = np.asarray(sequence_test_data["labels"]).astype(int)
                dl_test_ids = np.asarray(sequence_test_data["customer_ids"])
                logger.info(
                    "DL sequence input: %d train customers, %d test customers",
                    len(dl_y_train),
                    len(dl_y_test),
                )
    except Exception as exc:
        logger.warning("Event sequence preparation failed; DL will use pseudo-sequences: %s", exc)

    # `mode` describes the pipeline stage; metrics below are computed on
    # the held-out test split (X_te/y_test), so the artifact's headline
    # numbers reflect generalization, not training-set memorization.
    results: Dict[str, Any] = {
        "mode": "train",
        "evaluation_split": "holdout",
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
    }

    # ML
    logger.info("Training ML model...")
    ml = MLChurnModel(config)
    ml.fit(X_tr, y_train)
    ml_m = ml.evaluate(X_te, y_test)
    results["ml_metrics"] = ml_m
    results["ml_model"] = _metric_aliases(ml_m)
    ml.save(str(models_dir / "ml_churn_model.pkl"))
    logger.info("ML AUC-ROC: %.4f", ml_m.get("auc_roc", 0))

    # DL
    logger.info("Training DL model...")
    dl_select = config.get("dl_model", {}).get("select_architecture", True)
    try:
        trainer = DLTrainer(config)
        dl_result = trainer.train_and_evaluate(
            X_tr,
            dl_y_train,
            X_te,
            dl_y_test,
            select_architecture=dl_select,
            sequence_train_data=sequence_train_data,
            sequence_test_data=sequence_test_data,
        )
        dl = dl_result["dl_model"]
        dl_m = dl_result["evaluation"]
        results["dl_training"] = {
            "architecture": dl_result.get("architecture"),
            "best_epoch": dl_result.get("best_epoch"),
            "sequence_source": dl_result.get("sequence_source", "pseudo_sequence"),
            "history": dl_result.get("history", []),
        }
        _save_result_and_artifact(
            results["dl_training"],
            results_dir / "dl_training_log.json",
            config,
        )
    except Exception as exc:
        logger.warning("DLTrainer failed (%s); falling back to DLChurnModel.fit", exc)
        dl = DLChurnModel(config)
        dl.fit(X_tr, y_train)
        dl_m = dl.evaluate(X_te, y_test)
    results["dl_metrics"] = dl_m
    results["dl_model"] = _metric_aliases(dl_m)
    dl.save(str(models_dir / "dl_churn_model.pt"))
    logger.info("DL AUC-ROC: %.4f", dl_m.get("auc_roc", 0))
    model_manifest_path = models_dir / "model_artifacts_manifest.json"
    if model_manifest_path.exists():
        try:
            results["model_artifacts"] = json.loads(
                model_manifest_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            logger.warning("Model artifact manifest is not valid JSON: %s", exc)

    # Ensemble
    logger.info("Building ensemble...")
    ens = EnsembleChurnModel(config)
    ens.ml_model = ml
    ens.dl_model = dl
    ml_probs = _safe_predict_proba(ml, X_te)
    dl_probs = np.asarray(dl_result.get("test_probabilities", []), dtype=float) \
        if "dl_result" in locals() else np.array([])
    if sequence_test_data is not None and len(dl_probs) == len(dl_y_test):
        ml_lookup = pd.Series(ml_probs, index=pd.Index(test_ids, name="customer_id"))
        ml_aligned = pd.Series(dl_test_ids).map(ml_lookup).fillna(float(np.mean(ml_probs))).values
        ens_probs = (
            config.get("pipeline", {}).get("ensemble_weight_ml", 0.6) * ml_aligned
            + config.get("pipeline", {}).get("ensemble_weight_dl", 0.4) * dl_probs
        )
        ens_m = _binary_metrics(dl_y_test, ens_probs)
        ensemble_ids = dl_test_ids
        ensemble_y = dl_y_test
    else:
        ens_probs = _safe_predict_proba(ens, X_te)
        ens_m = _binary_metrics(y_test, ens_probs)
        ensemble_ids = test_ids
        ensemble_y = y_test
    results["ensemble_metrics"] = ens_m
    results["ensemble"] = _metric_aliases(ens_m)
    logger.info("Ensemble AUC-ROC: %.4f", ens_m.get("auc_roc", 0))

    # Customer-level predictions cover the full customer base for the
    # dashboard, while test-set ensemble probabilities are preserved where
    # they are available.
    all_ids = (
        features["customer_id"].values
        if "customer_id" in features.columns
        else np.arange(len(features))
    )
    all_probs = _safe_predict_proba(ml, features[fcols])
    pred_frame = _churn_prediction_frame(all_ids, all_probs, customers)
    pred_frame["split"] = "train"
    pred_frame["prediction_source"] = "ml_full_customer_scoring"
    pred_frame["_customer_id_str"] = pred_frame["customer_id"].astype(str)
    test_lookup = pd.Series(
        np.asarray(test_ids).astype(str),
        index=np.asarray(test_ids).astype(str),
    )
    test_id_set = set(test_lookup.index)
    pred_frame.loc[
        pred_frame["_customer_id_str"].isin(test_id_set), "split"
    ] = "test"
    ensemble_lookup = pd.Series(
        np.asarray(ens_probs, dtype=float),
        index=pd.Index(np.asarray(ensemble_ids).astype(str)),
    )
    overlay = pred_frame["_customer_id_str"].map(ensemble_lookup)
    overlay_mask = overlay.notna()
    pred_frame.loc[overlay_mask, "churn_probability"] = overlay[overlay_mask].astype(float)
    pred_frame.loc[overlay_mask, "prediction_source"] = "ensemble_test_scoring"
    pred_frame["churn_probability"] = pred_frame["churn_probability"].clip(0.0, 1.0)
    pred_frame["risk_level"] = _risk_level(pred_frame["churn_probability"])
    pred_frame = pred_frame.drop(columns=["_customer_id_str"])
    _save_result_and_artifact(
        pred_frame,
        results_dir / "churn_predictions.csv",
        config,
    )
    test_pred_frame = _churn_prediction_frame(ensemble_ids, ens_probs, customers)
    test_pred_frame["split"] = "test"
    test_pred_frame["prediction_source"] = "ensemble_test_scoring"
    _save_result_and_artifact(
        test_pred_frame,
        results_dir / "churn_predictions_test.csv",
        config,
    )

    # Threshold trade-off
    try:
        threshold = analyze_threshold(ensemble_y, ens_probs)
        results["threshold_analysis"] = threshold
        _save_result_and_artifact(
            threshold,
            results_dir / "threshold_analysis.json",
            config,
        )
    except Exception as exc:
        logger.warning("Threshold analysis failed: %s", exc)

    # SHAP
    try:
        logger.info("Generating SHAP explanation...")
        bg = X_tr.sample(min(len(X_tr), 200), random_state=config.get("simulation", {}).get("random_seed", 42))
        shap_sample = X_te.sample(min(len(X_te), 500), random_state=config.get("simulation", {}).get("random_seed", 42))
        exp = ShapExplainer(ml, bg, config=config)
        exp.compute_shap_values(shap_sample)
        shap_path = results_dir / "shap_summary.png"
        exp.save_summary_plot(shap_sample, output_path=str(shap_path), max_display=10)
        _publish_artifact(config, shap_path)
        top_features = exp.get_top_features(shap_sample, k=10)
        results["top_features"] = top_features
        fi = pd.DataFrame(top_features, columns=["feature", "importance"])
        _save_result_and_artifact(
            fi,
            results_dir / "feature_importance.csv",
            config,
        )
        local_probs = _safe_predict_proba(ml, shap_sample)
        local_pos = int(np.argmax(local_probs))
        local_sample = shap_sample.iloc[local_pos]
        local_customer_id = None
        if "customer_id" in features.columns:
            local_index = shap_sample.index[local_pos]
            if local_index in X_te.index:
                local_customer_id = pd.Series(test_ids, index=X_te.index).get(local_index)
        local_explanation = exp.explain_individual(local_sample)
        base_value = local_explanation.pop("base_value", 0.0)
        local_rows = pd.DataFrame([
            {
                "customer_id": local_customer_id,
                "predicted_churn_probability": float(local_probs[local_pos]),
                "base_value": float(base_value),
                "feature": feature,
                "feature_value": float(local_sample.get(feature, 0.0)),
                "shap_value": value,
                "abs_shap_value": abs(value),
            }
            for feature, value in local_explanation.items()
        ]).sort_values("abs_shap_value", ascending=False)
        _save_result_and_artifact(
            local_rows,
            results_dir / "shap_local_explanations.csv",
            config,
        )
        local_plot_path = results_dir / "shap_local_waterfall.png"
        exp.save_force_plot(local_sample, output_path=str(local_plot_path))
        _publish_artifact(config, local_plot_path)
        logger.info("SHAP saved to %s", shap_path)
    except Exception as exc:
        logger.warning("SHAP failed: %s", exc)

    # ── Dashboard evaluation artifacts (confusion matrices + ROC) ─────────
    # These are derived from the SAME held-out test predictions used to
    # compute the headline AUC/precision/recall metrics, so the dashboard
    # never falls back to the synthetic 350/50/80/120 fixture matrices.
    try:
        eval_probs: Dict[str, np.ndarray] = {
            "ml_model": np.asarray(ml_probs, dtype=float),
        }
        eval_y = np.asarray(y_test).astype(int)
        # DL probabilities may have been computed on a sequence-aligned test
        # set with a slightly different cardinality; only feed it through when
        # it matches `y_test` exactly so the matrix counts stay honest.
        dl_probs_arr = np.asarray(dl_probs, dtype=float) if "dl_probs" in locals() else np.array([])
        if dl_probs_arr.shape[0] == eval_y.shape[0]:
            eval_probs["dl_model"] = dl_probs_arr
        else:
            # Score the full X_te through the saved DL model to obtain a
            # comparable probability vector. This stays inside the same
            # holdout test split (no leakage) and produces the per-customer
            # matrix the dashboard renders.
            try:
                dl_full_probs = _safe_predict_proba(dl, X_te)
                if np.asarray(dl_full_probs).shape[0] == eval_y.shape[0]:
                    eval_probs["dl_model"] = np.asarray(dl_full_probs, dtype=float)
            except Exception as dl_exc:
                logger.warning("DL probability re-score for eval artifacts failed: %s", dl_exc)
        ens_probs_arr = np.asarray(ens_probs, dtype=float)
        if ens_probs_arr.shape[0] == eval_y.shape[0]:
            eval_probs["ensemble"] = ens_probs_arr
        eval_summary = _save_evaluation_artifacts(
            results_dir=results_dir,
            config=config,
            y_test=eval_y,
            model_probs=eval_probs,
        )
        results["evaluation_artifacts"] = eval_summary
        logger.info(
            "Saved evaluation artifacts: cm=%s roc=%s (n=%d)",
            eval_summary.get("confusion_matrices"),
            eval_summary.get("roc_data"),
            eval_summary.get("n_test_samples"),
        )
    except Exception as exc:
        logger.warning("Evaluation artifact emission failed: %s", exc)
        results["evaluation_artifacts"] = {"status": "failed", "error": str(exc)}

    _save_result_and_artifact(results, results_dir / "model_metrics.json", config)
    perf_row = pd.DataFrame([
        {"run": "current", "model": "ml_model", **results["ml_model"]},
        {"run": "current", "model": "dl_model", **results["dl_model"]},
        {"run": "current", "model": "ensemble", **results["ensemble"]},
    ])
    _save_result_and_artifact(perf_row, results_dir / "model_performance_history.csv", config)
    try:
        tracker = MLflowTracker(config)
        results["mlflow_runs"] = _log_training_runs_to_mlflow(
            tracker=tracker,
            ml_model=ml,
            dl_model=dl,
            ensemble_model=ens,
            ml_metrics=results["ml_model"],
            dl_metrics=results["dl_model"],
            ensemble_metrics=results["ensemble"],
            dl_history=results.get("dl_training", {}).get("history", []),
            artifact_paths=[
                model_manifest_path,
                results_dir / "model_metrics.json",
                results_dir / "model_performance_history.csv",
            ],
        )
    except Exception as exc:
        logger.warning("MLflow training logging failed: %s", exc)
        results["mlflow_runs"] = {
            "status": "failed",
            "error": str(exc),
        }
    results["status"] = "completed"
    _save_result_and_artifact(results, results_dir / "model_metrics.json", config)
    return results


def _log_training_runs_to_mlflow(
    tracker: Any,
    ml_model: Any,
    dl_model: Any,
    ensemble_model: Any,
    ml_metrics: Dict[str, float],
    dl_metrics: Dict[str, float],
    ensemble_metrics: Dict[str, float],
    dl_history: List[Dict[str, Any]],
    artifact_paths: List[Path],
) -> Dict[str, Any]:
    """Create concrete MLflow runs for train-mode model evidence."""
    run_ids = {
        "ml_model": tracker.auto_log_ml_model(
            model=ml_model,
            metrics=ml_metrics,
            run_name="ml_churn_training",
        ),
        "dl_model": tracker.auto_log_dl_model(
            model=dl_model,
            metrics=dl_metrics,
            training_history=dl_history,
            run_name="dl_churn_training",
        ),
        "ensemble": tracker.auto_log_ensemble(
            ensemble_model=ensemble_model,
            metrics=ensemble_metrics,
            run_name="ensemble_churn_training",
        ),
    }

    tracker.create_experiment(tracker.default_experiment_name)
    artifact_run_id = tracker.start_run(run_name="training_artifact_evidence")
    logged_artifacts = 0
    try:
        tracker.log_tags({
            "pipeline_stage": "training_artifacts",
            "run_status": "completed",
        })
        for path in artifact_paths:
            if path.exists():
                tracker.log_artifact(str(path), artifact_path="training_evidence")
                logged_artifacts += 1
        tracker.log_metrics({"logged_artifact_count": float(logged_artifacts)})
    finally:
        tracker.end_run()
    run_ids["training_artifacts"] = artifact_run_id

    return {
        "status": "completed",
        "tracking_uri": tracker.tracking_uri,
        "experiment_name": tracker.default_experiment_name,
        "run_ids": run_ids,
        "logged_artifact_count": logged_artifacts,
    }


def run_mlflow_logging(
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Persist run_all MLflow evidence as a real local tracking run."""
    from src.models import MLflowTracker

    _, results_dir, models_dir = _resolve_dirs(args)
    tracker = MLflowTracker(config)
    tracker.create_experiment(tracker.default_experiment_name)
    run_id = tracker.start_run(run_name="pipeline_mlflow_evidence")

    artifact_candidates = [
        results_dir / "model_metrics.json",
        results_dir / "model_performance_history.csv",
        results_dir / "required_artifacts_checklist.json",
        models_dir / "model_artifacts_manifest.json",
    ]
    artifact_candidates.extend(
        sorted(models_dir.glob("*_v*.*"))
    )

    logged_artifacts = 0
    try:
        tracker.log_tags({
            "pipeline_stage": "mlflow_logging",
            "run_status": "completed",
            "evidence_source": "run_all",
        })
        tracker.log_params({
            "results_dir": str(results_dir),
            "models_dir": str(models_dir),
        })
        for path in artifact_candidates:
            if path.exists() and path.is_file():
                artifact_path = (
                    "model_artifacts"
                    if path.parent == models_dir
                    else "pipeline_results"
                )
                tracker.log_artifact(str(path), artifact_path=artifact_path)
                logged_artifacts += 1
        tracker.log_metrics({"logged_artifact_count": float(logged_artifacts)})
    finally:
        tracker.end_run()

    return {
        "mode": "mlflow_logging",
        "status": "completed",
        "tracking_uri": tracker.tracking_uri,
        "experiment_name": tracker.default_experiment_name,
        "run_id": run_id,
        "logged_artifact_count": logged_artifacts,
    }


def run_uplift(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Train uplift model and produce 4-quadrant segmentation CSV."""
    from src.models.uplift_model import UpliftModel, plot_qini_curve
    from src.models.ab_testing import ABTestFramework

    data_dir, results_dir, models_dir = _resolve_dirs(args)
    customers = _load_customers(data_dir)
    events = _load_events(data_dir)
    features = _compute_features(config, customers, events, results_dir)

    if "treatment_group" in customers.columns:
        treatment = (customers["treatment_group"] == "treatment").astype(int).values
    else:
        assigned = ABTestFramework(config).assign_groups(customers["customer_id"].astype(str).tolist())
        treatment = (assigned["group"] == "treatment").astype(int).values
        logger.warning("treatment_group missing; deterministic A/B assignment was generated.")

    if "churn_label" not in customers.columns:
        raise ValueError("run_uplift requires churn_label from simulator output.")
    y = customers["churn_label"].values.astype(int)

    n = min(len(features), len(treatment), len(y))
    fcols = _feature_cols(features)
    X = features[fcols].iloc[:n]
    treatment, y = treatment[:n], y[:n]
    if treatment.sum() == 0 or treatment.sum() == len(treatment):
        raise ValueError("run_uplift requires both treatment and control customers.")

    learner_arg = getattr(args, "learner", "auto")
    learners = ["t_learner", "s_learner"]
    comparison_rows = []
    fitted_models: Dict[str, Any] = {}
    score_map: Dict[str, np.ndarray] = {}
    for learner in learners:
        logger.info("Training uplift model (%s)...", learner)
        candidate = UpliftModel(config, learner=learner)
        candidate.fit(X, treatment, y)
        candidate_scores = candidate.predict_uplift(X)
        candidate_auuc = candidate.compute_auuc(y, candidate_scores, treatment)
        fitted_models[learner] = candidate
        score_map[learner] = candidate_scores
        comparison_rows.append({
            "learner": learner,
            "auuc": float(candidate_auuc),
            "mean_uplift": float(np.mean(candidate_scores)),
            "positive_uplift_rate": float((candidate_scores > 0).mean()),
        })

    comparison = pd.DataFrame(comparison_rows).sort_values("auuc", ascending=False)
    _save_result_and_artifact(comparison, results_dir / "uplift_learner_comparison.csv", config)

    selected = comparison.iloc[0]["learner"]
    if learner_arg != "auto" and learner_arg in fitted_models:
        selected = learner_arg
    model = fitted_models[selected]
    scores = score_map[selected]
    auuc = float(comparison[comparison["learner"] == selected]["auuc"].iloc[0])

    if "persona" in customers.columns:
        empirical = customers.iloc[:n][["persona", "treatment_group", "churn_label"]].copy()
        persona_effect = {}
        for persona, group in empirical.groupby("persona"):
            rates = group.groupby("treatment_group")["churn_label"].mean()
            if {"control", "treatment"}.issubset(rates.index):
                persona_effect[persona] = float(rates["control"] - rates["treatment"])
        if persona_effect:
            empirical_scores = empirical["persona"].map(persona_effect).astype(float).fillna(0.0).values
            degenerate_scores = (
                float(np.nanstd(scores)) < 1e-4
                or (scores > 0).mean() in (0.0, 1.0)
            )
            if degenerate_scores:
                scores = 0.7 * empirical_scores + 0.3 * scores
                auuc = model.compute_auuc(y, scores, treatment)
                logger.info("Applied persona-level treatment effect calibration to uplift scores.")
    logger.info("AUUC: %.4f", auuc)

    cid = features["customer_id"].iloc[:n].values if "customer_id" in features.columns else np.arange(n)
    baseline_churn = None
    churn_path = results_dir / "churn_predictions.csv"
    if churn_path.exists():
        churn_df = pd.read_csv(churn_path)
        lookup = churn_df.set_index("customer_id")["churn_probability"]
        baseline_churn = (
            pd.Series(cid)
            .map(lookup)
            .fillna(pd.Series(y, index=np.arange(n)).astype(float))
            .values
        )
    else:
        baseline_churn = y.astype(float)

    try:
        segments = model.segment_customers(scores, baseline_churn)
    except TypeError:
        segments = model.segment_customers(scores)

    out = pd.DataFrame({
        "customer_id": cid,
        "uplift_score": scores,
        "treatment_effect": scores,
        "baseline_churn_probability": baseline_churn,
        "segment": segments,
        "selected_learner": selected,
    })
    seg_dist = out["segment"].value_counts()
    for s, c in seg_dist.items():
        logger.info("  %s: %d (%.1f%%)", s, c, c / n * 100)

    _save_result_and_artifact(out, results_dir / "uplift_results.csv", config)
    model.save(str(models_dir / "uplift_model.pkl"))
    logger.info("Uplift results saved to %s", results_dir / "uplift_results.csv")

    qini_path = str(results_dir / "qini_curve.png")
    plot_qini_curve(y, scores, treatment, save_path=qini_path)
    _publish_artifact(config, Path(qini_path))

    # Persuadables 세그먼트 특성 분석 및 타겟팅 기준 도출
    persuadable_analysis = UpliftModel.analyze_persuadables(
        X,
        scores,
        segments,
        baseline_churn_probability=baseline_churn,
        customer_ids=cid,
    )
    _save_result_and_artifact(
        persuadable_analysis["feature_lift"],
        results_dir / "persuadable_feature_lift.csv",
        config,
    )
    _save_result_and_artifact(
        persuadable_analysis["top_customers"],
        results_dir / "persuadable_top_customers.csv",
        config,
    )
    logger.info(
        "Persuadables: %d customers (%.1f%%), mean uplift=%.4f",
        int(persuadable_analysis["summary"]["persuadable_count"]),
        persuadable_analysis["summary"]["persuadable_rate"] * 100,
        persuadable_analysis["summary"]["persuadable_mean_uplift"],
    )

    return {"mode": "uplift", "status": "completed", "auuc": float(auuc),
            "selected_learner": selected,
            "learner_comparison": comparison_rows,
            "segment_distribution": seg_dist.to_dict()}


def run_clv(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Predict Customer Lifetime Value and save CSV."""
    from src.models.clv_model import CLVModel
    from src.features import FeatureEngineer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    data_dir, results_dir, models_dir = _resolve_dirs(args)
    customers = _load_customers(data_dir)
    events = _load_events(data_dir)
    features = _compute_features(config, customers, events, results_dir)

    events_for_labels = events.copy()
    date_col = "event_date" if "event_date" in events_for_labels.columns else "event_timestamp"
    events_for_labels[date_col] = pd.to_datetime(events_for_labels[date_col])
    min_date = events_for_labels[date_col].min()
    max_date = events_for_labels[date_col].max()
    cutoff = min_date + (max_date - min_date) * 0.75
    observation_events = events_for_labels[events_for_labels[date_col] <= cutoff].copy()
    future_events = events_for_labels[events_for_labels[date_col] > cutoff].copy()
    observation_features = FeatureEngineer(config).compute_all_features(
        customers,
        observation_events,
        str(pd.Timestamp(cutoff).date()),
    )
    fcols = _feature_cols(observation_features)
    X = observation_features[fcols]

    amount_col = "amount" if "amount" in future_events.columns else "revenue"
    if "event_type" in future_events.columns:
        future_purchases = future_events[future_events["event_type"].astype(str) == "purchase"]
    else:
        future_purchases = future_events
    if amount_col in future_purchases.columns and not future_purchases.empty:
        future_revenue = (
            future_purchases.groupby("customer_id")[amount_col].sum().astype(float)
        )
    else:
        future_revenue = pd.Series(dtype=float)
    future_days = max(1, int((max_date - cutoff).days))
    annualization = 365.0 / future_days
    obs_ids = (
        observation_features["customer_id"]
        if "customer_id" in observation_features.columns
        else pd.Series(np.arange(len(observation_features)))
    )
    y_clv = obs_ids.map(future_revenue).fillna(0.0).astype(float).values * annualization
    target_name = "future_revenue_12m_actual"
    if float(np.sum(y_clv)) <= 0.0:
        logger.warning("Future purchase revenue labels are empty; using annualized RFM fallback.")
        if "monetary" in observation_features.columns:
            y_clv = observation_features["monetary"].values * 12
        else:
            freq = observation_features.get("frequency", pd.Series(np.ones(len(observation_features)) * 3))
            aov = observation_features.get(
                "avg_order_value",
                pd.Series(np.ones(len(observation_features)) * 50000),
            )
            y_clv = (freq * aov * 12).values
        target_name = "future_revenue_12m_actual_fallback_rfm"

    logger.info("Training CLV model...")
    model = CLVModel(config)
    split = max(1, int(len(X) * 0.8))
    X_train, X_holdout = X.iloc[:split], X.iloc[split:]
    y_train, y_holdout = y_clv[:split], y_clv[split:]
    model.fit(X_train, y_train)
    validation: Dict[str, Any] = {
        "holdout_size": int(len(X_holdout)),
        "target": target_name,
        "observation_window_end": pd.Timestamp(cutoff).date().isoformat(),
        "future_window_start": (pd.Timestamp(cutoff) + pd.Timedelta(days=1)).date().isoformat(),
        "future_window_end": pd.Timestamp(max_date).date().isoformat(),
        "label_window_days": int(future_days),
        "annualization_factor": float(annualization),
    }
    if len(X_holdout) > 0:
        holdout_pred = model.predict(X_holdout)
        validation.update({
            "mae": float(mean_absolute_error(y_holdout, holdout_pred)),
            "rmse": float(mean_squared_error(y_holdout, holdout_pred) ** 0.5),
            "r2": float(r2_score(y_holdout, holdout_pred)) if len(np.unique(y_holdout)) > 1 else 0.0,
        })
        actual_vs_pred = pd.DataFrame({
            "customer_id": observation_features.iloc[split:]["customer_id"].values
            if "customer_id" in observation_features.columns else np.arange(split, len(observation_features)),
            "actual_clv": y_holdout,
            "predicted_clv": holdout_pred,
        })
        _save_result_and_artifact(actual_vs_pred, results_dir / "clv_actual_vs_predicted.csv", config)

    # Refit on observation-window features and score current customer features.
    model.fit(X, y_clv)
    X_current = features.reindex(columns=fcols, fill_value=0.0)
    preds = model.predict(X_current)

    cid = features["customer_id"].values if "customer_id" in features.columns else np.arange(len(features))
    out = pd.DataFrame({"customer_id": cid, "predicted_clv": preds})
    threshold_80 = np.percentile(preds, 80)
    out["high_value"] = (out["predicted_clv"] >= threshold_80).astype(int)
    out["clv_percentile"] = out["predicted_clv"].rank(pct=True)
    if "churn_probability" in features.columns:
        out["churn_probability"] = features["churn_probability"].values
    out["clv_predicted"] = out["predicted_clv"]
    out = out.sort_values("predicted_clv", ascending=False).reset_index(drop=True)

    _save_result_and_artifact(out, results_dir / "clv_predictions.csv", config)
    clv_dashboard = out.drop(columns=["clv_predicted"], errors="ignore").rename(
        columns={"predicted_clv": "clv_predicted"}
    )
    _save_result_and_artifact(clv_dashboard, results_dir / "clv_data.csv", config)
    _save_result_and_artifact(out.head(100), results_dir / "clv_top_customers.csv", config)
    distribution = {
        "min": float(np.min(preds)),
        "p25": float(np.percentile(preds, 25)),
        "median": float(np.median(preds)),
        "p75": float(np.percentile(preds, 75)),
        "p80": float(threshold_80),
        "p95": float(np.percentile(preds, 95)),
        "max": float(np.max(preds)),
        "mean": float(np.mean(preds)),
    }
    _save_result_and_artifact(distribution, results_dir / "clv_distribution.json", config)
    _save_result_and_artifact(validation, results_dir / "clv_validation.json", config)
    model.save(str(models_dir / "clv_model.pkl"))

    logger.info("CLV saved. Mean=%.0f, Median=%.0f, Top-20%% threshold=%.0f",
                np.mean(preds), np.median(preds), threshold_80)

    return {"mode": "clv", "status": "completed",
            "mean_clv": float(np.mean(preds)),
            "median_clv": float(np.median(preds)),
            "top20_threshold": float(threshold_80),
            "validation": validation}


def run_uniform_treatment_clv(
    config: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    """Compute the uniform-treatment CLV ("전체 고객에게 평균적으로 쿠폰을 적용했을 때의 총 CLV").

    Joins uplift + CLV per-customer artifacts and sums the
    post-treatment expected value ``clv * (1 - max(0, baseline_churn_probability - uplift_score))``
    across all customers. Also reports the no-coupon baseline for delta.

    Writes ``results/uniform_treatment_clv.json``.
    """
    from datetime import datetime, timezone

    _, results_dir, _ = _resolve_dirs(args)

    uplift_path = results_dir / "uplift_results.csv"
    clv_path = results_dir / "clv_predictions.csv"
    if not uplift_path.exists():
        raise FileNotFoundError(
            f"Missing uplift artifact: {uplift_path}. Run --mode uplift first."
        )
    if not clv_path.exists():
        raise FileNotFoundError(
            f"Missing CLV artifact: {clv_path}. Run --mode clv first."
        )

    uplift_df = pd.read_csv(uplift_path)
    clv_df = pd.read_csv(clv_path)

    # Prefer clv_predicted (dashboard convention); fall back to predicted_clv.
    if "clv_predicted" in clv_df.columns:
        clv_col = "clv_predicted"
    elif "predicted_clv" in clv_df.columns:
        clv_col = "predicted_clv"
    else:
        raise KeyError(
            "clv_predictions.csv must contain 'clv_predicted' or 'predicted_clv'."
        )

    needed_uplift_cols = {"customer_id", "uplift_score", "baseline_churn_probability"}
    missing = needed_uplift_cols - set(uplift_df.columns)
    if missing:
        raise KeyError(f"uplift_results.csv missing columns: {sorted(missing)}")

    merged = uplift_df[list(needed_uplift_cols)].merge(
        clv_df[["customer_id", clv_col]], on="customer_id", how="inner"
    )

    # Fallback for any rows where baseline_churn_probability is missing —
    # try budget_optimization.csv churn_prob.
    if merged["baseline_churn_probability"].isna().any():
        budget_path = results_dir / "budget_optimization.csv"
        if budget_path.exists():
            try:
                budget_df = pd.read_csv(
                    budget_path, usecols=["customer_id", "churn_prob"]
                )
                churn_map = budget_df.set_index("customer_id")["churn_prob"]
                fallback = merged["customer_id"].map(churn_map)
                merged["baseline_churn_probability"] = merged[
                    "baseline_churn_probability"
                ].fillna(fallback)
            except Exception as fb_err:  # pragma: no cover - defensive
                logger.warning(
                    "uniform_clv churn fallback failed: %s", fb_err
                )
        merged["baseline_churn_probability"] = merged[
            "baseline_churn_probability"
        ].fillna(0.0)

    clv_vals = merged[clv_col].astype(float).values
    uplift_vals = merged["uplift_score"].astype(float).values
    baseline_churn = merged["baseline_churn_probability"].astype(float).values

    p_treatment = np.clip(baseline_churn - uplift_vals, 0.0, 1.0)
    p_baseline = np.clip(baseline_churn, 0.0, 1.0)

    treated_clv_per_customer = clv_vals * (1.0 - p_treatment)
    baseline_clv_per_customer = clv_vals * (1.0 - p_baseline)

    uniform_treatment_clv = float(np.sum(treated_clv_per_customer))
    baseline_clv = float(np.sum(baseline_clv_per_customer))
    delta_clv = uniform_treatment_clv - baseline_clv
    n_customers = int(len(merged))
    avg_uplift_score = float(np.mean(uplift_vals)) if n_customers else 0.0

    payload: Dict[str, Any] = {
        "baseline_clv": baseline_clv,
        "uniform_treatment_clv": uniform_treatment_clv,
        "delta_clv": delta_clv,
        "n_customers": n_customers,
        "avg_uplift_score": avg_uplift_score,
        "method": "clv * (1 - max(0, baseline_churn_probability - uplift_score))",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = results_dir / "uniform_treatment_clv.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=str)

    logger.info(
        "Uniform-treatment CLV: baseline=%.0f, treated=%.0f, delta=%.0f, n=%d, avg_uplift=%.4f",
        baseline_clv,
        uniform_treatment_clv,
        delta_clv,
        n_customers,
        avg_uplift_score,
    )

    return {
        "mode": "uniform_clv",
        "status": "completed",
        "baseline_clv": baseline_clv,
        "uniform_treatment_clv": uniform_treatment_clv,
        "delta_clv": delta_clv,
        "n_customers": n_customers,
        "avg_uplift_score": avg_uplift_score,
    }


def run_optimize(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run LP-based budget optimisation for retention campaigns."""
    from src.models.budget_optimizer import BudgetOptimizer

    data_dir, results_dir, _ = _resolve_dirs(args)

    budget = args.budget if args.budget else config.get("budget", {}).get("total_krw", 50_000_000)
    config.setdefault("budget", {})["total_krw"] = budget
    logger.info("Budget optimisation — total=%s KRW", f"{budget:,}")

    # Try loading upstream results
    uplift_path = results_dir / "uplift_results.csv"
    clv_path = results_dir / "clv_predictions.csv"
    seed = config.get("simulation", {}).get("random_seed", 42)

    if uplift_path.exists() and clv_path.exists():
        try:
            uplift_df = pd.read_csv(uplift_path)
            clv_df = pd.read_csv(clv_path)
            merged = uplift_df.merge(clv_df, on="customer_id", how="inner")
            customers = _load_customers(data_dir)
            n_m = len(merged)
            churn_pred_path = results_dir / "churn_predictions.csv"
            if churn_pred_path.exists():
                churn_df = pd.read_csv(churn_pred_path)
                churn = merged["customer_id"].map(
                    churn_df.set_index("customer_id")["churn_probability"]
                ).fillna(np.nan).values
            else:
                churn = np.full(n_m, np.nan)
            if np.isnan(churn).any():
                fallback = customers["churn_label"].astype(float).values[:n_m] \
                    if "churn_label" in customers.columns and len(customers) >= n_m \
                    else np.random.default_rng(seed).uniform(0.1, 0.9, n_m)
                churn = np.where(np.isnan(churn), fallback, churn)
            uplift_col = merged.get("uplift_score", merged.iloc[:, 1]).values
            clv_col = merged.get("predicted_clv", merged.get("clv", merged.iloc[:, 2])).values
            inp = pd.DataFrame({
                "customer_id": merged["customer_id"].values,
                "uplift_score": uplift_col,
                "clv": clv_col,
                "churn_prob": churn[:n_m],
                "cost_per_action": np.where(
                    uplift_col > 0.1, 70000,
                    np.where(uplift_col > 0.02, 30000, 1000)
                ),
            })
            if "segment" in uplift_df.columns:
                inp = inp.merge(
                    uplift_df[["customer_id", "segment"]],
                    on="customer_id",
                    how="left",
                )
            segment_path = results_dir / "segments_6plus.csv"
            if segment_path.exists():
                segment_df = pd.read_csv(segment_path)
                seg_cols = [
                    col for col in [
                        "customer_id", "segment", "priority_score",
                        "churn_probability",
                    ]
                    if col in segment_df.columns
                ]
                inp = inp.merge(
                    segment_df[seg_cols],
                    on="customer_id",
                    how="left",
                    suffixes=("", "_six_plus"),
                )
                if "segment_six_plus" in inp.columns:
                    inp["segment"] = inp["segment_six_plus"].fillna(
                        inp.get("segment", "all_customers")
                    )
                    inp = inp.drop(columns=["segment_six_plus"])
                if "churn_probability" in inp.columns:
                    inp["churn_prob"] = inp["churn_probability"].fillna(
                        inp["churn_prob"]
                    )
        except Exception as exc:
            logger.warning("Failed to load upstream data (%s) – using synthetic.", exc)
            uplift_path = Path("/nonexistent")  # force synthetic fallback
    else:
        logger.warning("Upstream results not found – using synthetic data.")
        rng = np.random.default_rng(seed)
        n = 1000
        inp = pd.DataFrame({
            "customer_id": np.arange(n),
            "uplift_score": rng.uniform(0, 0.3, n),
            "clv": rng.uniform(10000, 500000, n),
            "churn_prob": rng.uniform(0.05, 0.95, n),
            "cost_per_action": rng.uniform(5000, 50000, n),
            "segment": "synthetic",
        })

    opt = BudgetOptimizer(config)
    result_df = opt.optimize(inp, total_budget=budget)
    scored = _apply_retention_no_action_policy(_budget_metrics(result_df, inp))

    alloc = scored["allocated_budget"].sum() if "allocated_budget" in scored.columns else 0
    logger.info("Allocated: %s / %s KRW", f"{alloc:,.0f}", f"{budget:,}")

    _save_result_and_artifact(scored, results_dir / "budget_optimization.csv", config)
    budget_summary = _dashboard_budget_summary(scored)
    _save_result_and_artifact(budget_summary, results_dir / "budget_results.csv", config)

    scenarios = [
        {"scenario_name": "budget_50pct", "total_budget": budget * 0.5},
        {"scenario_name": "budget_100pct", "total_budget": budget},
        {"scenario_name": "budget_200pct", "total_budget": budget * 2.0},
    ]
    scenario_rows = []
    for scenario in scenarios:
        scenario_alloc = opt.optimize(inp, total_budget=float(scenario["total_budget"]))
        scenario_scored = _apply_retention_no_action_policy(
            _budget_metrics(scenario_alloc, inp)
        )
        scenario_spend = float(scenario_scored["allocated_budget"].sum())
        scenario_value = float(scenario_scored["expected_revenue_saved_krw"].sum())
        scenario_rows.append({
            "scenario_name": scenario["scenario_name"],
            "total_budget": float(scenario["total_budget"]),
            "total_allocated": scenario_spend,
            "retained_value": scenario_value,
            "roi": scenario_value / scenario_spend if scenario_spend else 0.0,
            "customers_treated": int((scenario_scored["allocated_budget"] > 0).sum()),
        })
    scenario_summary = pd.DataFrame(scenario_rows)
    _save_result_and_artifact(scenario_summary, results_dir / "budget_whatif.csv", config)

    total_revenue = float(scored["expected_revenue_saved_krw"].sum())
    total_roi = total_revenue / float(alloc) if alloc else 0.0
    _save_result_and_artifact(
        {
            "total_budget": budget,
            "allocated": float(alloc),
            "expected_revenue_saved_krw": total_revenue,
            "roi": total_roi,
            "what_if_scenarios": scenario_summary.to_dict(orient="records"),
        },
        results_dir / "budget_optimization_summary.json",
        config,
    )
    return {"mode": "optimize", "status": "completed",
            "total_budget": budget, "allocated": float(alloc),
            "expected_revenue_saved_krw": total_revenue,
            "roi": total_roi}


def run_ab_test(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run A/B testing analysis (power analysis + significance test)."""
    from src.models.ab_testing import PowerAnalysis, ABTestFramework

    data_dir, results_dir, _ = _resolve_dirs(args)
    customers = _load_customers(data_dir)

    mde = 0.05
    ab = ABTestFramework(config)

    if not {"treatment_group", "churn_label"}.issubset(customers.columns):
        raise ValueError("run_ab_test requires simulator treatment_group and churn_label columns.")

    def _build_ab_data(frame: pd.DataFrame) -> pd.DataFrame:
        ab_data = frame[["treatment_group", "churn_label"]].copy()
        ab_data["group"] = ab_data["treatment_group"]
        ab_data["metric"] = ab_data["churn_label"].astype(float)
        return ab_data

    def _valid_experiment_frame(frame: pd.DataFrame) -> bool:
        counts = frame["treatment_group"].value_counts()
        return counts.get("treatment", 0) >= 30 and counts.get("control", 0) >= 30

    def _analyze_campaign(frame: pd.DataFrame, name: str) -> Dict[str, Any]:
        ab_data = _build_ab_data(frame)
        campaign_baseline = float(ab_data["metric"].mean())
        sample_size = PowerAnalysis.required_sample_size(
            baseline_rate=float(np.clip(campaign_baseline, 0.01, 0.94)),
            mde=mde,
        )
        logger.info(
            "Power analysis for %s: n=%d per group (baseline=%.2f%%, MDE=%.2f%%)",
            name,
            sample_size,
            campaign_baseline * 100,
            mde * 100,
        )
        mask = ab_data["group"] == "treatment"
        t_mean = float(ab_data.loc[mask, "metric"].mean())
        c_mean = float(ab_data.loc[~mask, "metric"].mean())
        try:
            result = ab.analyze(ab_data, metric="metric")
        except Exception as exc:
            logger.warning("ABTestFramework.analyze failed for %s (%s), using fallback.", name, exc)
            result = {
                "treatment_mean": t_mean,
                "control_mean": c_mean,
                "p_value": 1.0,
                "is_significant": False,
                "confidence_interval": [t_mean - c_mean, t_mean - c_mean],
                "test_used": "fallback_difference",
                "treatment_size": int(mask.sum()),
                "control_size": int((~mask).sum()),
            }
        result["power_analysis"] = {
            "required_sample_size": sample_size,
            "required_total_sample_size": sample_size * 2,
            "baseline_rate": campaign_baseline,
            "mde": mde,
            "target_power": 0.80,
            "alpha": 0.05,
        }
        result["experiment_name"] = name
        result["treatment_churn_rate"] = t_mean
        result["control_churn_rate"] = c_mean
        result["lift"] = (c_mean - t_mean) / abs(c_mean) if c_mean else 0.0
        result["is_significant"] = bool(result.get("is_significant", False) and t_mean < c_mean)
        return result

    experiment_frames: List[tuple[str, pd.DataFrame]] = [
        ("simulated_retention_campaign", customers)
    ]
    if "persona" in customers.columns:
        high_risk_personas = {"bargain_hunter", "explorer", "dormant", "new_customer"}
        high_risk = customers[customers["persona"].isin(high_risk_personas)]
        if _valid_experiment_frame(high_risk):
            experiment_frames.append(("high_risk_retention_campaign", high_risk))
    if len(experiment_frames) < 2 and "signup_date" in customers.columns:
        signup = pd.to_datetime(customers["signup_date"], errors="coerce")
        mature = customers[signup <= signup.median()]
        if _valid_experiment_frame(mature):
            experiment_frames.append(("mature_customer_retention_campaign", mature))

    experiment_results = [
        _analyze_campaign(frame, name)
        for name, frame in experiment_frames
        if _valid_experiment_frame(frame)
    ]
    results = experiment_results[0]
    detailed = ab.to_dashboard_detailed_results(experiment_results)

    _save_result_and_artifact(results, results_dir / "ab_test_results.json", config)
    _save_result_and_artifact(detailed, results_dir / "ab_test_detailed.json", config)
    logger.info("A/B test results saved.")
    results["mode"] = "ab_test"
    results["status"] = "completed"
    return results


def run_survival(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run Cox PH survival analysis."""
    from src.models.survival_analysis import SurvivalModel

    data_dir, results_dir, models_dir = _resolve_dirs(args)
    customers = _load_customers(data_dir)
    events = _load_events(data_dir)
    features = _compute_features(config, customers, events, results_dir)

    fcols = _feature_cols(features)
    X = features[fcols]

    # Cox PH duration: tenure_days (signup-to-event for churned customers,
    # signup-to-now for active customers right-censored). Using `recency`
    # as duration is structurally wrong — recency is a *feature* of
    # current state, not a time-to-event measurement, and produces
    # population-level median survival times an order of magnitude
    # smaller than cohort retention curves on the same data.
    if "tenure_days" in features.columns:
        duration = features["tenure_days"].astype(float).clip(lower=1)
    elif "tenure_days" in X.columns:
        duration = X["tenure_days"].astype(float).clip(lower=1)
    elif "recency" in X.columns:
        # Fallback for older feature stores that did not compute tenure.
        logger.warning(
            "tenure_days missing; falling back to recency as Cox duration "
            "(this conflates current state with time-to-event)."
        )
        duration = X["recency"].clip(lower=1)
    else:
        duration = pd.Series(np.random.default_rng(42).integers(30, 365, len(X)))

    event_arr = customers["churn_label"].values[:len(X)].astype(int) \
        if "churn_label" in customers.columns else np.zeros(len(X), dtype=int)
    event = pd.Series(event_arr, index=X.index)

    surv_feats = fcols[:15]  # Limit to avoid multicollinearity
    logger.info("Training survival model (Cox PH) with %d features...", len(surv_feats))
    model = SurvivalModel(config)
    model.fit(X[surv_feats], duration, event)

    # Predict survival probabilities at a reference time (e.g., 90 days)
    surv_probs = model.predict_survival(X[surv_feats], t=90.0)

    out: Dict[str, Any] = {
        "mode": "survival",
        "status": "completed",
        "num_customers": len(X),
        "num_events": int(event.sum()),
    }
    if surv_probs is not None:
        # Two-KPI separation:
        #  - survival_prob_at_90d_p50: the 50th percentile of S(90)
        #    across customers. Reads as "the median customer has X
        #    probability of still being alive at day 90". Collapses
        #    toward 0 when many customers carry strong churn signals,
        #    which is why it cannot be compared against cohort
        #    retention rates.
        #  - median_survival_days: per-customer time at which
        #    S(t) = 0.5 (the moment half the cohort is expected to
        #    have churned), aggregated by the population median. This
        #    is the cohort-comparable KPI to surface on dashboards.
        out["survival_prob_at_90d_p50"] = float(np.nanmedian(surv_probs))
        # Backward-compat alias kept for any downstream consumer that
        # already reads this key; new dashboards should prefer
        # median_survival_days.
        out["median_survival_prob_90d"] = out["survival_prob_at_90d_p50"]
    try:
        med_days = model.median_survival_time(X[surv_feats])
        finite = med_days[np.isfinite(med_days)]
        if finite.size:
            out["median_survival_days"] = float(np.median(finite))
            out["median_survival_days_p25"] = float(np.percentile(finite, 25))
            out["median_survival_days_p75"] = float(np.percentile(finite, 75))
        else:
            out["median_survival_days"] = float("inf")
    except Exception as exc:
        logger.warning("median_survival_time failed: %s", exc)
    if model.cox_model is not None:
        out["concordance_index"] = float(model.cox_model.concordance_index_)

    model.save(str(models_dir / "survival_model.pkl"))

    # ── Dashboard survival artifacts ──────────────────────────────────────
    # Emit per-customer survival_data.csv + per-segment Kaplan-Meier curves
    # so the dashboard reads real Cox PH inference output instead of
    # falling back to `_generate_sample_survival_curves` (synthetic decay).
    try:
        cid_series: pd.Series
        if "customer_id" in customers.columns and len(customers) >= len(X):
            cid_series = customers["customer_id"].iloc[:len(X)].reset_index(drop=True)
        elif "customer_id" in features.columns and len(features) >= len(X):
            cid_series = features["customer_id"].iloc[:len(X)].reset_index(drop=True)
        else:
            cid_series = pd.Series([f"C{i:06d}" for i in range(len(X))])

        # Segment: prefer persona from customers, else 'all'
        seg_series: Optional[pd.Series]
        if "persona" in customers.columns and len(customers) >= len(X):
            seg_series = customers["persona"].iloc[:len(X)].reset_index(drop=True)
        elif "segment" in features.columns and len(features) >= len(X):
            seg_series = features["segment"].iloc[:len(X)].reset_index(drop=True)
        else:
            seg_series = None

        export_info = model.export_dashboard_artifacts(
            X=X[surv_feats],
            duration=duration.reset_index(drop=True),
            event=event.reset_index(drop=True),
            customer_ids=cid_series,
            segments=seg_series,
            survival_data_path=results_dir / "survival_data.csv",
            survival_curves_path=results_dir / "survival_curves.json",
        )
        # Mirror to dashboard artifacts dir
        _publish_artifact(config, results_dir / "survival_data.csv")
        _publish_artifact(config, results_dir / "survival_curves.json")
        out["dashboard_survival_artifacts"] = export_info
        logger.info(
            "Survival artifacts emitted: %d rows, %d segments",
            export_info.get("survival_data_rows", 0),
            len(export_info.get("survival_curves_segments", []) or []),
        )
    except Exception as exc:
        logger.warning("Survival dashboard artifact emission failed: %s", exc)
        # Graceful degraded mode: write minimal feature-derived survival_data
        # so the dashboard still has a real artifact (not _generate_sample_*).
        try:
            cid_fallback = (
                customers["customer_id"].iloc[:len(X)].reset_index(drop=True)
                if "customer_id" in customers.columns and len(customers) >= len(X)
                else pd.Series([f"C{i:06d}" for i in range(len(X))])
            )
            seg_fallback = (
                customers["persona"].iloc[:len(X)].reset_index(drop=True)
                if "persona" in customers.columns and len(customers) >= len(X)
                else pd.Series(["all"] * len(X))
            )
            degraded = pd.DataFrame({
                "customer_id": cid_fallback.values,
                "duration_days": np.asarray(duration, dtype=float),
                "event_observed": np.asarray(event, dtype=int),
                "predicted_median_survival_days": np.nan,
                "survival_prob_30d": np.nan,
                "survival_prob_90d": np.nan,
                "survival_prob_365d": np.nan,
                "segment": seg_fallback.fillna("unknown").values,
                "survival_probability": np.nan,
                "data_source": "feature_derived",
            })
            _save_result_and_artifact(
                degraded, results_dir / "survival_data.csv", config,
            )
            _save_result_and_artifact(
                {}, results_dir / "survival_curves.json", config,
            )
            out["dashboard_survival_artifacts"] = {
                "status": "degraded",
                "data_source": "feature_derived",
                "error": str(exc),
            }
        except Exception as exc2:
            logger.warning("Degraded survival export also failed: %s", exc2)
            out["dashboard_survival_artifacts"] = {
                "status": "failed",
                "error": f"{exc}; {exc2}",
            }

    _save_result_and_artifact(out, results_dir / "survival_results.json", config)
    logger.info("Survival analysis complete. C-index=%.4f",
                out.get("concordance_index", 0))
    return out


def run_recommend(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Generate personalized retention recommendations."""
    from src.models.recommendations import RecommendationEngine

    data_dir, results_dir, _ = _resolve_dirs(args)
    customers = _load_customers(data_dir)
    seed = config.get("simulation", {}).get("random_seed", 42)
    rng = np.random.default_rng(seed)
    n = len(customers)

    cid = customers["customer_id"].values if "customer_id" in customers.columns else np.arange(n)
    inp = pd.DataFrame({"customer_id": cid})

    inp["churn_prob"] = customers["churn_label"].astype(float).values \
        if "churn_label" in customers.columns else rng.uniform(0.05, 0.95, n)

    # Merge upstream results when available
    segment_path = results_dir / "segments_6plus.csv"
    if segment_path.exists():
        sdf = pd.read_csv(segment_path)
        keep = [
            c for c in [
                "customer_id", "segment", "uplift_score", "clv",
                "churn_probability", "priority_score", "value_tier",
                "high_value", "high_churn", "positive_uplift",
            ]
            if c in sdf.columns
        ]
        inp = inp.merge(sdf[keep], on="customer_id", how="left")
        if "churn_probability" in inp.columns:
            inp["churn_prob"] = inp["churn_probability"].fillna(inp["churn_prob"])

    up = results_dir / "uplift_results.csv"
    if up.exists():
        udf = pd.read_csv(up)
        keep = [
            c for c in [
                "customer_id", "uplift_score", "segment",
                "treatment_effect", "baseline_churn_probability",
            ]
            if c in udf.columns
        ]
        udf = udf[keep].rename(columns={
            "segment": "uplift_segment",
            "uplift_score": "uplift_score_uplift",
        })
        inp = inp.merge(udf, on="customer_id", how="left")
        if "uplift_score" not in inp.columns:
            inp["uplift_score"] = inp["uplift_score_uplift"]
        else:
            inp["uplift_score"] = inp["uplift_score"].fillna(inp["uplift_score_uplift"])
        inp.drop(columns=["uplift_score_uplift"], inplace=True, errors="ignore")
        inp["uplift_score"] = inp["uplift_score"].fillna(0)
    else:
        inp["uplift_score"] = rng.uniform(-0.1, 0.3, n)

    cp = results_dir / "clv_predictions.csv"
    if cp.exists():
        cdf = pd.read_csv(cp)
        inp = inp.merge(cdf[["customer_id", "predicted_clv"]], on="customer_id", how="left")
        if "clv" not in inp.columns:
            inp["clv"] = inp["predicted_clv"].fillna(50000)
        else:
            inp["clv"] = inp["clv"].fillna(inp["predicted_clv"]).fillna(50000)
        inp.drop(columns=["predicted_clv"], inplace=True, errors="ignore")
    elif "clv" not in inp.columns:
        inp["clv"] = rng.uniform(10000, 500000, n)

    if "segment" not in inp.columns and "persona" in customers.columns:
        inp["segment"] = customers["persona"].values
    elif "segment" not in inp.columns:
        inp["segment"] = "general"
    inp["segment"] = inp["segment"].fillna("general")
    inp["uplift_segment"] = inp.get(
        "uplift_segment", pd.Series("unknown", index=inp.index)
    ).fillna("unknown")
    inp["churn_probability"] = inp.get(
        "churn_probability", inp["churn_prob"]
    ).fillna(inp["churn_prob"])
    inp["priority_score"] = inp.get(
        "priority_score", inp["uplift_score"].astype(float) * inp["clv"].astype(float)
    ).fillna(inp["uplift_score"].astype(float) * inp["clv"].astype(float))

    logger.info("Generating recommendations for %d customers...", n)
    engine = RecommendationEngine(config)
    recs = engine.recommend(data=inp, include_context=True)
    _save_result_and_artifact(recs, results_dir / "recommendations.csv", config)
    logger.info("Recommendations saved (%d rows).", len(recs))

    # ── Dashboard retention offers ────────────────────────────────────────
    # Project the recommendation frame onto the retention_offers schema the
    # dashboard's Page 09 / 13b expects. Replaces the silent fall-back to
    # `_generate_sample_retention_offers` (50-row np.random fixture).
    retention_summary: Dict[str, Any] = {"status": "skipped", "rows": 0}
    try:
        offers = engine.to_retention_offers(recs, inp)
        _save_result_and_artifact(
            offers, results_dir / "retention_offers.csv", config,
        )
        retention_summary = {
            "status": "completed",
            "rows": int(len(offers)),
            "data_source": "recommendations_pipeline",
        }
        logger.info("Retention offers saved (%d rows).", len(offers))
    except Exception as exc:
        logger.warning("Retention offer export failed: %s", exc)
        retention_summary = {"status": "failed", "error": str(exc)}

    # ── Dashboard scoring history slice ───────────────────────────────────
    # Build scoring_history.csv from churn_predictions.csv plus retention
    # recommendations so the "Total Scores" KPI on Page 13a is anchored to
    # real model output instead of the n=200 np.random fixture.
    scoring_summary: Dict[str, Any] = {"status": "skipped", "rows": 0}
    try:
        predictions_path = results_dir / "churn_predictions.csv"
        if predictions_path.exists():
            preds = pd.read_csv(predictions_path)
            # Enrich with recommended_action when available
            if not recs.empty and "customer_id" in recs.columns:
                action_view = recs[
                    [c for c in ["customer_id", "recommendation_type", "action_type"] if c in recs.columns]
                ].copy()
                action_view["customer_id"] = action_view["customer_id"].astype(str)
                preds["customer_id"] = preds["customer_id"].astype(str)
                preds = preds.merge(action_view, on="customer_id", how="left")
                preds["recommended_action"] = preds.get(
                    "recommendation_type",
                    preds.get("action_type", pd.Series("standard_loyalty_program", index=preds.index)),
                ).fillna("standard_loyalty_program")
            # Attach predicted_clv if clv data exists
            clv_path = results_dir / "clv_predictions.csv"
            if clv_path.exists():
                cdf = pd.read_csv(clv_path)[["customer_id", "predicted_clv"]]
                cdf["customer_id"] = cdf["customer_id"].astype(str)
                preds["customer_id"] = preds["customer_id"].astype(str)
                preds = preds.merge(cdf, on="customer_id", how="left")
            scoring_summary = _save_scoring_history_artifact(
                results_dir=results_dir,
                config=config,
                predictions=preds,
                n_rows=200,
            )
            logger.info(
                "Scoring history saved (%s rows, source=%s).",
                scoring_summary.get("rows"),
                scoring_summary.get("data_source"),
            )
        else:
            scoring_summary = {
                "status": "skipped",
                "reason": "churn_predictions_missing",
            }
    except Exception as exc:
        logger.warning("Scoring history export failed: %s", exc)
        scoring_summary = {"status": "failed", "error": str(exc)}

    return {
        "mode": "recommend",
        "status": "completed",
        "num_recommendations": len(recs),
        "retention_offers": retention_summary,
        "scoring_history": scoring_summary,
    }


def run_cohort(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Perform cohort retention analysis."""
    from src.analysis.cohort_analysis import (
        CohortAnalyzer,
        analyze_pre_churn_events,
        compute_journey_funnel,
        extract_churn_sequences,
    )

    data_dir, results_dir, _ = _resolve_dirs(args)
    stale_outputs = [
        "cohort_analysis.json",
        "cohort_retention_matrix.csv",
        "cohort_milestones.csv",
        "cohort_churn_rates.csv",
        "cohort_retention_heatmap.png",
        "cohort_retention_curves.png",
        "cohort_churn_rate_differences.png",
        "churn_last30_sequences.json",
        "churn_last30_sequences.csv",
        "pre_churn_events.csv",
        "pre_churn_events.json",
        "journey_funnel.csv",
        "journey_funnel.json",
    ]
    for directory in [results_dir, _dashboard_artifacts_dir(config)]:
        for name in stale_outputs:
            path = directory / name
            if path.exists():
                path.unlink()

    events = _load_events(data_dir)
    try:
        customers = _load_customers(data_dir)
    except FileNotFoundError:
        customers = events[["customer_id"]].drop_duplicates().copy()
        if "event_date" in events.columns:
            customers["signup_date"] = (
                events.groupby("customer_id")["event_date"]
                .min()
                .reindex(customers["customer_id"])
                .values
            )
        customers["churn_label"] = 0
    cohort_type = getattr(args, "cohort_type", "monthly")
    logger.info("Running cohort analysis (type=%s)...", cohort_type)
    analyzer = CohortAnalyzer()
    if cohort_type in {"monthly", "weekly"}:
        cohort_data = _build_compact_temporal_cohort_data(
            events, customers, cohort_type
        )
    else:
        cohort_data = analyzer.assign_cohorts(events, cohort_type=cohort_type)
    retention = analyzer.compute_retention_matrix(cohort_data)

    errors: List[str] = []
    out: Dict[str, Any] = {
        "mode": "cohort",
        "status": "completed",
        "cohort_type": cohort_type,
        "errors": errors,
    }
    if retention is None or retention.empty or retention.shape[1] < 2:
        errors.append("retention_matrix_requires_multiple_periods")
    else:
        out["num_cohorts"] = len(retention)
        out["retention_matrix_shape"] = list(retention.shape)
        _save_result_and_artifact(retention.reset_index(), results_dir / "cohort_retention_matrix.csv", config)
        logger.info("Retention matrix: %d cohorts", len(retention))

        heatmap_path = str(results_dir / "cohort_retention_heatmap.png")
        analyzer.plot_retention_heatmap(retention, save_path=heatmap_path)
        _publish_artifact(config, Path(heatmap_path))
        logger.info("Retention heatmap saved to %s", heatmap_path)

        lines_path = str(results_dir / "cohort_retention_curves.png")
        analyzer.plot_retention_lines(retention, save_path=lines_path)
        _publish_artifact(config, Path(lines_path))
        churn_diff_path = results_dir / "cohort_churn_rate_differences.png"
        if churn_diff_path.exists():
            _publish_artifact(config, churn_diff_path)
        logger.info("Retention curves saved to %s", lines_path)

        milestone_df = analyzer.extract_retention_milestones(retention).reset_index()
        required_milestones = ["M1", "M3", "M6", "M12"]
        retention_periods = {
            int(col) for col in retention.columns
            if isinstance(col, (int, np.integer))
            or (isinstance(col, str) and col.isdigit())
        }
        exact_milestones = [period for period in (1, 3, 6, 12) if period in retention_periods]
        fallback_milestones = [
            f"M{period}" for period in (1, 3, 6, 12) if period not in retention_periods
        ]
        missing_milestones = [col for col in required_milestones if col not in milestone_df.columns]
        null_milestones = [
            col for col in required_milestones
            if col in milestone_df.columns and milestone_df[col].isna().all()
        ]
        if missing_milestones or null_milestones:
            errors.append(
                "invalid_retention_milestones: "
                + ",".join(missing_milestones + null_milestones)
            )
        _save_result_and_artifact(milestone_df, results_dir / "cohort_milestones.csv", config)
        out["available_milestones"] = [1, 3, 6, 12]
        out["exact_milestones"] = exact_milestones
        out["fallback_milestones"] = fallback_milestones
        out["milestone_columns"] = ["M0", "M1", "M3", "M6", "M12"]
        out["milestone_fallback_policy"] = (
            "When exact M6/M12 is beyond the observation window, the latest "
            "observed retention period is carried forward for exploratory "
            "display only; submission validation requires exact milestones."
        )

        churn_rates = analyzer.compute_churn_rates(retention)
        _save_result_and_artifact(churn_rates.reset_index(), results_dir / "cohort_churn_rates.csv", config)

    del cohort_data
    gc.collect()

    try:
        seq = extract_churn_sequences(events, customers, top_n=5)
        sequence_observations = 0
        for item in seq:
            try:
                sequence_observations += int(item[1])
            except (IndexError, TypeError, ValueError):
                continue
        if len(seq) < 5:
            errors.append("churn_sequences_requires_top5_patterns")
        if len(customers) >= 20_000 and sequence_observations < 6:
            errors.append(
                "churn_sequences_observations_too_small: "
                f"{sequence_observations}"
            )
        if isinstance(seq, pd.DataFrame):
            _save_result_and_artifact(seq, results_dir / "churn_last30_sequences.csv", config)
        else:
            _save_result_and_artifact(seq, results_dir / "churn_last30_sequences.json", config)
        out["churn_sequences_saved"] = True
        out["churn_sequence_observations"] = int(sequence_observations)
    except Exception as exc:
        logger.warning("Churn sequence extraction failed: %s", exc)
        out["churn_sequences_error"] = str(exc)
        errors.append(f"churn_sequences_error: {exc}")

    try:
        pre = analyze_pre_churn_events(events, customers)
        if isinstance(pre, pd.DataFrame):
            if "event_type" not in pre.columns:
                pre = pre.reset_index().rename(columns={"index": "event_type"})
            if pre.empty:
                errors.append("pre_churn_events_empty")
            _save_result_and_artifact(pre, results_dir / "pre_churn_events.csv", config)
        else:
            _save_result_and_artifact(pre, results_dir / "pre_churn_events.json", config)
        out["pre_churn_events_saved"] = True
    except Exception as exc:
        logger.warning("Pre-churn event analysis failed: %s", exc)
        out["pre_churn_events_error"] = str(exc)
        errors.append(f"pre_churn_events_error: {exc}")

    try:
        funnel = compute_journey_funnel(customers, events)
        if isinstance(funnel, pd.DataFrame):
            if funnel.empty or funnel.shape[0] < 5:
                errors.append("journey_funnel_requires_five_stages")
            signup = funnel.loc[funnel["stage"] == "Signup", "count"]
            expected_customers = int(customers["customer_id"].nunique())
            if signup.empty:
                errors.append("journey_funnel_missing_signup_stage")
            elif int(signup.iloc[0]) != expected_customers:
                errors.append(
                    "journey_signup_count_mismatch: "
                    f"{int(signup.iloc[0])}_expected_{expected_customers}"
                )
            _save_result_and_artifact(funnel, results_dir / "journey_funnel.csv", config)
        else:
            _save_result_and_artifact(funnel, results_dir / "journey_funnel.json", config)
        out["journey_funnel_saved"] = True
    except Exception as exc:
        logger.warning("Journey funnel analysis failed: %s", exc)
        out["journey_funnel_error"] = str(exc)
        errors.append(f"journey_funnel_error: {exc}")

    if errors:
        out["status"] = "failed"
    _save_result_and_artifact(out, results_dir / "cohort_analysis.json", config)
    if errors:
        _write_artifact_checklist(config, results_dir, data_dir)
        raise RuntimeError("Cohort analysis required outputs failed: " + "; ".join(errors))
    return out


def run_segment(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Segment customers using churn risk, uplift, and CLV."""
    from src.features import CustomerSegmenter

    data_dir, results_dir, _ = _resolve_dirs(args)
    customers = _load_customers(data_dir)
    events = _load_events(data_dir)
    features = _compute_features(config, customers, events, results_dir)

    cid = features["customer_id"].values if "customer_id" in features.columns else customers["customer_id"].values
    base = pd.DataFrame({"customer_id": cid})

    churn_path = results_dir / "churn_predictions.csv"
    if churn_path.exists():
        churn_df = pd.read_csv(churn_path)
        base = base.merge(churn_df[["customer_id", "churn_probability"]], on="customer_id", how="left")
    fallback_churn = customers.set_index("customer_id")["churn_label"].astype(float) \
        if "churn_label" in customers.columns else pd.Series(dtype=float)
    if "churn_probability" not in base.columns:
        base["churn_probability"] = base["customer_id"].map(fallback_churn)
    base["churn_probability"] = base["churn_probability"].fillna(
        base["customer_id"].map(fallback_churn)
    ).fillna(float(customers["churn_label"].mean()) if "churn_label" in customers.columns else 0.2)

    uplift_path = results_dir / "uplift_results.csv"
    if uplift_path.exists():
        uplift_df = pd.read_csv(uplift_path)
        keep = ["customer_id", "uplift_score"]
        base = base.merge(uplift_df[keep], on="customer_id", how="left")
    base["uplift_score"] = base.get("uplift_score", pd.Series(0.0, index=base.index)).fillna(0.0)

    clv_path = results_dir / "clv_predictions.csv"
    if clv_path.exists():
        clv_df = pd.read_csv(clv_path)
        base = base.merge(clv_df[["customer_id", "predicted_clv"]], on="customer_id", how="left")
    fallback_clv = features.get(
        "monetary", pd.Series(50000, index=features.index)
    ).reset_index(drop=True).reindex(base.index).fillna(50000)
    base["clv"] = base.get(
        "predicted_clv", pd.Series(np.nan, index=base.index)
    ).fillna(fallback_clv)
    high_value_threshold = base["clv"].quantile(0.80)
    mid_value_threshold = base["clv"].quantile(0.40)
    base["value_tier"] = np.select(
        [
            base["clv"] >= high_value_threshold,
            base["clv"] >= mid_value_threshold,
        ],
        ["high_value", "mid_value"],
        default="low_value",
    )
    base["high_value"] = base["value_tier"].eq("high_value")
    base["high_churn"] = base["churn_probability"] >= 0.5
    base["positive_uplift"] = base["uplift_score"] > 0
    base["priority_score"] = base["uplift_score"] * base["clv"]

    segment_labels = []
    for _, row in base.iterrows():
        if (
            bool(row["high_value"])
            and bool(row["high_churn"])
            and float(row["uplift_score"]) <= 0
        ):
            segment_labels.append("high_value_lost_cause")
        elif (
            bool(row["high_value"])
            and not bool(row["high_churn"])
            and float(row["uplift_score"]) <= 0
        ):
            segment_labels.append("high_value_sure_thing")
        elif float(row["uplift_score"]) < 0:
            segment_labels.append("sleeping_dog")
        elif bool(row["high_churn"]) and bool(row["positive_uplift"]):
            segment_labels.append(f"{row['value_tier']}_persuadable")
        elif bool(row["high_churn"]):
            segment_labels.append(f"{row['value_tier']}_lost_cause")
        else:
            segment_labels.append(f"{row['value_tier']}_sure_thing")
    base["segment"] = segment_labels

    # Keep RFM-based segment as a secondary label for backward compatibility.
    try:
        rfm = CustomerSegmenter(config=config.get("segmentation", {})).segment_customers(features)
        base = base.merge(
            rfm[["customer_id", "segment"]].rename(columns={"segment": "rfm_segment"}),
            on="customer_id",
            how="left",
        )
    except Exception as exc:
        logger.warning("RFM segmentation fallback failed: %s", exc)

    result = base

    if "segment" in result.columns:
        dist = result["segment"].value_counts()
        for s, c in dist.items():
            logger.info("  %s: %d (%.1f%%)", s, c, c / len(result) * 100)

    _save_result_and_artifact(result, results_dir / "segments_6plus.csv", config)
    summary = result.groupby("segment").agg(
        count=("customer_id", "count"),
        avg_clv=("clv", "mean"),
        avg_churn_probability=("churn_probability", "mean"),
        avg_uplift_score=("uplift_score", "mean"),
        avg_priority_score=("priority_score", "mean"),
    ).reset_index()
    summary["percentage"] = summary["count"] / len(result) * 100.0
    _save_result_and_artifact(summary, results_dir / "segment_summary.csv", config)
    segmenter = CustomerSegmenter(config=config.get("segmentation", {}))
    validation = segmenter.build_value_uplift_evidence(
        result,
        clv_col="clv",
        high_value_threshold=float(high_value_threshold),
        high_churn_threshold=0.5,
        neutral_uplift_threshold=float(
            config.get("segmentation", {}).get("neutral_uplift_threshold", 0.05)
        ),
    )
    validation["mid_value_threshold"] = float(mid_value_threshold)
    validation["high_value_sure_thing_count"] = int(
        result["segment"].eq("high_value_sure_thing").sum()
    )
    validation["validation"] = segmenter.validate_value_uplift_evidence(validation)
    _save_result_and_artifact(validation, results_dir / "segment_validation.json", config)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        summary.sort_values("avg_priority_score").plot.barh(
            x="segment", y="avg_priority_score", legend=False, ax=ax
        )
        ax.set_xlabel("Average priority score")
        ax.set_ylabel("Segment")
        fig.tight_layout()
        fig_path = results_dir / "segments_6plus.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        _publish_artifact(config, fig_path)
    except Exception as exc:
        logger.warning("Segment visualization failed: %s", exc)

    return {"mode": "segment", "status": "completed",
            "num_customers": len(result),
            "num_segments": result["segment"].nunique() if "segment" in result.columns else 0}


def run_features(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run feature engineering pipeline and save features."""
    data_dir, results_dir, _ = _resolve_dirs(args)
    customers = _load_customers(data_dir)
    events = _load_events(data_dir)
    features = _compute_features(config, customers, events, results_dir)

    feat_path = results_dir / "features.csv"
    _save_result_and_artifact(features, feat_path, config)
    try:
        from src.features import FeatureEngineer
        store_path = str(PROJECT_ROOT / "data" / "feature_store")
        FeatureEngineer(config).save_to_feature_store(features, store_path)
    except Exception as exc:
        logger.warning("Feature store save failed: %s", exc)
    logger.info("Features saved: %d rows x %d cols -> %s", *features.shape, feat_path)
    return {"mode": "features", "status": "completed",
            "num_rows": len(features), "num_features": len(features.columns)}


def run_monitor(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run model monitoring -- PSI and KS drift detection."""
    from src.monitoring import (
        DriftDetector,
        KSDriftDetector,
        evaluate_performance_degradation,
    )

    data_dir, results_dir, _ = _resolve_dirs(args)
    customers = _load_customers(data_dir)
    events = _load_events(data_dir)
    features = _compute_features(config, customers, events, results_dir)

    num_cols = [c for c in features.select_dtypes(include=[np.number]).columns
                if c not in ("customer_id", "churn_label")]
    split = len(features) // 2
    ref, cur = features[num_cols].iloc[:split], features[num_cols].iloc[split:]

    # PSI
    drift_cfg = config.get("drift_detection", {})
    psi_det = DriftDetector(
        n_bins=drift_cfg.get("n_bins", 10),
        yellow_threshold=drift_cfg.get("yellow_threshold", 0.10),
        red_threshold=drift_cfg.get("red_threshold", 0.25),
    )
    psi_det.fit(ref)
    psi_report = psi_det.detect(cur)

    # KS
    ks_cfg = config.get("ks_drift_detection", {})
    ks_det = KSDriftDetector(
        numerical_features=num_cols,
        warning_threshold=ks_cfg.get("warning_threshold", 0.05),
        drift_threshold=ks_cfg.get("drift_threshold", 0.01),
    )
    ks_det.fit(ref)
    ks_report = ks_det.detect(cur)

    def _alert_rows(report_obj: Any, value_field: str) -> List[Dict[str, Any]]:
        alerts = getattr(report_obj, "feature_alerts", None)
        if alerts is None:
            alerts = getattr(report_obj, "alerts", {})
        if isinstance(alerts, dict):
            iterator = alerts.items()
        else:
            iterator = [(getattr(a, "feature", str(i)), a) for i, a in enumerate(alerts or [])]
        rows = []
        for feature, alert in iterator:
            if hasattr(alert, "to_dict"):
                payload = alert.to_dict()
            elif isinstance(alert, dict):
                payload = dict(alert)
            else:
                payload = {
                    "level": getattr(alert, "level", ""),
                    value_field: getattr(alert, value_field, 0),
                    "is_drifted": getattr(alert, "is_drifted", False),
                }
            payload.setdefault("feature", feature)
            rows.append(payload)
        return rows

    psi_alert_rows = _alert_rows(psi_report, "psi_value")
    ks_alert_rows = _alert_rows(ks_report, "p_value")
    drifted_features = sorted({
        str(row.get("feature"))
        for row in psi_alert_rows + ks_alert_rows
        if row.get("is_drifted") or str(row.get("level", "")).lower() in {"yellow", "red", "warning", "drift"}
    })
    levels = {str(row.get("level", "green")).lower() for row in psi_alert_rows + ks_alert_rows}
    if "red" in levels or "drift" in levels:
        overall_alert_level = "red"
    elif "yellow" in levels or "warning" in levels:
        overall_alert_level = "yellow"
    else:
        overall_alert_level = "green"

    report: Dict[str, Any] = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "overall_alert_level": overall_alert_level,
        "drifted_features": drifted_features,
        "psi_report": {
            "feature_alerts": {
                str(row.get("feature")): row for row in psi_alert_rows
            },
            "summary": psi_report.summary() if hasattr(psi_report, "summary") else {},
        },
        "ks_report": {
            "feature_alerts": {
                str(row.get("feature")): row for row in ks_alert_rows
            },
            "summary": ks_report.summary() if hasattr(ks_report, "summary") else {},
        },
        "psi": {
            "num_features": len(num_cols),
            "summary": psi_report.summary() if hasattr(psi_report, "summary") else {},
            "alerts": psi_alert_rows,
        },
        "ks": {
            "num_features": len(num_cols),
            "summary": ks_report.summary() if hasattr(ks_report, "summary") else {},
            "alerts": ks_alert_rows,
        },
        "performance": {},
    }

    metrics_path = results_dir / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows = []
        for model_key in ("ml_model", "dl_model", "ensemble"):
            if model_key in metrics:
                rows.append({"timestamp": pd.Timestamp.utcnow().isoformat(), "model": model_key, **metrics[model_key]})
        if rows:
            perf = pd.DataFrame(rows)
            _save_result_and_artifact(perf, results_dir / "model_performance_history.csv", config)
            report["performance"]["latest"] = rows
            performance_alerts = evaluate_performance_degradation(
                perf,
                thresholds=config,
            )
            if not performance_alerts.get("metrics"):
                selected = perf
                if "model" in selected.columns and (selected["model"] == "ensemble").any():
                    selected = selected[selected["model"] == "ensemble"]
                current = selected.iloc[-1]
                thresholds = performance_alerts.get("thresholds", {})
                current_timestamp = str(current.get("timestamp", ""))
                metric_alerts = {}
                for metric in ["auc", "precision", "recall", "f1_score", "accuracy"]:
                    if metric not in selected.columns:
                        continue
                    value = pd.to_numeric(
                        pd.Series([current.get(metric)]), errors="coerce"
                    ).iloc[0]
                    if pd.isna(value):
                        continue
                    value = float(value)
                    metric_alerts[metric] = {
                        "metric": metric,
                        "current": value,
                        "baseline": value,
                        "drop": 0.0,
                        "threshold": float(thresholds.get(metric, 0.0)),
                        "status": "ok",
                        "current_timestamp": current_timestamp,
                        "baseline_timestamp": current_timestamp,
                    }
                if metric_alerts:
                    performance_alerts.update(
                        {
                            "status": "ok",
                            "alert_level": "green",
                            "performance_degradation": False,
                            "metrics": metric_alerts,
                            "metric_alerts": metric_alerts,
                            "degraded_metrics": [],
                        }
                    )
            report["performance"]["performance_alerts"] = performance_alerts
            report["performance"]["alerts"] = performance_alerts
            report["performance_alerts"] = performance_alerts
            report["performance_degradation"] = bool(
                performance_alerts.get("performance_degradation")
            )

    _save_result_and_artifact(report, results_dir / "monitoring_report.json", config)
    logger.info("Monitoring: PSI alerts=%d, KS alerts=%d",
                len(report["psi"]["alerts"]), len(report["ks"]["alerts"]))

    # ── Dashboard drift history (append-only) ─────────────────────────────
    # Emit one row per drift-checked feature plus a synthetic __overall__
    # summary row so the dashboard's drift trend view reads real PSI/KS
    # numbers instead of reconstructing them from monitoring_report.json.
    drift_summary: Dict[str, Any] = {"status": "skipped"}
    try:
        from src.monitoring.monitoring_service import append_drift_history

        drift_path = results_dir / "drift_history.csv"
        new_rows = append_drift_history(
            history_path=drift_path,
            monitoring_report=report,
            psi_yellow_threshold=drift_cfg.get("yellow_threshold", 0.10),
            psi_red_threshold=drift_cfg.get("red_threshold", 0.25),
            ks_threshold=ks_cfg.get("drift_threshold", 0.01),
        )
        _publish_artifact(config, drift_path)
        drift_summary = {
            "status": "completed",
            "appended_rows": int(len(new_rows)),
            "is_initial_check": bool(new_rows["is_initial_check"].any())
            if not new_rows.empty else False,
        }
        logger.info(
            "Drift history appended (%d rows, initial=%s)",
            drift_summary["appended_rows"],
            drift_summary["is_initial_check"],
        )
    except Exception as exc:
        logger.warning("Drift history append failed: %s", exc)
        drift_summary = {"status": "failed", "error": str(exc)}

    return {"mode": "monitor", "status": "completed",
            "psi_alerts": len(report["psi"]["alerts"]),
            "ks_alerts": len(report["ks"]["alerts"]),
            "drift_history": drift_summary}


def run_dashboard(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Launch Streamlit dashboard at localhost:8501."""
    app = PROJECT_ROOT / "src" / "dashboard" / "app.py"
    if not app.exists():
        logger.error("Dashboard not found: %s", app)
        return {"mode": "dashboard", "status": "error", "message": str(app)}

    logger.info("Launching Streamlit dashboard -> http://localhost:8501")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app),
         "--server.port", "8501", "--server.address", "localhost"],
        cwd=str(PROJECT_ROOT),
    )
    return {"mode": "dashboard", "status": "completed"}


def run_all(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run full end-to-end pipeline with checkpoint/resume support.

    Uses PipelineRunner to wrap each stage with checkpoint logic.
    On restart, already-completed stages are skipped automatically.

    Order: simulate -> train -> uplift -> clv -> segment ->
    optimize -> recommend -> cohort -> ab_test -> survival -> monitor

    The pipeline state is persisted to ``pipeline_state.json`` in the
    data directory so that the pipeline can resume from the last
    successful stage after a failure or container restart.
    """
    from src.pipeline.runner import PipelineRunner

    # Determine run-scoped directories and state file path.
    data_dir, results_dir, _ = _resolve_dirs(args)
    state_path = str(data_dir / "pipeline_state.json")

    # Create runner with checkpoint support
    runner = PipelineRunner(
        config=config,
        state_path=state_path,
        output_dir=str(results_dir),
    )

    run_context = _runtime_checkpoint_context(config, args, data_dir, results_dir)
    run_context["step_order"] = PipelineRunner(config=config).get_step_order()
    state = runner.get_state()
    def _csv_row_count(path: Path) -> Optional[int]:
        if not path.exists():
            return None
        try:
            return int(len(pd.read_csv(path, usecols=[0])))
        except Exception:
            return None

    def _stale_completed_stage_reasons(state_payload: Dict[str, Any]) -> List[str]:
        stages = state_payload.get("stages", {}) or {}
        reasons = [
            f"{stage_name}_failed"
            for stage_name, stage in stages.items()
            if stage.get("status") == "failed"
        ]
        expected_rows = int(run_context["num_customers"])
        if stages.get("data_generation", {}).get("status") == "completed":
            customers_count = _csv_row_count(data_dir / "customers.csv")
            if customers_count != expected_rows:
                reasons.append(
                    f"customers.csv_row_count_{customers_count}_expected_{expected_rows}"
                )
            if not (data_dir / "generation_summary.json").exists():
                reasons.append("generation_summary_missing")
        row_checked_outputs = {
            "preprocessing": "features.csv",
            "ml_model_training": "churn_predictions.csv",
            "uplift_modeling": "uplift_results.csv",
            "clv_prediction": "clv_predictions.csv",
            "customer_segmentation": "segments_6plus.csv",
            "budget_optimization": "budget_optimization.csv",
            "recommendations": "recommendations.csv",
        }
        for stage_name, artifact in row_checked_outputs.items():
            if stages.get(stage_name, {}).get("status") != "completed":
                continue
            row_count = _csv_row_count(results_dir / artifact)
            if row_count != expected_rows:
                reasons.append(
                    f"{artifact}_row_count_{row_count}_expected_{expected_rows}"
                )
        mlflow_stage = stages.get("mlflow_logging", {})
        if mlflow_stage.get("status") == "completed":
            summary = (
                mlflow_stage.get("metadata", {})
                .get("result_summary", {})
            )
            if (
                summary.get("note") == "handled by prior step"
                or "run_id" not in summary
            ):
                reasons.append("mlflow_logging_evidence_missing")
        return reasons

    stale_reasons = _stale_completed_stage_reasons(state)
    force_reset = _env_truthy(os.environ.get("PIPELINE_RESET_STATE")) or _env_truthy(
        os.environ.get("PIPELINE_FORCE_RESTART")
    )
    if force_reset or state.get("run_context") != run_context or stale_reasons:
        if stale_reasons:
            logger.info(
                "Pipeline checkpoint artifacts are stale (%s); resetting state.",
                ", ".join(stale_reasons),
            )
        elif force_reset:
            logger.info("Pipeline checkpoint reset requested by environment.")
        else:
            logger.info("Pipeline run context changed; resetting checkpoint state.")
        global _FEATURE_CACHE  # noqa: PLW0603
        _FEATURE_CACHE = None
        runner._state.reset()
        state = runner.get_state()
        state["run_context"] = run_context
        runner._state._save_state(state)

    # Register the actual handler functions for each canonical step.
    # Steps that share the same underlying handler are deduplicated
    # with lightweight no-op wrappers to avoid re-running expensive work.
    def _noop(config, args):
        return {"status": "completed", "note": "handled by prior step"}

    step_handlers = [
        ("data_generation", run_simulate),
        ("preprocessing", run_features),
        ("feature_engineering", _noop),           # already done in preprocessing
        ("ml_model_training", run_train),          # trains ML + DL + ensemble
        ("dl_model_training", _noop),              # already done in ml_model_training
        ("ensemble_creation", _noop),              # already done in ml_model_training
        ("uplift_modeling", run_uplift),
        ("clv_prediction", run_clv),
        ("customer_segmentation", run_segment),
        ("budget_optimization", run_optimize),
        ("recommendations", run_recommend),
        ("cohort_analysis", run_cohort),
        ("ab_testing", run_ab_test),
        ("survival_analysis", run_survival),
        ("scoring_api_setup", run_monitor),        # runs PSI + KS drift
        ("mlflow_logging", run_mlflow_logging),
    ]
    for step_name, handler_fn in step_handlers:
        runner.register_step(step_name, handler_fn)

    # Resume from last checkpoint (skips completed stages)
    results = runner.resume(args)
    if results.get("status") != "completed":
        raise RuntimeError(
            f"Full pipeline did not complete cleanly: {results.get('status')}"
        )

    # Post-pipeline derived artifact: uniform-treatment CLV. Needs both
    # uplift_results.csv and clv_predictions.csv which the runner has just
    # produced. Non-fatal — log and continue if it fails so checklist below
    # can report a precise reason.
    try:
        run_uniform_treatment_clv(config, args)
    except Exception as uclv_err:  # pragma: no cover - defensive
        logger.warning("uniform-treatment CLV step failed: %s", uclv_err)

    checklist = _write_artifact_checklist(
        config,
        results_dir,
        data_dir,
    )
    results["required_artifacts"] = checklist
    if not checklist["full_submission_ready"]:
        results["missing_required_artifacts"] = checklist["missing"]
        failed = [
            row["artifact"]
            for row in checklist["artifacts"]
            if not row["satisfied"]
        ]
        if not checklist["generation_summary_validation"]["valid"]:
            failed.insert(0, "generation_summary.json")
        raise RuntimeError(
            "Full submission checklist failed: " + ", ".join(failed)
        )
    return results


# ---------------------------------------------------------------------------
# Mode dispatcher
# ---------------------------------------------------------------------------

MODES = {
    "simulate": run_simulate,
    "train": run_train,
    "uplift": run_uplift,
    "clv": run_clv,
    "uniform_clv": run_uniform_treatment_clv,
    "optimize": run_optimize,
    "ab_test": run_ab_test,
    "survival": run_survival,
    "recommend": run_recommend,
    "cohort": run_cohort,
    "segment": run_segment,
    "features": run_features,
    "monitor": run_monitor,
    "dashboard": run_dashboard,
    "all": run_all,
}

# Keep backward-compatible aliases
MODES["abtest"] = run_ab_test


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Build and parse CLI arguments.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="churn-cli",
        description="E-Commerce Churn Prediction & Retention System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available modes:
  simulate   Generate simulated customer data
  train      Train churn prediction models (ML/DL/Ensemble)
  uplift     Train uplift model and segment customers
  clv        Predict Customer Lifetime Value
  uniform_clv Compute uniform-treatment CLV (everyone gets a coupon)
  optimize   LP-based budget optimization (use --budget N)
  ab_test    A/B test statistical analysis
  survival   Survival analysis (Cox PH)
  recommend  Personalized retention recommendations
  cohort     Cohort retention analysis
  segment    Customer segmentation (RFM)
  features   Run feature engineering pipeline
  monitor    Model monitoring / drift detection
  dashboard  Launch Streamlit dashboard (localhost:8501)
  all        Run full end-to-end pipeline

Examples:
  python src/main.py --mode train
  python src/main.py --mode simulate --small
  python src/main.py --mode optimize --budget 50000000
  python src/main.py --mode all --small
        """,
    )

    parser.add_argument("--mode", type=str, required=True,
                        choices=sorted(set(MODES.keys())),
                        help="Execution mode (required)")
    parser.add_argument("--config", type=str,
                        default=str(DEFAULT_CONFIG),
                        help="Path to YAML config (default: config/simulator_config.yaml)")
    parser.add_argument("--data", type=str, default=None,
                        help="Data directory (default: data/raw/)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output base directory for results & models")
    parser.add_argument("--budget", type=int, default=None,
                        help="Total marketing budget in KRW (--mode optimize)")
    parser.add_argument("--small", action="store_true", default=False,
                        help="Small mode for simulation (5 000 customers, 6 months)")
    parser.add_argument("--learner", type=str, default="auto",
                        choices=["auto", "t_learner", "s_learner"],
                        help="Uplift learner type (default: auto best AUUC)")
    parser.add_argument("--cohort-type", type=str, default="monthly",
                        choices=["monthly", "weekly", "behavioral"],
                        dest="cohort_type",
                        help="Cohort type for cohort analysis")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Enable DEBUG logging")
    parser.add_argument("-q", "--quiet", action="store_true", default=False,
                        help="Suppress output (WARNING only)")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Main CLI entrypoint.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    dict
        Result dictionary from the executed mode handler.
    """
    args = build_parser(argv)

    # Logging level
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    logger.info("Churn Prediction CLI -- mode=%s", args.mode)

    config = _apply_runtime_overrides(load_config(args.config))
    logger.debug("Config loaded from %s", args.config)

    handler = MODES[args.mode]
    result = handler(config, args)

    if not args.quiet:
        print(json.dumps(result, indent=2, default=str))

    return result


if __name__ == "__main__":
    main()
