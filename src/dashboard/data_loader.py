"""
Dashboard Data Loader Module.

Loads predictions, model metrics, A/B test results, budget optimization
results, survival data, recommendations, uplift results, and CLV data
from file-based feature store artifacts for the Streamlit dashboard.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default artifacts directory
DEFAULT_ARTIFACTS_DIR = Path("data/artifacts")
DEFAULT_RESULTS_DIR = Path("results")


@dataclass
class DashboardArtifact:
    """Wraps a loaded dashboard artifact with provenance metadata.

    Views can opt-in via ``loader.load_*(as_artifact=True)`` to receive a
    ``DashboardArtifact`` and use ``.is_real`` to decide whether to render
    KPI cards or fall back to a clearly-labeled empty state. The default
    return type of each ``load_*`` method is unchanged for backward
    compatibility — the loader silently records the issue on
    ``loader.get_artifact_issue(name)``.
    """

    data: Any
    is_real: bool
    source_path: Optional[str] = None
    reason: Optional[str] = None
    name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.is_real


class DashboardDataLoader:
    """Loads all dashboard data from file-based feature store.

    Attributes:
        config: Parsed YAML configuration dictionary.
        artifacts_dir: Path to the artifacts directory.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize data loader from config.

        Args:
            config: Parsed simulator_config.yaml dict.
        """
        self.config = config
        dashboard_cfg = config.get("dashboard", {}) or {}
        self.artifacts_dir = Path(
            dashboard_cfg.get("artifacts_dir", str(DEFAULT_ARTIFACTS_DIR))
        )
        self.results_dir = Path(
            dashboard_cfg.get("results_dir", str(DEFAULT_RESULTS_DIR))
        )
        if "results_dir" in dashboard_cfg:
            self.search_dirs = [self.results_dir, self.artifacts_dir]
        elif "artifacts_dir" in dashboard_cfg:
            self.search_dirs = [self.artifacts_dir, self.results_dir]
        else:
            self.search_dirs = [self.results_dir, self.artifacts_dir]
        self._artifact_issues: Dict[str, str] = {}
        # is_real_* flags: True when the most recent load_* call returned a
        # real on-disk artifact, False otherwise. Views can also use
        # get_artifact_issue(name) to read the human-readable reason.
        self._is_real: Dict[str, bool] = {}

    def is_real_artifact(self, artifact_name: str) -> bool:
        """Return True if the most recent load for this artifact was real."""
        return bool(self._is_real.get(artifact_name, False))

    def _mark_real(self, artifact_name: str) -> None:
        self._is_real[artifact_name] = True
        self._clear_artifact_issue(artifact_name)

    def _mark_missing(self, artifact_name: str, message: str) -> None:
        self._is_real[artifact_name] = False
        self._record_artifact_issue(artifact_name, message)

    def _record_artifact_issue(self, artifact_name: str, message: str) -> None:
        """Remember a dashboard-visible issue for the most recent load."""
        self._artifact_issues[artifact_name] = message
        logger.warning("%s: %s", artifact_name, message)

    def _clear_artifact_issue(self, artifact_name: str) -> None:
        self._artifact_issues.pop(artifact_name, None)

    def get_artifact_issue(self, artifact_name: str) -> Optional[str]:
        """Return the latest dashboard-visible issue for a loader."""
        return self._artifact_issues.get(artifact_name)

    def get_artifact_issues(self) -> Dict[str, str]:
        """Return current dashboard-visible loader issues."""
        return dict(self._artifact_issues)

    def _resolve_existing_path(self, *candidates: str) -> Optional[Path]:
        """Find the first existing file across dashboard artifact locations."""
        for directory in self.search_dirs:
            for candidate in candidates:
                path = directory / candidate
                if path.exists():
                    return path
        return None

    def _read_json(self, path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _empty_frame(
        columns: List[str],
        artifact_name: str,
        issue: str,
    ) -> pd.DataFrame:
        df = pd.DataFrame(columns=columns)
        df.attrs["artifact_name"] = artifact_name
        df.attrs["dashboard_issue"] = issue
        return df

    def _required_csv(
        self,
        artifact_name: str,
        candidates: List[str],
        required_columns: List[str],
    ) -> Optional[pd.DataFrame]:
        path = self._resolve_existing_path(*candidates)
        if path is None:
            self._is_real[artifact_name] = False
            self._record_artifact_issue(
                artifact_name,
                f"Required artifact missing: one of {', '.join(candidates)}.",
            )
            return None

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            self._is_real[artifact_name] = False
            self._record_artifact_issue(
                artifact_name,
                f"Required artifact {path} could not be read: {exc}.",
            )
            return None

        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            self._is_real[artifact_name] = False
            self._record_artifact_issue(
                artifact_name,
                f"Required artifact {path} is invalid; missing columns: {', '.join(missing)}.",
            )
            return None
        if df.empty:
            self._is_real[artifact_name] = False
            self._record_artifact_issue(
                artifact_name,
                f"Required artifact {path} is empty.",
            )
            return None

        self._is_real[artifact_name] = True
        self._clear_artifact_issue(artifact_name)
        return df

    def _required_json(
        self,
        artifact_name: str,
        candidates: List[str],
        required_keys: List[str],
    ) -> Optional[Dict[str, Any]]:
        path = self._resolve_existing_path(*candidates)
        if path is None:
            self._is_real[artifact_name] = False
            self._record_artifact_issue(
                artifact_name,
                f"Required artifact missing: one of {', '.join(candidates)}.",
            )
            return None

        try:
            payload = self._read_json(path)
        except Exception as exc:
            self._is_real[artifact_name] = False
            self._record_artifact_issue(
                artifact_name,
                f"Required artifact {path} could not be read: {exc}.",
            )
            return None
        if not isinstance(payload, dict):
            self._is_real[artifact_name] = False
            self._record_artifact_issue(
                artifact_name,
                f"Required artifact {path} is invalid; expected a JSON object.",
            )
            return None

        missing = [k for k in required_keys if k not in payload]
        if missing:
            self._is_real[artifact_name] = False
            self._record_artifact_issue(
                artifact_name,
                f"Required artifact {path} is invalid; missing keys: {', '.join(missing)}.",
            )
            return None

        self._is_real[artifact_name] = True
        self._clear_artifact_issue(artifact_name)
        return payload

    def _customers_path(self) -> Optional[Path]:
        for path in [
            self.results_dir / "customers.csv",
            self.artifacts_dir / "customers.csv",
            Path("data/raw/customers.csv"),
        ]:
            if path.exists():
                return path
        return None

    def _raw_events_path(self) -> Optional[Path]:
        """Return the simulator event log path when available."""
        raw_dir = Path(
            (self.config.get("dashboard", {}) or {}).get(
                "raw_data_dir", "data/raw"
            )
        )
        for path in [
            raw_dir / "events.parquet",
            raw_dir / "events.csv",
        ]:
            if path.exists():
                return path
        return None

    def _load_metric_timeseries(self, metric_name: str) -> pd.DataFrame:
        """Load time-series metrics from dedicated files or MLflow exports."""
        path = self._resolve_existing_path(
            f"{metric_name}_history.csv",
            f"{metric_name}_timeseries.csv",
            "model_performance_timeseries.csv",
            "model_performance_history.csv",
        )
        if path is not None:
            df = pd.read_csv(path)
            if "model_type" not in df.columns and "model" in df.columns:
                df = df.rename(columns={"model": "model_type"})
            if metric_name in df.columns:
                cols = [c for c in ["timestamp", "model_type", metric_name] if c in df.columns]
                return df[cols]
            value_cols = [
                c for c in df.columns
                if c.lower() in {metric_name.lower(), f"{metric_name.lower()}_value"}
            ]
            if value_cols:
                rename_map = {value_cols[0]: metric_name}
                return df.rename(columns=rename_map)

        runs = self.load_mlflow_runs()
        if metric_name in runs.columns:
            cols = [c for c in ["timestamp", "model_type", metric_name] if c in runs.columns]
            return runs[cols].copy()
        return pd.DataFrame(columns=["timestamp", "model_type", metric_name])

    def _adapt_budget_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map pipeline budget outputs to dashboard schema."""
        adapted = df.copy()
        if "allocated_budget_krw" not in adapted.columns and "allocated_budget" in adapted.columns:
            adapted["allocated_budget_krw"] = adapted["allocated_budget"]
        if "expected_revenue_saved_krw" not in adapted.columns:
            for candidate in [
                "expected_revenue_saved",
                "retained_value",
                "expected_retained_value",
            ]:
                if candidate in adapted.columns:
                    adapted["expected_revenue_saved_krw"] = adapted[candidate]
                    break
        if "expected_retained" not in adapted.columns:
            if "customers_treated" in adapted.columns:
                adapted["expected_retained"] = adapted["customers_treated"]
            elif "customer_id" in adapted.columns:
                adapted["expected_retained"] = 1
        if "roi" not in adapted.columns:
            spent = adapted.get("allocated_budget_krw", pd.Series(dtype=float)).replace(0, np.nan)
            revenue = adapted.get("expected_revenue_saved_krw", pd.Series(dtype=float))
            adapted["roi"] = (revenue / spent).fillna(0.0)
        if "segment" not in adapted.columns:
            if "customer_id" in adapted.columns:
                adapted["segment"] = "all_customers"
            else:
                adapted["segment"] = [f"scenario_{i+1}" for i in range(len(adapted))]

        required = [
            "segment", "allocated_budget_krw", "expected_retained",
            "expected_revenue_saved_krw", "roi",
        ]
        for col in required:
            if col not in adapted.columns:
                adapted[col] = 0.0 if col != "segment" else "unknown"
        return adapted[required]

    def _adapt_clv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        adapted = df.copy()
        if "clv_predicted" not in adapted.columns and "predicted_clv" in adapted.columns:
            adapted["clv_predicted"] = adapted["predicted_clv"]
        if "clv_predicted" in adapted.columns:
            adapted["clv_predicted"] = pd.to_numeric(
                adapted["clv_predicted"], errors="coerce"
            )
            issues = []
            null_count = int(adapted["clv_predicted"].isna().sum())
            negative_count = int((adapted["clv_predicted"] < 0).sum())
            if null_count:
                issues.append(f"{null_count:,} null/non-numeric CLV rows")
            if negative_count:
                issues.append(f"{negative_count:,} negative CLV rows")
            if "customer_id" in adapted.columns:
                duplicate_count = int(
                    adapted["customer_id"].astype(str).duplicated().sum()
                )
                if duplicate_count:
                    issues.append(f"{duplicate_count:,} duplicate customer rows")
                customers_path = self._customers_path()
                if customers_path is not None:
                    try:
                        expected_ids = pd.read_csv(
                            customers_path, usecols=["customer_id"]
                        )["customer_id"].astype(str)
                        actual_ids = adapted["customer_id"].astype(str)
                        missing_count = int(len(set(expected_ids) - set(actual_ids)))
                        if missing_count:
                            issues.append(
                                f"{missing_count:,} customers missing from CLV artifact"
                            )
                    except Exception as exc:
                        issues.append(f"customer coverage could not be checked: {exc}")
            if issues:
                self._record_artifact_issue(
                    "clv_data",
                    "CLV artifact has invalid coverage: " + "; ".join(issues) + ".",
                )
            else:
                self._clear_artifact_issue("clv_data")
        if "segment" not in adapted.columns:
            adapted["segment"] = "all_customers"
        segment_path = self._resolve_existing_path("segments_6plus.csv")
        if segment_path is not None and "customer_id" in adapted.columns:
            segments = pd.read_csv(segment_path)
            if {"customer_id", "segment"}.issubset(segments.columns):
                adapted["customer_id"] = adapted["customer_id"].astype(str)
                segments["customer_id"] = segments["customer_id"].astype(str)
                adapted = adapted.drop(columns=["segment"], errors="ignore").merge(
                    segments[["customer_id", "segment"]],
                    on="customer_id",
                    how="left",
                )
                adapted["segment"] = adapted["segment"].fillna("all_customers")
        return adapted

    def _adapt_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add optional CLV fields expected by dashboard scatter views."""
        adapted = df.copy()
        adapted["churn_probability"] = pd.to_numeric(
            adapted["churn_probability"], errors="coerce"
        ).clip(0.0, 1.0)
        adapted = adapted.dropna(subset=["customer_id", "churn_probability"])
        if "risk_level" not in adapted.columns:
            adapted["risk_level"] = pd.cut(
                adapted["churn_probability"],
                bins=[-0.01, 0.25, 0.5, 0.75, 1.0],
                labels=["low", "medium", "high", "critical"],
            ).astype(str)
        if "segment" not in adapted.columns:
            adapted["segment"] = adapted.get("persona", "unknown")
        adapted["customer_id"] = adapted["customer_id"].astype(str)
        if "clv_predicted" not in adapted.columns:
            clv = self.load_clv_data()
            if not clv.empty and "customer_id" in adapted.columns:
                adapted = adapted.merge(
                    clv[["customer_id", "clv_predicted"]],
                    on="customer_id",
                    how="left",
                )
            elif self.get_artifact_issue("clv_data") is None:
                self._record_artifact_issue(
                    "clv_data",
                    "Required CLV artifact unavailable for churn prediction enrichment.",
                )
        if "clv_predicted" not in adapted.columns:
            adapted["clv_predicted"] = pd.NA
        adapted["clv_predicted"] = pd.to_numeric(
            adapted["clv_predicted"], errors="coerce"
        )
        adapted = self._enrich_predictions_with_sidecar_artifacts(adapted)
        return adapted

    def _enrich_predictions_with_sidecar_artifacts(
        self, adapted: pd.DataFrame
    ) -> pd.DataFrame:
        """Left-join recommendation/feature columns so dashboard widgets
        (e.g. the Overview Customer Lookup card) can render real
        ``recommended_action``, ``offer_type``, ``days_since_last_purchase``,
        and ``churn_label`` values instead of "N/A" placeholders.

        Each artifact is loaded defensively — a missing or unreadable file
        only skips that particular join and emits a warning via the module
        logger, leaving the rest of the prediction frame intact.
        """
        if "customer_id" not in adapted.columns:
            return adapted

        # Retention offers -> recommended_action, offer_type, expected_revenue_saved_krw, priority_score
        offers_path = self._resolve_existing_path("retention_offers.csv")
        if offers_path is not None:
            try:
                offers = pd.read_csv(offers_path)
                offer_cols = [
                    c for c in [
                        "customer_id",
                        "recommended_action",
                        "offer_type",
                        "expected_revenue_saved_krw",
                        "priority_score",
                    ]
                    if c in offers.columns
                ]
                if "customer_id" in offer_cols and len(offer_cols) > 1:
                    offers = offers[offer_cols].copy()
                    offers["customer_id"] = offers["customer_id"].astype(str)
                    # Avoid duplicate customer rows poisoning the left join.
                    offers = offers.drop_duplicates(subset=["customer_id"], keep="last")
                    adapted = adapted.merge(offers, on="customer_id", how="left")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "retention_offers enrichment skipped (%s): %s",
                    offers_path, exc,
                )
        else:
            logger.warning(
                "retention_offers enrichment skipped: retention_offers.csv not found "
                "in dashboard search paths."
            )

        # Features -> days_since_last_purchase, churn_label
        features_path = self._resolve_existing_path("features.csv")
        if features_path is not None:
            try:
                feat_cols = [
                    "customer_id",
                    "days_since_last_purchase",
                    "churn_label",
                ]
                features = pd.read_csv(features_path, usecols=lambda c: c in feat_cols)
                keep_cols = [c for c in feat_cols if c in features.columns]
                if "customer_id" in keep_cols and len(keep_cols) > 1:
                    features = features[keep_cols].copy()
                    features["customer_id"] = features["customer_id"].astype(str)
                    features = features.drop_duplicates(subset=["customer_id"], keep="last")
                    adapted = adapted.merge(features, on="customer_id", how="left")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "features enrichment skipped (%s): %s",
                    features_path, exc,
                )
        else:
            logger.warning(
                "features enrichment skipped: features.csv not found in dashboard "
                "search paths."
            )

        # Fill sensible defaults for downstream renderers.
        if "recommended_action" in adapted.columns:
            adapted["recommended_action"] = (
                adapted["recommended_action"].fillna("no_action")
            )
        if "offer_type" in adapted.columns:
            adapted["offer_type"] = adapted["offer_type"].fillna("")
        for numeric_col in [
            "expected_revenue_saved_krw",
            "priority_score",
            "days_since_last_purchase",
            "churn_label",
        ]:
            if numeric_col in adapted.columns:
                adapted[numeric_col] = pd.to_numeric(
                    adapted[numeric_col], errors="coerce"
                ).fillna(0)

        return adapted

    def get_prediction_coverage(self) -> Dict[str, Any]:
        """Compare churn prediction rows with the current customer universe."""
        issue = self.get_artifact_issue("churn_predictions")
        predictions_path = self._resolve_existing_path("churn_predictions.csv")
        customers_path = self._customers_path()
        if issue is not None:
            return {
                "status": "invalid",
                "message": issue,
                "prediction_count": 0,
                "customer_count": 0,
                "covered_count": 0,
                "missing_count": None,
                "coverage_ratio": 0.0,
                "is_full_coverage": False,
            }
        if predictions_path is None or customers_path is None:
            return {
                "status": "unknown",
                "message": "Customer universe or churn prediction artifact is unavailable.",
                "prediction_count": 0,
                "customer_count": 0,
                "covered_count": 0,
                "missing_count": None,
                "coverage_ratio": 0.0,
                "is_full_coverage": False,
            }

        try:
            predictions = pd.read_csv(predictions_path, usecols=["customer_id"])
            customers = pd.read_csv(customers_path, usecols=["customer_id"])
        except Exception as exc:
            return {
                "status": "invalid",
                "message": f"Could not evaluate churn prediction coverage: {exc}.",
                "prediction_count": 0,
                "customer_count": 0,
                "covered_count": 0,
                "missing_count": None,
                "coverage_ratio": 0.0,
                "is_full_coverage": False,
            }

        prediction_ids = set(predictions["customer_id"].astype(str))
        customer_ids = set(customers["customer_id"].astype(str))
        covered = len(prediction_ids & customer_ids)
        total = len(customer_ids)
        missing = max(total - covered, 0)
        ratio = covered / total if total else 0.0
        full = total > 0 and missing == 0
        message = (
            f"Churn predictions cover all {total:,} customers."
            if full
            else (
                f"Churn predictions cover {covered:,}/{total:,} customers; "
                f"{missing:,} customers are missing."
            )
        )
        return {
            "status": "complete" if full else "partial",
            "message": message,
            "prediction_count": len(prediction_ids),
            "customer_count": total,
            "covered_count": covered,
            "missing_count": missing,
            "coverage_ratio": ratio,
            "is_full_coverage": full,
        }

    def _adapt_uplift_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map uplift pipeline output to dashboard schema."""
        adapted = df.copy()
        if "treatment_effect" not in adapted.columns:
            adapted["treatment_effect"] = adapted.get("uplift_score", 0.0)
        if "segment" not in adapted.columns:
            adapted["segment"] = "unknown"
        return adapted

    def _load_survival_from_segments(self) -> Optional[pd.DataFrame]:
        """Build survival view rows from 6+ segmentation output."""
        segment_path = self._resolve_existing_path("segments_6plus.csv")
        if segment_path is None:
            return None
        segments = pd.read_csv(segment_path)
        required = {"customer_id", "segment"}
        if not required.issubset(segments.columns):
            return None
        churn = segments.get(
            "churn_probability",
            pd.Series(0.2, index=segments.index),
        ).astype(float).clip(0.0, 1.0)
        duration = (
            365 * (1.0 - churn)
        ).clip(lower=1).round().astype(int)
        return pd.DataFrame({
            "customer_id": segments["customer_id"],
            "duration_days": duration,
            "event_observed": (churn >= 0.5).astype(int),
            "segment": segments["segment"],
            "survival_probability": (1.0 - churn).clip(0.0, 1.0),
        })

    def _adapt_model_metrics(self, payload: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Keep only dashboard-ready model metric mappings."""
        aliases = {
            "ml_metrics": "ml_model",
            "ml_model": "ml_model",
            "dl_metrics": "dl_model",
            "dl_model": "dl_model",
            "ensemble_metrics": "ensemble",
            "ensemble": "ensemble",
        }
        adapted: Dict[str, Dict[str, float]] = {}
        for raw_key, canonical_key in aliases.items():
            metrics = payload.get(raw_key)
            if not isinstance(metrics, dict):
                continue
            auc = metrics.get("auc", metrics.get("auc_roc"))
            if auc is None:
                continue
            adapted[canonical_key] = {
                "auc": float(auc),
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1_score": float(metrics.get("f1_score", metrics.get("f1", 0.0))),
                "accuracy": float(metrics.get("accuracy", 0.0)),
            }
        return adapted

    def _adapt_recommendations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map recommendation pipeline outputs to dashboard schema."""
        adapted = df.copy()
        if "recommendation_type" not in adapted.columns and "action" in adapted.columns:
            adapted["recommendation_type"] = adapted["action"]
        if "recommendation_type" not in adapted.columns and "action_type" in adapted.columns:
            adapted["recommendation_type"] = adapted["action_type"]
        if "priority_score" not in adapted.columns:
            if "score" in adapted.columns:
                adapted["priority_score"] = adapted["score"]
            else:
                adapted["priority_score"] = 0.0
        if "expected_uplift" not in adapted.columns:
            adapted["expected_uplift"] = adapted.get("uplift_score", adapted["priority_score"])
        if "recommended_offer" not in adapted.columns:
            adapted["recommended_offer"] = adapted["recommendation_type"].astype(str)
        if "estimated_cost" not in adapted.columns:
            default_costs = {
                "email": 1000,
                "push_notification": 500,
                "coupon": 5000,
                "loyalty_points": 3000,
                "personal_outreach": 20000,
                "exclusive_offer": 10000,
                "no_action": 0,
            }
            adapted["estimated_cost"] = (
                adapted["recommendation_type"].astype(str).map(default_costs).fillna(1000)
            )
        if "segment" not in adapted.columns:
            segment_path = self._resolve_existing_path("segments_6plus.csv")
            if segment_path is not None and "customer_id" in adapted.columns:
                segments = pd.read_csv(segment_path)
                if {"customer_id", "segment"}.issubset(segments.columns):
                    adapted["customer_id"] = adapted["customer_id"].astype(str)
                    segments["customer_id"] = segments["customer_id"].astype(str)
                    adapted = adapted.merge(
                        segments[["customer_id", "segment"]],
                        on="customer_id",
                        how="left",
                    )
            if "segment" not in adapted.columns:
                adapted["segment"] = "vip_loyal"
            else:
                adapted["segment"] = adapted["segment"].fillna("vip_loyal")
        adapted["priority_score"] = adapted["priority_score"].clip(0.0, 1.0)
        adapted["expected_uplift"] = adapted["expected_uplift"].clip(0.0, 1.0)
        return adapted

    def _adapt_ab_results(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        adapted = dict(payload)
        if "lift" not in adapted:
            treatment_rate = adapted.get(
                "treatment_churn_rate",
                adapted.get("treatment_mean", 0.0),
            )
            control_rate = adapted.get("control_churn_rate", adapted.get("control_mean", 0.0))
            adapted["lift"] = (
                (control_rate - treatment_rate) / abs(control_rate)
                if control_rate not in (0, None)
                else 0.0
            )
        return adapted

    def _adapt_ab_detailed(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from src.models.ab_testing import ABTestFramework

        framework = ABTestFramework(self.config)
        return framework.to_dashboard_detailed_results(payload)

    def load_predictions(self) -> pd.DataFrame:
        """Load churn prediction results.

        Returns:
            DataFrame with customer_id, churn_probability, risk_level,
            segment, recommended_action, clv_predicted, etc.
        """
        artifact_name = "churn_predictions"
        df = self._required_csv(
            artifact_name,
            ["churn_predictions.csv"],
            ["customer_id", "churn_probability"],
        )
        if df is None:
            return self._empty_frame(
                [
                    "customer_id", "churn_probability", "risk_level",
                    "segment", "clv_predicted",
                ],
                artifact_name,
                self.get_artifact_issue(artifact_name) or "Churn predictions unavailable.",
            )
        adapted = self._adapt_predictions(df)
        if adapted.empty:
            issue = "Required churn prediction artifact has no valid prediction rows."
            self._record_artifact_issue(artifact_name, issue)
            return self._empty_frame(
                [
                    "customer_id", "churn_probability", "risk_level",
                    "segment", "clv_predicted",
                ],
                artifact_name,
                issue,
            )
        return adapted

    def load_model_metrics(self) -> Dict[str, Dict[str, float]]:
        """Load model performance metrics.

        Returns:
            Dict with ml_model, dl_model, ensemble keys, each containing
            auc, precision, recall, f1_score, accuracy.
        """
        payload = self._required_json(
            "model_metrics",
            ["model_metrics.json"],
            [],
        )
        if payload is not None:
            adapted = self._adapt_model_metrics(payload)
            if adapted:
                return adapted
            self._record_artifact_issue(
                "model_metrics",
                "Required model_metrics.json is invalid; no dashboard-ready model metrics found.",
            )
        return {}

    def load_ab_test_results(self) -> Dict[str, Any]:
        """Load A/B test experiment results.

        Returns:
            Dict with experiment_name, treatment/control sizes,
            churn rates, lift, p_value, is_significant, CI.
        """
        payload = self._required_json(
            "ab_test_results",
            ["ab_test_results.json"],
            ["p_value"],
        )
        if payload is not None:
            return self._adapt_ab_results(payload)
        return {}

    def load_uniform_treatment_clv(self) -> dict:
        """Load uniform-treatment CLV summary.

        Returns:
            Dict with baseline_clv, uniform_treatment_clv, delta_clv,
            n_customers, avg_uplift_score, method, generated_at.
            Empty dict if the artifact is missing.
        """
        path = self.results_dir / "uniform_treatment_clv.json"
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("load_uniform_treatment_clv failed: %s", e)
            return {}

    def load_budget_results(self) -> pd.DataFrame:
        """Load budget optimization results by segment.

        Returns:
            DataFrame with segment, allocated_budget_krw,
            expected_retained, expected_revenue_saved_krw, roi.
        """
        artifact_name = "budget_results"
        df = self._required_csv(
            artifact_name,
            ["budget_results.csv", "budget_optimization.csv"],
            [],
        )
        if df is not None:
            adapted = self._adapt_budget_results(df)
            if not adapted.empty:
                return adapted
        return self._empty_frame(
            [
                "segment", "allocated_budget_krw", "expected_retained",
                "expected_revenue_saved_krw", "roi",
            ],
            artifact_name,
            self.get_artifact_issue(artifact_name) or "Budget optimization artifact unavailable.",
        )

    def load_survival_data(
        self, as_artifact: bool = False
    ) -> Union[pd.DataFrame, DashboardArtifact]:
        """Load survival analysis data.

        Real survival rows are loaded from ``survival_data.csv`` when the
        pipeline produces it. As a transitional measure we still derive a
        per-customer survival view from ``segments_6plus.csv`` (which the
        pipeline already emits); the synthetic ``_generate_sample_*``
        fallback has been removed.

        Returns:
            DataFrame with customer_id, duration_days, event_observed,
            segment, survival_probability (or DashboardArtifact wrapper
            when ``as_artifact=True``).
        """
        artifact_name = "survival_data"
        empty_columns = [
            "customer_id", "duration_days", "event_observed",
            "segment", "survival_probability",
        ]

        path = self._resolve_existing_path("survival_data.csv")
        if path is not None:
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                issue = (
                    f"Required survival artifact {path} could not be "
                    f"read: {exc}."
                )
                self._mark_missing(artifact_name, issue)
                empty = self._empty_frame(empty_columns, artifact_name, issue)
                if as_artifact:
                    return DashboardArtifact(
                        data=empty, is_real=False, source_path=str(path),
                        reason=issue, name=artifact_name,
                    )
                return empty
            self._mark_real(artifact_name)
            if as_artifact:
                return DashboardArtifact(
                    data=df, is_real=True, source_path=str(path),
                    name=artifact_name,
                )
            return df

        segmented = self._load_survival_from_segments()
        if segmented is not None:
            # Segments-derived survival is real upstream data even though it
            # is re-encoded from churn probabilities; surface that distinction
            # via a dashboard-visible issue while keeping is_real=False so
            # views can downgrade KPI cards.
            issue = (
                "survival_data.csv missing — derived from segments_6plus.csv. "
                "Run `python -m src.main --mode all` to produce a real "
                "survival_data.csv artifact."
            )
            self._mark_missing(artifact_name, issue)
            if as_artifact:
                return DashboardArtifact(
                    data=segmented, is_real=False,
                    source_path="segments_6plus.csv",
                    reason=issue, name=artifact_name,
                )
            return segmented

        issue = (
            "Required artifact missing: survival_data.csv. Dashboard will "
            "NOT render synthetic data. Run `python -m src.main --mode all` "
            "to produce a real survival_data.csv."
        )
        self._mark_missing(artifact_name, issue)
        empty = self._empty_frame(empty_columns, artifact_name, issue)
        if as_artifact:
            return DashboardArtifact(
                data=empty, is_real=False, source_path=None,
                reason=issue, name=artifact_name,
            )
        return empty

    def load_recommendations(self) -> pd.DataFrame:
        """Load personalized recommendation results.

        Returns:
            DataFrame with customer_id, recommendation_type,
            expected_uplift, priority_score, recommended_offer.
        """
        artifact_name = "recommendations"
        df = self._required_csv(
            artifact_name,
            ["recommendations.csv"],
            ["customer_id"],
        )
        if df is not None:
            return self._adapt_recommendations(df)
        return self._empty_frame(
            [
                "customer_id", "recommendation_type", "expected_uplift",
                "priority_score", "recommended_offer",
            ],
            artifact_name,
            self.get_artifact_issue(artifact_name) or "Recommendations artifact unavailable.",
        )

    def load_uplift_results(self) -> pd.DataFrame:
        """Load uplift modeling results.

        Returns:
            DataFrame with customer_id, uplift_score, treatment_effect,
            segment.
        """
        artifact_name = "uplift_results"
        df = self._required_csv(
            artifact_name,
            ["uplift_results.csv"],
            ["customer_id", "uplift_score"],
        )
        if df is not None:
            return self._adapt_uplift_results(df)
        return self._empty_frame(
            ["customer_id", "uplift_score", "treatment_effect", "segment"],
            artifact_name,
            self.get_artifact_issue(artifact_name) or "Uplift results artifact unavailable.",
        )

    def load_clv_data(self) -> pd.DataFrame:
        """Load CLV prediction data.

        Returns:
            DataFrame with customer_id, clv_predicted, segment.
        """
        artifact_name = "clv_data"
        df = self._required_csv(
            artifact_name,
            ["clv_data.csv", "clv_predictions.csv"],
            ["customer_id"],
        )
        if df is not None:
            adapted = self._adapt_clv_data(df)
            if "clv_predicted" in adapted.columns and not adapted.empty:
                return adapted
            self._record_artifact_issue(
                artifact_name,
                "Required CLV artifact is invalid; missing non-negative "
                "clv_predicted or predicted_clv values.",
            )
        return self._empty_frame(
            ["customer_id", "clv_predicted", "segment"],
            artifact_name,
            self.get_artifact_issue(artifact_name) or "CLV artifact unavailable.",
        )

    def load_feature_importance(self) -> pd.DataFrame:
        """Load feature importance scores from trained model.

        Returns:
            DataFrame with feature, importance columns sorted descending.
        """
        artifact_name = "feature_importance"
        df = self._required_csv(
            artifact_name,
            ["feature_importance.csv"],
            ["feature", "importance"],
        )
        if df is not None:
            return df.sort_values("importance", ascending=False).reset_index(
                drop=True
            )
        return self._empty_frame(
            ["feature", "importance"],
            artifact_name,
            self.get_artifact_issue(artifact_name) or "Feature importance artifact unavailable.",
        )

    # -----------------------------------------------------------------
    # Deprecated sample data generators (iter13)
    #
    # All ``_generate_sample_*`` fallbacks have been removed. Dashboard
    # views must rely on real pipeline artifacts. The stubs below remain
    # so external callers referencing the method name receive a clear
    # FileNotFoundError instead of silently rendering synthetic data.
    # -----------------------------------------------------------------

    @staticmethod
    def _missing_artifact_error(artifact: str) -> FileNotFoundError:
        return FileNotFoundError(
            f"{artifact} missing — run `python -m src.main --mode all` to "
            "produce real artifact. Dashboard will NOT render synthetic data."
        )

    def _generate_sample_predictions(self) -> pd.DataFrame:
        raise self._missing_artifact_error("churn_predictions.csv")

    def _generate_sample_metrics(self) -> Dict[str, Dict[str, float]]:
        raise self._missing_artifact_error("model_metrics.json")

    def _generate_sample_ab_results(self) -> Dict[str, Any]:
        raise self._missing_artifact_error("ab_test_results.json")

    def _generate_sample_budget_results(self) -> pd.DataFrame:
        raise self._missing_artifact_error("budget_results.csv")

    def _generate_sample_survival_data(self) -> pd.DataFrame:
        raise self._missing_artifact_error("survival_data.csv")

    def _generate_sample_recommendations(self) -> pd.DataFrame:
        raise self._missing_artifact_error("recommendations.csv")

    def _generate_sample_uplift_results(self) -> pd.DataFrame:
        raise self._missing_artifact_error("uplift_results.csv")

    def _generate_sample_clv_data(self) -> pd.DataFrame:
        raise self._missing_artifact_error("clv_data.csv")

    def load_cohort_data(self) -> pd.DataFrame:
        """Load cohort analysis event data.

        Returns:
            DataFrame with customer_id, event_date, revenue, segment
            suitable for CohortAnalyzer.assign_cohorts().
        """
        artifact_name = "cohort_data"
        empty_columns = ["customer_id", "event_date", "revenue", "segment"]
        path = self._resolve_existing_path("cohort_data.csv")
        if path is not None:
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                issue = f"Required cohort artifact {path} could not be read: {exc}."
                self._record_artifact_issue(artifact_name, issue)
                return self._empty_frame(empty_columns, artifact_name, issue)

            required = {"customer_id", "event_date"}
            if not required.issubset(df.columns):
                missing = ", ".join(sorted(required - set(df.columns)))
                issue = (
                    f"Required cohort artifact {path} is invalid; "
                    f"missing columns: {missing}."
                )
                self._record_artifact_issue(artifact_name, issue)
                return self._empty_frame(empty_columns, artifact_name, issue)

            if "revenue" not in df.columns:
                df["revenue"] = df.get("amount", 0.0)
            if "segment" not in df.columns:
                df["segment"] = "all_customers"
            df["event_date"] = pd.to_datetime(df["event_date"])
            self._clear_artifact_issue(artifact_name)
            return df

        raw_events = self._raw_events_path()
        if raw_events is not None:
            try:
                if raw_events.suffix == ".parquet":
                    df = pd.read_parquet(
                        raw_events,
                        columns=["customer_id", "event_date", "amount"],
                    )
                    df = df[pd.to_numeric(df["amount"], errors="coerce") > 0]
                else:
                    chunks = []
                    for chunk in pd.read_csv(
                        raw_events,
                        usecols=["customer_id", "event_date", "amount"],
                        chunksize=100_000,
                    ):
                        purchases = chunk[
                            pd.to_numeric(chunk["amount"], errors="coerce") > 0
                        ]
                        if not purchases.empty:
                            chunks.append(purchases)
                    df = (
                        pd.concat(chunks, ignore_index=True)
                        if chunks
                        else pd.DataFrame()
                    )
            except Exception as exc:
                issue = (
                    "Required raw simulator events could not be read "
                    f"for cohort data: {exc}."
                )
                self._record_artifact_issue(artifact_name, issue)
                return self._empty_frame(empty_columns, artifact_name, issue)

            if df.empty:
                issue = "Required raw simulator events contain no purchase rows for cohort data."
                self._record_artifact_issue(artifact_name, issue)
                return self._empty_frame(empty_columns, artifact_name, issue)

            df = df.rename(columns={"amount": "revenue"})
            df["event_date"] = pd.to_datetime(df["event_date"])
            df["segment"] = "all_customers"
            self._clear_artifact_issue(artifact_name)
            return df[empty_columns]

        issue = (
            "Required artifact missing: cohort_data.csv or data/raw/events. "
            "Cohort analysis cannot use generated samples for required dashboard evidence."
        )
        self._record_artifact_issue(artifact_name, issue)
        return self._empty_frame(empty_columns, artifact_name, issue)

    def load_cohort_sizes(self) -> pd.DataFrame:
        """Load acquisition cohort sizes using the same signup basis as retention."""
        empty_columns = ["Cohort", "Customers"]
        path = self._customers_path()
        if path is None:
            return pd.DataFrame(columns=empty_columns)

        try:
            df = pd.read_csv(path, usecols=["customer_id", "signup_date"])
        except (ValueError, FileNotFoundError):
            return pd.DataFrame(columns=empty_columns)
        except Exception as exc:
            self._record_artifact_issue(
                "cohort_sizes",
                f"Customer cohort sizes could not be read from {path}: {exc}.",
            )
            return pd.DataFrame(columns=empty_columns)

        signup = pd.to_datetime(df["signup_date"], errors="coerce")
        valid = df.loc[signup.notna(), ["customer_id"]].copy()
        valid["Cohort"] = signup[signup.notna()].dt.to_period("M").astype(str).values
        if valid.empty:
            return pd.DataFrame(columns=empty_columns)

        sizes = (
            valid.groupby("Cohort")["customer_id"]
            .nunique()
            .sort_index()
            .reset_index(name="Customers")
        )
        self._clear_artifact_issue("cohort_sizes")
        return sizes

    def load_cohort_retention_matrix(self) -> pd.DataFrame:
        """Load pre-computed cohort retention matrix.

        Returns:
            DataFrame with cohort labels as index, period indices as columns,
            and retention rates (0.0 to 1.0) as values.
        """
        path = self._resolve_existing_path("cohort_retention_matrix.csv")
        if path is not None:
            df = pd.read_csv(path, index_col=0)
            df.columns = [
                int(col) if str(col).isdigit() else col
                for col in df.columns
            ]
            if df.shape[0] >= 2 and df.shape[1] >= 2:
                self._clear_artifact_issue("cohort_retention_matrix")
                return df
            issue = f"Required cohort retention matrix {path} is incomplete."
            self._record_artifact_issue("cohort_retention_matrix", issue)
            return self._empty_frame([], "cohort_retention_matrix", issue)

        # Compute from cohort data using CohortAnalyzer
        cohort_data = self.load_cohort_data()
        if cohort_data.empty:
            issue = (
                self.get_artifact_issue("cohort_data")
                or "Cohort retention matrix unavailable."
            )
            self._record_artifact_issue("cohort_retention_matrix", issue)
            return self._empty_frame([], "cohort_retention_matrix", issue)

        try:
            from src.analysis.cohort_analysis import CohortAnalyzer
            analyzer = CohortAnalyzer()
            assigned = analyzer.assign_cohorts(cohort_data, cohort_type="monthly")
            retention = analyzer.compute_retention_matrix(assigned)
            if retention.shape[0] >= 2 and retention.shape[1] >= 2:
                self._clear_artifact_issue("cohort_retention_matrix")
                return retention
            issue = "Computed cohort retention matrix is incomplete."
        except Exception as exc:
            issue = f"Cohort retention matrix could not be computed: {exc}."
        self._record_artifact_issue("cohort_retention_matrix", issue)
        return self._empty_frame([], "cohort_retention_matrix", issue)

    def _generate_sample_cohort_data(self) -> pd.DataFrame:
        raise self._missing_artifact_error("cohort_data.csv")

    def _generate_sample_retention_matrix(self) -> pd.DataFrame:
        raise self._missing_artifact_error("cohort_retention_matrix.csv")

    # -----------------------------------------------------------------
    # Model Performance & A/B Testing enhanced loaders
    # -----------------------------------------------------------------

    def load_model_performance_history(self) -> pd.DataFrame:
        """Load real model performance history in dashboard run-history shape."""
        artifact_name = "model_performance_history"
        df = self._required_csv(
            artifact_name,
            ["model_performance_history.csv", "mlflow_runs.csv"],
            [],
        )
        if df is None:
            return self._empty_frame(
                [
                    "run_id", "model_type", "auc", "precision", "recall",
                    "f1_score", "accuracy", "training_time_s", "timestamp",
                    "params_lr", "params_epochs",
                ],
                artifact_name,
                self.get_artifact_issue(artifact_name)
                or "Model performance history unavailable.",
            )
        if "model_type" not in df.columns and "model" in df.columns:
            df = df.rename(columns={"model": "model_type"})
        if "auc" not in df.columns and "auc_roc" in df.columns:
            df["auc"] = df["auc_roc"]
        if "f1_score" not in df.columns and "f1" in df.columns:
            df["f1_score"] = df["f1"]
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.Timestamp.utcnow().isoformat()
        if "run_id" not in df.columns:
            run_col = (
                df["run"].astype(str)
                if "run" in df.columns
                else pd.Series("current", index=df.index)
            )
            df["run_id"] = [f"{run}_{idx}" for idx, run in enumerate(run_col)]
        for metric in ["auc", "precision", "recall", "f1_score", "accuracy"]:
            if metric not in df.columns:
                df[metric] = 0.0
            df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)
        if "model_type" not in df.columns:
            df["model_type"] = "unknown"
        if "training_time_s" not in df.columns:
            df["training_time_s"] = 1.0
        else:
            df["training_time_s"] = pd.to_numeric(
                df["training_time_s"], errors="coerce"
            ).fillna(1.0).clip(lower=1e-6)
        if "params_lr" not in df.columns:
            df["params_lr"] = 0.1
        if "params_epochs" not in df.columns:
            df["params_epochs"] = 1
        if df.empty:
            self._record_artifact_issue(
                artifact_name,
                "Required model performance history artifact has no rows.",
            )
        else:
            self._clear_artifact_issue(artifact_name)
        return df

    def _load_live_mlflow_runs(self, max_results: int = 500) -> pd.DataFrame:
        """Query the configured MLflow tracking backend for run history."""
        mlflow_cfg = self.config.get("mlflow", {}) or {}
        tracking_uri = mlflow_cfg.get("tracking_uri")
        experiment_name = mlflow_cfg.get("experiment_name", "churn_prediction")
        if not tracking_uri:
            return pd.DataFrame()

        try:
            import mlflow  # type: ignore

            mlflow.set_tracking_uri(tracking_uri)
            client = mlflow.tracking.MlflowClient()
            experiment_ids: List[str] = []
            experiment_names: Dict[str, str] = {}

            exp = client.get_experiment_by_name(experiment_name)
            if exp is not None:
                experiment_ids.append(exp.experiment_id)
                experiment_names[exp.experiment_id] = exp.name
            else:
                for candidate in client.search_experiments():
                    experiment_ids.append(candidate.experiment_id)
                    experiment_names[candidate.experiment_id] = candidate.name

            if not experiment_ids:
                return pd.DataFrame()

            try:
                runs = client.search_runs(
                    experiment_ids=experiment_ids,
                    max_results=max_results,
                    order_by=["attributes.start_time DESC"],
                )
            except Exception:
                runs = client.search_runs(
                    experiment_ids=experiment_ids,
                    max_results=max_results,
                )
        except Exception as exc:
            self._record_artifact_issue(
                "mlflow_runs",
                f"Live MLflow query unavailable; using cached artifact if present: {exc}.",
            )
            return pd.DataFrame()

        records = []
        for run in runs:
            metrics = dict(run.data.metrics or {})
            params = dict(run.data.params or {})
            tags = dict(run.data.tags or {})
            start_ms = run.info.start_time
            end_ms = run.info.end_time
            training_time_s = metrics.get("training_time_s")
            if training_time_s is None and start_ms and end_ms:
                training_time_s = max((end_ms - start_ms) / 1000.0, 0.0)
            records.append({
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "experiment_name": experiment_names.get(
                    run.info.experiment_id, experiment_name,
                ),
                "model_type": (
                    params.get("model_type")
                    or tags.get("model_type")
                    or tags.get("pipeline_stage")
                    or tags.get("mlflow.runName")
                    or "unknown"
                ),
                "auc": metrics.get("auc", metrics.get("auc_roc", np.nan)),
                "precision": metrics.get("precision", np.nan),
                "recall": metrics.get("recall", np.nan),
                "f1_score": metrics.get(
                    "f1_score", metrics.get("f1", np.nan),
                ),
                "accuracy": metrics.get("accuracy", np.nan),
                "training_time_s": training_time_s
                                   if training_time_s is not None
                                   else np.nan,
                "params_lr": params.get("lr", params.get("learning_rate", np.nan)),
                "params_epochs": params.get("epochs", np.nan),
                "timestamp": pd.to_datetime(
                    start_ms, unit="ms", errors="coerce",
                ),
                "source": "live_mlflow",
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        for metric in ["auc", "precision", "recall", "f1_score", "accuracy"]:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df[df["auc"].notna()].copy()
        if df.empty:
            return pd.DataFrame()
        current_run_ids = self._current_mlflow_run_ids()
        if current_run_ids:
            current_df = df[df["run_id"].isin(current_run_ids)].copy()
            if not current_df.empty:
                df = current_df
        df["training_time_s"] = pd.to_numeric(
            df["training_time_s"], errors="coerce",
        ).fillna(1.0).clip(lower=0.0)
        df["params_lr"] = pd.to_numeric(df["params_lr"], errors="coerce")
        df["params_epochs"] = (
            pd.to_numeric(df["params_epochs"], errors="coerce")
            .fillna(1.0)
            .clip(lower=1e-6)
        )
        self._mark_real("mlflow_runs")
        return df

    def _current_mlflow_run_ids(self) -> set:
        """Return MLflow run IDs recorded by the latest model_metrics.json."""
        metrics_path = self._resolve_existing_path("model_metrics.json")
        if metrics_path is None:
            return set()
        try:
            payload = self._read_json(metrics_path)
        except Exception:
            return set()
        run_info = payload.get("mlflow_runs", {}) if isinstance(payload, dict) else {}
        run_ids = run_info.get("run_ids", {}) if isinstance(run_info, dict) else {}
        if not isinstance(run_ids, dict):
            return set()
        return {str(value) for value in run_ids.values() if value}

    def load_mlflow_runs(
        self, as_artifact: bool = False,
    ) -> Union[pd.DataFrame, DashboardArtifact]:
        """Load MLflow experiment run metrics.

        Returns:
            DataFrame with run_id, model_type, auc, precision, recall,
            f1_score, accuracy, training_time_s, timestamp.
        """
        live = self._load_live_mlflow_runs()
        if not live.empty:
            if as_artifact:
                return DashboardArtifact(
                    data=live,
                    is_real=True,
                    source_path=str(
                        self.config.get("mlflow", {}).get("tracking_uri", "")
                    ),
                    name="mlflow_runs",
                    extra={"source": "live_mlflow"},
                )
            return live

        cached = self.load_model_performance_history()
        reason = (
            self.get_artifact_issue("mlflow_runs")
            or "Live MLflow run query returned no rows; using cached "
               "model_performance_history.csv."
        )
        if as_artifact:
            return DashboardArtifact(
                data=cached,
                is_real=False,
                source_path=str(
                    self._resolve_existing_path(
                        "model_performance_history.csv", "mlflow_runs.csv",
                    ) or ""
                ),
                reason=reason,
                name="mlflow_runs",
                extra={"source": "cached_model_performance_history"},
            )
        return cached

    def get_active_model(self) -> Dict[str, str]:
        """Return the best available model stamp for dashboard captions."""
        history = self.load_scoring_history()
        if not history.empty:
            for column in ("model_version", "model_type"):
                if column in history.columns:
                    value = history[column].dropna().astype(str)
                    value = value[value.str.len() > 0]
                    if not value.empty:
                        version = value.iloc[-1]
                        name = "ensemble"
                        if "model_type" in history.columns:
                            model_values = (
                                history["model_type"].dropna().astype(str)
                            )
                            if not model_values.empty:
                                name = model_values.iloc[-1]
                        if name == version and "_v" in name:
                            name = name.split("_v", 1)[0]
                        return {"name": name, "version": version}

        metrics = self.load_model_metrics()
        if metrics:
            candidates = [
                (name, vals.get("auc", float("-inf")))
                for name, vals in metrics.items()
                if isinstance(vals, dict)
            ]
            if candidates:
                best_name, _ = max(candidates, key=lambda item: item[1])
                return {"name": best_name, "version": "current"}

        return {}

    def load_roc_data(
        self, as_artifact: bool = False
    ) -> Union[Dict[str, Dict[str, list]], DashboardArtifact]:
        """Load ROC curve data for each model.

        Returns:
            Dict mapping model name to dict with fpr, tpr lists. Empty
            dict when ``roc_data.json`` is missing. The synthetic
            Beta-distributed fallback has been removed.
        """
        artifact_name = "roc_data"
        path = self._resolve_existing_path("roc_data.json")
        if path is not None:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as exc:
                issue = f"Required artifact {path} could not be read: {exc}."
                self._mark_missing(artifact_name, issue)
                if as_artifact:
                    return DashboardArtifact(
                        data={}, is_real=False, source_path=str(path),
                        reason=issue, name=artifact_name,
                    )
                return {}
            if isinstance(payload, dict) and payload:
                self._mark_real(artifact_name)
                if as_artifact:
                    return DashboardArtifact(
                        data=payload, is_real=True,
                        source_path=str(path), name=artifact_name,
                    )
                return payload
            issue = f"Required artifact {path} is empty or invalid."
            self._mark_missing(artifact_name, issue)
            if as_artifact:
                return DashboardArtifact(
                    data={}, is_real=False, source_path=str(path),
                    reason=issue, name=artifact_name,
                )
            return {}

        issue = (
            "Required artifact missing: roc_data.json. Dashboard will "
            "NOT render synthetic data. Run `python -m src.main --mode all` "
            "to produce real ROC curves."
        )
        self._mark_missing(artifact_name, issue)
        if as_artifact:
            return DashboardArtifact(
                data={}, is_real=False, source_path=None,
                reason=issue, name=artifact_name,
            )
        return {}

    def load_confusion_matrices(
        self, as_artifact: bool = False
    ) -> Union[Dict[str, list], DashboardArtifact]:
        """Load confusion matrix data for each model.

        Returns:
            Dict mapping model name to 2x2 matrix as nested list. Empty
            dict when ``confusion_matrices.json`` is missing. The
            hardcoded ``[[350,50],[80,120]]`` fixture has been removed.
        """
        artifact_name = "confusion_matrices"
        path = self._resolve_existing_path("confusion_matrices.json")
        if path is not None:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as exc:
                issue = f"Required artifact {path} could not be read: {exc}."
                self._mark_missing(artifact_name, issue)
                if as_artifact:
                    return DashboardArtifact(
                        data={}, is_real=False, source_path=str(path),
                        reason=issue, name=artifact_name,
                    )
                return {}
            if isinstance(payload, dict) and payload:
                self._mark_real(artifact_name)
                if as_artifact:
                    return DashboardArtifact(
                        data=payload, is_real=True,
                        source_path=str(path), name=artifact_name,
                    )
                return payload
            issue = f"Required artifact {path} is empty or invalid."
            self._mark_missing(artifact_name, issue)
            if as_artifact:
                return DashboardArtifact(
                    data={}, is_real=False, source_path=str(path),
                    reason=issue, name=artifact_name,
                )
            return {}

        issue = (
            "Required artifact missing: confusion_matrices.json. "
            "Dashboard will NOT render synthetic data. Run "
            "`python -m src.main --mode all` to produce real "
            "confusion matrices."
        )
        self._mark_missing(artifact_name, issue)
        if as_artifact:
            return DashboardArtifact(
                data={}, is_real=False, source_path=None,
                reason=issue, name=artifact_name,
            )
        return {}

    def load_ab_test_detailed(self) -> Dict[str, Any]:
        """Load detailed A/B test results including multiple experiments.

        Returns:
            Dict with experiments list, each containing name, metrics,
            and statistical test details.
        """
        artifact_name = "ab_test_detailed"
        path = self._resolve_existing_path("ab_test_detailed.json")
        if path is not None:
            try:
                detail = self._adapt_ab_detailed(self._read_json(path))
            except Exception as exc:
                self._record_artifact_issue(
                    artifact_name,
                    f"Required A/B detail artifact {path} could not be read: {exc}.",
                )
                return {}
            if detail.get("experiments"):
                self._clear_artifact_issue(artifact_name)
                return detail
            self._record_artifact_issue(
                artifact_name,
                f"Required A/B detail artifact {path} has no experiments.",
            )
            return {}
        path = self._resolve_existing_path("ab_test_results.json")
        if path is not None:
            try:
                detail = self._adapt_ab_detailed(self._read_json(path))
            except Exception as exc:
                self._record_artifact_issue(
                    artifact_name,
                    f"Required A/B result artifact {path} could not be read: {exc}.",
                )
                return {}
            if detail.get("experiments"):
                self._clear_artifact_issue(artifact_name)
                return detail
            self._record_artifact_issue(
                artifact_name,
                f"Required A/B result artifact {path} has no experiments.",
            )
            return {}
        self._record_artifact_issue(
            artifact_name,
            "Required artifact missing: ab_test_detailed.json or ab_test_results.json. "
            "A/B dashboard cannot use generated samples for required evidence.",
        )
        return {}

    def load_survival_curves(
        self, as_artifact: bool = False
    ) -> Union[Dict[str, Dict[str, list]], DashboardArtifact]:
        """Load Kaplan-Meier survival curves per segment.

        Returns:
            Dict mapping segment name to dict with timeline, survival_prob.
            Empty dict when ``survival_curves.json`` is missing. The
            synthetic Kaplan-Meier fallback has been removed.
        """
        artifact_name = "survival_curves"
        path = self._resolve_existing_path("survival_curves.json")
        if path is not None:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as exc:
                issue = f"Required artifact {path} could not be read: {exc}."
                self._mark_missing(artifact_name, issue)
                if as_artifact:
                    return DashboardArtifact(
                        data={}, is_real=False, source_path=str(path),
                        reason=issue, name=artifact_name,
                    )
                return {}
            if isinstance(payload, dict) and payload:
                self._mark_real(artifact_name)
                if as_artifact:
                    return DashboardArtifact(
                        data=payload, is_real=True,
                        source_path=str(path), name=artifact_name,
                    )
                return payload
            issue = f"Required artifact {path} is empty or invalid."
            self._mark_missing(artifact_name, issue)
            if as_artifact:
                return DashboardArtifact(
                    data={}, is_real=False, source_path=str(path),
                    reason=issue, name=artifact_name,
                )
            return {}

        issue = (
            "Required artifact missing: survival_curves.json. Dashboard "
            "will NOT render synthetic data. Run `python -m src.main "
            "--mode all` to produce real Kaplan-Meier curves."
        )
        self._mark_missing(artifact_name, issue)
        if as_artifact:
            return DashboardArtifact(
                data={}, is_real=False, source_path=None,
                reason=issue, name=artifact_name,
            )
        return {}

    # -----------------------------------------------------------------
    # Deprecated sample data generators for enhanced views (iter13)
    # All fallbacks removed; stubs raise FileNotFoundError.
    # -----------------------------------------------------------------

    def _generate_sample_mlflow_runs(self) -> pd.DataFrame:
        raise self._missing_artifact_error("model_performance_history.csv")

    def _generate_sample_roc_data(self) -> Dict[str, Dict[str, list]]:
        raise self._missing_artifact_error("roc_data.json")

    def _generate_sample_confusion_matrices(self) -> Dict[str, list]:
        raise self._missing_artifact_error("confusion_matrices.json")

    def _generate_sample_ab_detailed(self) -> Dict[str, Any]:
        raise self._missing_artifact_error("ab_test_detailed.json")

    def _generate_sample_survival_curves(
        self,
    ) -> Dict[str, Dict[str, list]]:
        raise self._missing_artifact_error("survival_curves.json")

    def load_scoring_history(
        self, as_artifact: bool = False
    ) -> Union[pd.DataFrame, DashboardArtifact]:
        """Load real-time scoring history log.

        Returns:
            DataFrame with customer_id, churn_probability, risk_level,
            recommended_action, model_type, scored_at columns. Empty
            frame when ``scoring_history.csv`` is missing — the 200-row
            ``np.random.beta`` fallback has been removed.
        """
        artifact_name = "scoring_history"
        empty_columns = [
            "customer_id", "churn_probability", "risk_level",
            "recommended_action", "model_type", "scored_at",
        ]
        path = self._resolve_existing_path("scoring_history.csv")
        if path is not None:
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                issue = f"Required artifact {path} could not be read: {exc}."
                self._mark_missing(artifact_name, issue)
                empty = self._empty_frame(empty_columns, artifact_name, issue)
                if as_artifact:
                    return DashboardArtifact(
                        data=empty, is_real=False, source_path=str(path),
                        reason=issue, name=artifact_name,
                    )
                return empty
            self._mark_real(artifact_name)
            if as_artifact:
                return DashboardArtifact(
                    data=df, is_real=True, source_path=str(path),
                    name=artifact_name,
                )
            return df

        issue = (
            "Required artifact missing: scoring_history.csv. Dashboard "
            "will NOT render synthetic data. Run `python -m src.main "
            "--mode all` to produce real scoring history."
        )
        self._mark_missing(artifact_name, issue)
        empty = self._empty_frame(empty_columns, artifact_name, issue)
        if as_artifact:
            return DashboardArtifact(
                data=empty, is_real=False, source_path=None,
                reason=issue, name=artifact_name,
            )
        return empty

    def load_drift_history(self) -> pd.DataFrame:
        """Load model drift detection history.

        Returns:
            DataFrame with timestamp, alert_level, num_drifted_features,
            psi_mean, ks_mean columns.
        """
        path = self._resolve_existing_path("drift_history.csv")
        if path is not None:
            self._clear_artifact_issue("monitoring_report")
            return pd.read_csv(path)

        report_path = self._resolve_existing_path("monitoring_report.json")
        empty_columns = [
            "timestamp", "alert_level", "num_drifted_features",
            "psi_mean", "ks_mean",
        ]
        if report_path is None:
            issue = (
                "Required artifact missing: monitoring_report.json. "
                "Drift history cannot use generated samples for required dashboard evidence."
            )
            self._record_artifact_issue("monitoring_report", issue)
            return self._empty_frame(empty_columns, "monitoring_report", issue)

        try:
            report = self._read_json(report_path)
        except Exception as exc:
            issue = f"Required artifact {report_path} could not be read: {exc}."
            self._record_artifact_issue("monitoring_report", issue)
            return self._empty_frame(empty_columns, "monitoring_report", issue)

        if not isinstance(report, dict):
            issue = f"Required artifact {report_path} is invalid; expected a JSON object."
            self._record_artifact_issue("monitoring_report", issue)
            return self._empty_frame(empty_columns, "monitoring_report", issue)

        psi_report = report.get("psi_report", {})
        ks_report = report.get("ks_report", {})
        if not psi_report and not ks_report:
            issue = (
                f"Required artifact {report_path} is invalid; "
                "missing PSI/KS drift reports."
            )
            self._record_artifact_issue("monitoring_report", issue)
            return self._empty_frame(empty_columns, "monitoring_report", issue)

        psi_alerts = (
            psi_report.get("feature_alerts")
            or psi_report.get("alerts", {})
            or {}
        )
        ks_alerts = (
            ks_report.get("feature_alerts")
            or ks_report.get("alerts", {})
            or {}
        )
        psi_values = [
            float(alert.get("psi_value", 0.0))
            for alert in psi_alerts.values()
        ]
        ks_values = [
            float(alert.get("statistic", 0.0))
            for alert in ks_alerts.values()
        ]
        self._clear_artifact_issue("monitoring_report")
        return pd.DataFrame([{
            "timestamp": report.get("timestamp"),
            "alert_level": report.get("overall_alert_level", "green"),
            "num_drifted_features": len(report.get("drifted_features", [])),
            "psi_mean": float(np.mean(psi_values)) if psi_values else 0.0,
            "ks_mean": float(np.mean(ks_values)) if ks_values else 0.0,
        }])

    def load_performance_alerts(self) -> Dict[str, Any]:
        """Load or derive model performance degradation alerts.

        New monitoring reports contain a top-level ``performance_alerts``
        payload. Older reports only contain ``performance.latest`` or the
        standalone model performance history CSV, so this loader derives the
        same alert shape from those artifacts for backward compatibility.
        """
        report_path = self._resolve_existing_path("monitoring_report.json")
        if report_path is not None:
            try:
                report = self._read_json(report_path)
            except Exception as exc:
                self._record_artifact_issue(
                    "performance_alerts",
                    f"Monitoring report could not be read for performance alerts: {exc}.",
                )
                return {}
            if isinstance(report, dict):
                alerts = report.get("performance_alerts")
                performance_section = report.get("performance", {})
                if not alerts and isinstance(performance_section, dict):
                    alerts = (
                        performance_section.get("performance_alerts")
                        or performance_section.get("alerts")
                    )
                if isinstance(alerts, dict):
                    self._clear_artifact_issue("performance_alerts")
                    return self._normalize_performance_alerts(alerts)

                latest = (
                    performance_section.get("latest")
                    if isinstance(performance_section, dict)
                    else None
                )
                if latest:
                    derived = self._derive_performance_alerts(
                        pd.DataFrame(latest)
                    )
                    if derived.get("metrics"):
                        return derived
                    snapshot = self._snapshot_performance_alerts(
                        pd.DataFrame(latest), derived
                    )
                    if snapshot.get("metrics"):
                        return snapshot

        history = self.load_model_performance_history()
        if history.empty:
            return {}
        derived = self._derive_performance_alerts(history)
        if derived.get("metrics"):
            return derived
        return self._snapshot_performance_alerts(history, derived)

    def _derive_performance_alerts(
        self, history: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Derive performance alerts from a history dataframe."""
        from src.monitoring.monitoring_service import (
            evaluate_performance_degradation,
        )

        alerts = evaluate_performance_degradation(
            history,
            thresholds=self.config,
        )
        return self._normalize_performance_alerts(alerts)

    def _snapshot_performance_alerts(
        self,
        history: pd.DataFrame,
        base_alerts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Expose current metric fields when degradation history is too short."""
        if history is None or history.empty:
            return self._normalize_performance_alerts(base_alerts or {})

        snapshot = history.copy()
        if "model_type" not in snapshot.columns and "model" in snapshot.columns:
            snapshot = snapshot.rename(columns={"model": "model_type"})
        if "auc" not in snapshot.columns and "auc_roc" in snapshot.columns:
            snapshot["auc"] = snapshot["auc_roc"]
        if "model_type" in snapshot.columns:
            models = snapshot["model_type"].astype(str)
            if (models == "ensemble").any():
                snapshot = snapshot.loc[models == "ensemble"]
        if "timestamp" in snapshot.columns:
            snapshot = snapshot.sort_values("timestamp")

        current_row = snapshot.iloc[-1]
        normalized = self._normalize_performance_alerts(base_alerts or {})
        thresholds = normalized.get("thresholds", {})
        current_timestamp = str(current_row.get("timestamp", ""))
        metrics: Dict[str, Dict[str, Any]] = {}
        for metric in ["auc", "precision", "recall", "f1_score", "accuracy"]:
            if metric not in snapshot.columns:
                continue
            current = pd.to_numeric(
                pd.Series([current_row.get(metric)]), errors="coerce"
            ).iloc[0]
            if pd.isna(current):
                continue
            current = float(current)
            metrics[metric] = {
                "metric": metric,
                "current": current,
                "baseline": current,
                "drop": 0.0,
                "threshold": float(thresholds.get(metric, 0.0)),
                "status": "ok",
                "current_timestamp": current_timestamp,
                "baseline_timestamp": current_timestamp,
            }

        normalized["metrics"] = metrics
        normalized["metric_alerts"] = metrics
        if metrics:
            normalized["status"] = "ok"
            normalized["alert_level"] = "green"
            normalized["performance_degradation"] = False
            normalized["degraded_metrics"] = []
        return normalized

    @staticmethod
    def _normalize_performance_alerts(
        alerts: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize legacy and current performance alert payload names."""
        normalized = dict(alerts)
        metrics = (
            normalized.get("metrics")
            or normalized.get("metric_alerts")
            or normalized.get("alerts")
            or {}
        )
        normalized["metrics"] = metrics
        normalized["metric_alerts"] = metrics
        degradation = bool(
            normalized.get("performance_degradation")
            or normalized.get("degraded")
            or normalized.get("status") == "degraded"
            or normalized.get("degraded_metrics")
        )
        normalized["performance_degradation"] = degradation
        normalized.setdefault(
            "status",
            "degraded" if degradation else "ok",
        )
        normalized.setdefault(
            "alert_level",
            "red" if degradation else "green",
        )
        normalized.setdefault("degraded_metrics", [])
        return normalized

    def load_auc_history(self) -> pd.DataFrame:
        """Load AUC metric time series."""
        return self._load_metric_timeseries("auc")

    def load_precision_history(self) -> pd.DataFrame:
        """Load Precision metric time series."""
        return self._load_metric_timeseries("precision")

    def load_recall_history(self) -> pd.DataFrame:
        """Load Recall metric time series."""
        return self._load_metric_timeseries("recall")

    def load_scoring_throughput(
        self, as_artifact: bool = False
    ) -> Union[pd.DataFrame, DashboardArtifact]:
        """Load scoring throughput metrics over time.

        Returns:
            DataFrame with timestamp, requests_per_minute,
            avg_latency_ms, error_rate columns. Empty frame when
            ``scoring_throughput.csv`` is missing — the 48-point
            sinusoidal sample fallback has been removed.
        """
        artifact_name = "scoring_throughput"
        empty_columns = [
            "timestamp", "requests_per_minute",
            "avg_latency_ms", "error_rate",
        ]
        path = self._resolve_existing_path("scoring_throughput.csv")
        if path is not None:
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                issue = f"Required artifact {path} could not be read: {exc}."
                self._mark_missing(artifact_name, issue)
                empty = self._empty_frame(empty_columns, artifact_name, issue)
                if as_artifact:
                    return DashboardArtifact(
                        data=empty, is_real=False, source_path=str(path),
                        reason=issue, name=artifact_name,
                    )
                return empty
            self._mark_real(artifact_name)
            if as_artifact:
                return DashboardArtifact(
                    data=df, is_real=True, source_path=str(path),
                    name=artifact_name,
                )
            return df

        issue = (
            "Required artifact missing: scoring_throughput.csv. "
            "Dashboard will NOT render synthetic data. Run "
            "`python -m src.main --mode all` to produce real throughput "
            "metrics."
        )
        self._mark_missing(artifact_name, issue)
        empty = self._empty_frame(empty_columns, artifact_name, issue)
        if as_artifact:
            return DashboardArtifact(
                data=empty, is_real=False, source_path=None,
                reason=issue, name=artifact_name,
            )
        return empty

    def load_retention_offers(
        self, as_artifact: bool = False
    ) -> Union[pd.DataFrame, DashboardArtifact]:
        """Load personalized retention offer details per customer.

        Returns:
            DataFrame with customer_id, segment, risk_level,
            churn_probability, offer_type, offer_detail,
            expected_uplift, estimated_cost_krw,
            expected_revenue_saved_krw, priority_score. Empty frame when
            ``retention_offers.csv`` is missing — the 50-row sample
            fallback has been removed.
        """
        artifact_name = "retention_offers"
        empty_columns = [
            "customer_id", "segment", "risk_level", "churn_probability",
            "offer_type", "offer_detail", "expected_uplift",
            "estimated_cost_krw", "expected_revenue_saved_krw",
            "priority_score",
        ]
        path = self._resolve_existing_path("retention_offers.csv")
        if path is not None:
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                issue = f"Required artifact {path} could not be read: {exc}."
                self._mark_missing(artifact_name, issue)
                empty = self._empty_frame(empty_columns, artifact_name, issue)
                if as_artifact:
                    return DashboardArtifact(
                        data=empty, is_real=False, source_path=str(path),
                        reason=issue, name=artifact_name,
                    )
                return empty
            self._mark_real(artifact_name)
            if as_artifact:
                return DashboardArtifact(
                    data=df, is_real=True, source_path=str(path),
                    name=artifact_name,
                )
            return df

        issue = (
            "Required artifact missing: retention_offers.csv. Dashboard "
            "will NOT render synthetic data. Run `python -m src.main "
            "--mode all` to produce real retention offers."
        )
        self._mark_missing(artifact_name, issue)
        empty = self._empty_frame(empty_columns, artifact_name, issue)
        if as_artifact:
            return DashboardArtifact(
                data=empty, is_real=False, source_path=None,
                reason=issue, name=artifact_name,
            )
        return empty

    # -----------------------------------------------------------------
    # Deprecated sample data generators for real-time scoring views
    # All fallbacks removed (iter13); stubs raise FileNotFoundError.
    # -----------------------------------------------------------------

    def _generate_sample_scoring_history(self) -> pd.DataFrame:
        raise self._missing_artifact_error("scoring_history.csv")

    def _generate_sample_drift_history(self) -> pd.DataFrame:
        raise self._missing_artifact_error("drift_history.csv")

    def _generate_sample_scoring_throughput(self) -> pd.DataFrame:
        raise self._missing_artifact_error("scoring_throughput.csv")

    def _generate_sample_retention_offers(self) -> pd.DataFrame:
        raise self._missing_artifact_error("retention_offers.csv")

    def _generate_sample_feature_importance(self) -> pd.DataFrame:
        raise self._missing_artifact_error("feature_importance.csv")
