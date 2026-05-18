"""
Cohort Analysis Module for E-Commerce Churn Prediction.

Provides time-based and behavior-based cohort assignment, retention matrix
computation, and cohort-level metric aggregation for understanding customer
lifecycle patterns.

Cohort Types:
    1. Monthly Acquisition Cohort - Groups by first purchase/signup month
    2. Weekly Acquisition Cohort - Groups by first purchase/signup week
    3. Behavioral Cohort - Groups by initial behavior patterns (e.g., channel, segment)

Key Metrics:
    - Retention Rate: % of cohort active in each subsequent period
    - Revenue per Cohort: Aggregated monetary value over time
    - Churn Rate per Cohort: Proportion churned by period
    - Average Order Value per Cohort: Mean transaction value over time

Usage:
    analyzer = CohortAnalyzer(config=cohort_config)
    cohorts = analyzer.assign_cohorts(events_df, cohort_type="monthly")
    retention = analyzer.compute_retention_matrix(cohorts)
    metrics = analyzer.compute_cohort_metrics(cohorts)
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


# Default cohort analysis configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "cohort_type": "monthly",           # "monthly", "weekly", or "behavioral"
    "periods": 12,                      # Number of periods to track
    "min_cohort_size": 5,               # Minimum customers per cohort
    "behavioral_column": "segment",     # Column for behavioral cohort grouping
    "date_column": "event_date",        # Column containing event timestamps
    "customer_column": "customer_id",   # Column containing customer identifier
    "revenue_column": "revenue",        # Column containing revenue/monetary value
    "metrics": [                        # Metrics to compute
        "retention_rate",
        "revenue",
        "avg_order_value",
        "churn_rate",
        "customer_count",
    ],
}


def _normalize_event_schema(events_df: pd.DataFrame) -> pd.DataFrame:
    """Return event data with canonical columns used by cohort helpers."""
    columns: Dict[str, pd.Series] = {}
    if "customer_id" in events_df.columns:
        columns["customer_id"] = events_df["customer_id"]

    date_source = None
    for candidate in ("event_date", "timestamp", "event_timestamp", "date"):
        if candidate in events_df.columns:
            date_source = candidate
            break
    if date_source is not None:
        dates = pd.to_datetime(events_df[date_source])
        columns["event_date"] = (
            dates.dt.normalize() if date_source != "event_date" else dates
        )

    type_source = None
    for candidate in ("event_type", "event_name", "type", "activity", "event"):
        if candidate in events_df.columns:
            type_source = candidate
            break
    if type_source is not None:
        columns["event_type"] = events_df[type_source]

    missing = {"customer_id", "event_date", "event_type"} - set(columns)
    if missing:
        raise KeyError(
            "Event data missing required cohort columns: "
            + ", ".join(sorted(missing))
        )
    events = pd.DataFrame(columns, copy=False).copy()
    events["event_type"] = events["event_type"].astype(str)
    return events


def _mean_median_days(values: pd.Series) -> Tuple[float, float]:
    """Return rounded mean/median day values, preserving NaN for no evidence."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan"), float("nan")
    return round(float(numeric.mean()), 2), round(float(numeric.median()), 2)


class CohortAnalyzer:
    """Cohort analysis for customer lifecycle and retention tracking.

    Supports monthly, weekly, and behavioral cohort assignment with
    configurable retention matrix computation and metric aggregation.

    Attributes:
        config: Cohort analysis configuration dictionary.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the cohort analyzer.

        Args:
            config: Cohort analysis configuration. If None, uses defaults.
                Expected keys: cohort_type, periods, min_cohort_size,
                behavioral_column, date_column, customer_column,
                revenue_column, metrics.
        """
        self.config: Dict[str, Any] = {**DEFAULT_CONFIG}
        if config:
            self.config.update(config)

    # ------------------------------------------------------------------
    # Cohort Assignment
    # ------------------------------------------------------------------

    def assign_cohorts(
        self,
        data: pd.DataFrame,
        cohort_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Assign each customer to a cohort based on their first activity.

        Args:
            data: DataFrame with at least customer_column and date_column.
                For behavioral cohorts, also requires behavioral_column.
            cohort_type: Override cohort type ("monthly", "weekly", or
                "behavioral"). If None, uses config default.

        Returns:
            DataFrame with original columns plus:
                - cohort: The assigned cohort label
                - cohort_period: The period index relative to cohort start
        """
        cohort_type = cohort_type or self.config["cohort_type"]
        customer_col = self.config["customer_column"]
        date_col = self.config["date_column"]

        result = data.copy()

        # Ensure date column is datetime
        if date_col in result.columns:
            result[date_col] = pd.to_datetime(result[date_col])

        if cohort_type == "monthly":
            result = self._assign_monthly_cohorts(result, customer_col, date_col)
        elif cohort_type == "weekly":
            result = self._assign_weekly_cohorts(result, customer_col, date_col)
        elif cohort_type == "behavioral":
            result = self._assign_behavioral_cohorts(result, customer_col)
        else:
            raise ValueError(
                f"Unknown cohort_type: {cohort_type}. "
                "Expected 'monthly', 'weekly', or 'behavioral'."
            )

        return result

    def _assign_monthly_cohorts(
        self,
        data: pd.DataFrame,
        customer_col: str,
        date_col: str,
    ) -> pd.DataFrame:
        """Assign cohorts by the month of each customer's first event.

        Args:
            data: DataFrame with customer and date columns.
            customer_col: Name of customer identifier column.
            date_col: Name of date column.

        Returns:
            DataFrame with cohort and cohort_period columns added.
        """
        use_signup_date = "signup_date" in data.columns

        if use_signup_date:
            first_event = (
                data[[customer_col, "signup_date"]]
                .dropna(subset=["signup_date"])
                .drop_duplicates(customer_col)
                .rename(columns={"signup_date": "first_event_date"})
            )
            first_event["first_event_date"] = pd.to_datetime(
                first_event["first_event_date"]
            )
        else:
            first_event = (
                data.groupby(customer_col)[date_col]
                .min()
                .reset_index()
                .rename(columns={date_col: "first_event_date"})
            )

        result = data.merge(first_event, on=customer_col, how="left")

        # Cohort = year-month of first event
        result["cohort"] = result["first_event_date"].dt.to_period("M").astype(str)

        if use_signup_date:
            result["cohort_period"] = (
                (result[date_col] - result["first_event_date"]).dt.days // 30
            ).clip(lower=0)
        else:
            # Period index = calendar months since cohort start
            result["cohort_period"] = (
                result[date_col].dt.to_period("M").astype("int64")
                - result["first_event_date"].dt.to_period("M").astype("int64")
            )

        result = result.drop(columns=["first_event_date"])
        return result

    def _assign_weekly_cohorts(
        self,
        data: pd.DataFrame,
        customer_col: str,
        date_col: str,
    ) -> pd.DataFrame:
        """Assign cohorts by the week of each customer's first event.

        Args:
            data: DataFrame with customer and date columns.
            customer_col: Name of customer identifier column.
            date_col: Name of date column.

        Returns:
            DataFrame with cohort and cohort_period columns added.
        """
        if "signup_date" in data.columns:
            first_event = (
                data[[customer_col, "signup_date"]]
                .dropna(subset=["signup_date"])
                .drop_duplicates(customer_col)
                .rename(columns={"signup_date": "first_event_date"})
            )
            first_event["first_event_date"] = pd.to_datetime(
                first_event["first_event_date"]
            )
        else:
            first_event = (
                data.groupby(customer_col)[date_col]
                .min()
                .reset_index()
                .rename(columns={date_col: "first_event_date"})
            )

        result = data.merge(first_event, on=customer_col, how="left")

        # Cohort = year-week of first event (ISO week)
        result["cohort"] = (
            result["first_event_date"].dt.isocalendar().year.astype(str)
            + "-W"
            + result["first_event_date"]
            .dt.isocalendar()
            .week.astype(str)
            .str.zfill(2)
        )

        # Period index = weeks since cohort start
        result["cohort_period"] = (
            (result[date_col] - result["first_event_date"]).dt.days // 7
        )

        result = result.drop(columns=["first_event_date"])
        return result

    def _assign_behavioral_cohorts(
        self,
        data: pd.DataFrame,
        customer_col: str,
    ) -> pd.DataFrame:
        """Assign cohorts based on a behavioral/categorical column.

        Uses the first observed value of the behavioral column for each
        customer as their cohort assignment.

        Args:
            data: DataFrame with customer and behavioral columns.
            customer_col: Name of customer identifier column.

        Returns:
            DataFrame with cohort column added. cohort_period is set to 0
            for behavioral cohorts (no temporal ordering).
        """
        behavioral_col = self.config["behavioral_column"]
        date_col = self.config["date_column"]

        if behavioral_col not in data.columns:
            raise ValueError(
                f"Behavioral column '{behavioral_col}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )

        result = data.copy()

        # Use first observed behavioral value per customer
        if date_col in result.columns:
            sorted_data = result.sort_values(date_col)
            first_behavior = (
                sorted_data.groupby(customer_col)[behavioral_col]
                .first()
                .reset_index()
                .rename(columns={behavioral_col: "cohort"})
            )
        else:
            first_behavior = (
                result.groupby(customer_col)[behavioral_col]
                .first()
                .reset_index()
                .rename(columns={behavioral_col: "cohort"})
            )

        result = result.merge(first_behavior, on=customer_col, how="left")

        # For behavioral cohorts, compute period from date if available
        if date_col in result.columns:
            first_event = (
                result.groupby(customer_col)[date_col]
                .min()
                .reset_index()
                .rename(columns={date_col: "first_event_date"})
            )
            result = result.merge(first_event, on=customer_col, how="left")
            result["cohort_period"] = (
                result[date_col].dt.to_period("M").astype("int64")
                - result["first_event_date"].dt.to_period("M").astype("int64")
            )
            result = result.drop(columns=["first_event_date"])
        else:
            result["cohort_period"] = 0

        return result

    # ------------------------------------------------------------------
    # Retention Matrix
    # ------------------------------------------------------------------

    def compute_retention_matrix(
        self,
        cohort_data: pd.DataFrame,
        max_periods: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute a retention matrix from cohort-assigned data.

        The retention matrix shows the percentage of each cohort's customers
        that remain active in each subsequent period.

        Args:
            cohort_data: DataFrame with cohort and cohort_period columns
                (output of assign_cohorts).
            max_periods: Maximum number of periods to include. If None,
                uses config['periods'].

        Returns:
            DataFrame with cohorts as rows, period indices as columns,
            and retention rates (0.0 to 1.0) as values.
        """
        max_periods = max_periods or self.config["periods"]
        customer_col = self.config["customer_column"]
        min_size = self.config["min_cohort_size"]

        # Count unique customers per cohort per period
        cohort_counts = (
            cohort_data.groupby(["cohort", "cohort_period"])[customer_col]
            .nunique()
            .reset_index()
            .rename(columns={customer_col: "customers"})
        )

        # Pivot to matrix form
        retention_pivot = cohort_counts.pivot_table(
            index="cohort",
            columns="cohort_period",
            values="customers",
            aggfunc="sum",
            fill_value=0,
        )

        # Filter to max_periods
        valid_cols = [c for c in retention_pivot.columns if c <= max_periods]
        retention_pivot = retention_pivot[sorted(valid_cols)]

        # Filter out small cohorts
        if 0 in retention_pivot.columns:
            retention_pivot = retention_pivot[
                retention_pivot[0] >= min_size
            ]

        # Convert to retention rate (divide by period 0 count)
        if 0 in retention_pivot.columns:
            base_counts = retention_pivot[0]
            retention_matrix = retention_pivot.div(base_counts, axis=0)
        else:
            retention_matrix = retention_pivot

        # Retention is a cumulative survival-style measure in this dashboard:
        # once a customer is absent from a later cohort period, a later purchase
        # must not make the displayed retention curve increase again.
        retention_matrix = retention_matrix.sort_index(axis=1).cummin(axis=1)
        return retention_matrix

    # ------------------------------------------------------------------
    # Cohort Metrics Aggregation
    # ------------------------------------------------------------------

    def compute_cohort_metrics(
        self,
        cohort_data: pd.DataFrame,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Compute multiple metrics aggregated by cohort and period.

        Args:
            cohort_data: DataFrame with cohort, cohort_period, and
                relevant value columns (output of assign_cohorts).
            metrics: List of metric names to compute. If None, uses
                config['metrics']. Supported metrics:
                - "retention_rate": Customer retention per period
                - "revenue": Total revenue per cohort per period
                - "avg_order_value": Mean revenue per cohort per period
                - "churn_rate": 1 - retention_rate
                - "customer_count": Unique customers per cohort per period

        Returns:
            Dictionary mapping metric names to DataFrames with cohorts
            as rows and periods as columns.
        """
        metrics = metrics or self.config.get("metrics", [])
        customer_col = self.config["customer_column"]
        revenue_col = self.config["revenue_column"]
        max_periods = self.config["periods"]
        min_size = self.config["min_cohort_size"]

        results: Dict[str, pd.DataFrame] = {}

        for metric in metrics:
            if metric == "retention_rate":
                results["retention_rate"] = self.compute_retention_matrix(
                    cohort_data, max_periods=max_periods
                )

            elif metric == "churn_rate":
                retention = self.compute_retention_matrix(
                    cohort_data, max_periods=max_periods
                )
                results["churn_rate"] = 1.0 - retention

            elif metric == "revenue":
                if revenue_col not in cohort_data.columns:
                    continue
                rev_agg = (
                    cohort_data.groupby(["cohort", "cohort_period"])[revenue_col]
                    .sum()
                    .reset_index()
                )
                rev_pivot = rev_agg.pivot_table(
                    index="cohort",
                    columns="cohort_period",
                    values=revenue_col,
                    aggfunc="sum",
                    fill_value=0,
                )
                valid_cols = [c for c in rev_pivot.columns if c <= max_periods]
                results["revenue"] = rev_pivot[sorted(valid_cols)]

            elif metric == "avg_order_value":
                if revenue_col not in cohort_data.columns:
                    continue
                aov_agg = (
                    cohort_data.groupby(["cohort", "cohort_period"])[revenue_col]
                    .mean()
                    .reset_index()
                )
                aov_pivot = aov_agg.pivot_table(
                    index="cohort",
                    columns="cohort_period",
                    values=revenue_col,
                    aggfunc="mean",
                    fill_value=0,
                )
                valid_cols = [c for c in aov_pivot.columns if c <= max_periods]
                results["avg_order_value"] = aov_pivot[sorted(valid_cols)]

            elif metric == "customer_count":
                count_agg = (
                    cohort_data.groupby(["cohort", "cohort_period"])[customer_col]
                    .nunique()
                    .reset_index()
                    .rename(columns={customer_col: "count"})
                )
                count_pivot = count_agg.pivot_table(
                    index="cohort",
                    columns="cohort_period",
                    values="count",
                    aggfunc="sum",
                    fill_value=0,
                )
                valid_cols = [c for c in count_pivot.columns if c <= max_periods]
                results["customer_count"] = count_pivot[sorted(valid_cols)]

        return results

    # ------------------------------------------------------------------
    # Summary & Utility Methods
    # ------------------------------------------------------------------

    def get_cohort_summary(
        self,
        cohort_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate a summary of each cohort.

        Args:
            cohort_data: DataFrame with cohort assignments.

        Returns:
            DataFrame with one row per cohort containing:
                - cohort: Cohort label
                - total_customers: Unique customer count
                - total_events: Number of events
                - avg_lifetime_periods: Mean number of active periods
                - total_revenue: Sum of revenue (if available)
        """
        customer_col = self.config["customer_column"]
        revenue_col = self.config["revenue_column"]

        agg_dict: Dict[str, Any] = {
            customer_col: "nunique",
            "cohort_period": ["count", "max"],
        }

        summary = cohort_data.groupby("cohort").agg(agg_dict)
        summary.columns = [
            "total_customers",
            "total_events",
            "max_period",
        ]

        # Average lifetime periods per customer
        lifetime = (
            cohort_data.groupby(["cohort", customer_col])["cohort_period"]
            .max()
            .reset_index()
            .groupby("cohort")["cohort_period"]
            .mean()
        )
        summary["avg_lifetime_periods"] = lifetime

        # Revenue if available
        if revenue_col in cohort_data.columns:
            rev = cohort_data.groupby("cohort")[revenue_col].sum()
            summary["total_revenue"] = rev

        summary = summary.reset_index()
        summary = summary.sort_values("total_customers", ascending=False)
        summary = summary.reset_index(drop=True)

        return summary

    def filter_cohorts(
        self,
        cohort_data: pd.DataFrame,
        cohorts: Optional[List[str]] = None,
        min_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """Filter cohort data by cohort names or minimum size.

        Args:
            cohort_data: DataFrame with cohort assignments.
            cohorts: List of cohort labels to keep. If None, keeps all.
            min_size: Minimum unique customers per cohort. If None, uses
                config['min_cohort_size'].

        Returns:
            Filtered DataFrame.
        """
        customer_col = self.config["customer_column"]
        min_size = min_size if min_size is not None else self.config["min_cohort_size"]

        result = cohort_data.copy()

        if cohorts is not None:
            result = result[result["cohort"].isin(cohorts)]

        # Filter by minimum size
        cohort_sizes = result.groupby("cohort")[customer_col].nunique()
        valid_cohorts = cohort_sizes[cohort_sizes >= min_size].index
        result = result[result["cohort"].isin(valid_cohorts)]

        return result

    # ------------------------------------------------------------------
    # Retention Curves
    # ------------------------------------------------------------------

    def get_retention_curves(
        self,
        retention_matrix: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """Extract per-cohort retention curves from a retention matrix.

        Each curve is an array of retention rates over successive periods,
        starting from period 0 (always 1.0) through the last tracked period.

        Args:
            retention_matrix: DataFrame from ``compute_retention_matrix()``,
                with cohort labels as index and period offsets as columns.

        Returns:
            Dictionary mapping cohort label strings to 1-D numpy arrays
            of retention rates.
        """
        curves: Dict[str, np.ndarray] = {}
        for cohort_label in retention_matrix.index:
            row = retention_matrix.loc[cohort_label].dropna()
            curves[str(cohort_label)] = row.values.astype(float)
        return curves

    def get_average_retention_curve(
        self,
        retention_matrix: pd.DataFrame,
    ) -> np.ndarray:
        """Compute the average retention curve across all cohorts.

        For each period offset, averages the retention rate across all
        cohorts that have data for that period.

        Args:
            retention_matrix: DataFrame from ``compute_retention_matrix()``.

        Returns:
            1-D numpy array of average retention rates per period.
        """
        return retention_matrix.mean(axis=0, skipna=True).values.astype(float)

    def compute_half_life(
        self,
        retention_matrix: pd.DataFrame,
    ) -> pd.Series:
        """Compute the retention half-life for each cohort.

        Half-life is the first period offset where retention drops below 50%.

        Args:
            retention_matrix: DataFrame from ``compute_retention_matrix()``.

        Returns:
            Series indexed by cohort label with half-life period values.
            ``NaN`` if retention never drops below 50%.
        """
        curves = self.get_retention_curves(retention_matrix)
        half_lives: Dict[str, float] = {}
        for cohort_label, curve in curves.items():
            below_50 = np.where(curve < 0.5)[0]
            if len(below_50) > 0:
                half_lives[cohort_label] = int(below_50[0])
            else:
                half_lives[cohort_label] = np.nan
        return pd.Series(half_lives, name="half_life")

    def compute_churn_rates(
        self,
        retention_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute period-over-period churn rates from a retention matrix.

        Churn rate at period *t* is defined as:
            ``churn(t) = 1 - retention(t) / retention(t-1)``

        This gives the *incremental* churn between consecutive periods,
        not just ``1 - retention``.

        Args:
            retention_matrix: DataFrame from ``compute_retention_matrix()``.

        Returns:
            DataFrame with same shape as ``retention_matrix``, where each
            cell contains the incremental churn rate for that
            (cohort, period) pair. Period 0 is always 0.0.
        """
        shifted = retention_matrix.shift(1, axis=1)
        # Avoid division by zero
        churn = 1.0 - retention_matrix.div(shifted.replace(0, np.nan))
        # Period 0 churn is 0 by definition
        if 0 in churn.columns:
            churn[0] = 0.0
        return churn

    def extract_retention_milestones(
        self,
        retention_matrix: pd.DataFrame,
        milestones: Optional[List[int]] = None,
        include_m0: bool = True,
    ) -> pd.DataFrame:
        """Extract M1/M3/M6/M12 retention with latest-observed fallback.

        Exact milestone columns are used when present. For short observation
        windows, the most recent available period before the requested
        milestone is carried forward so required reporting columns are still
        populated and auditable instead of silently becoming all-null.
        """
        milestone_periods = list(milestones or [1, 3, 6, 12])
        selected_periods = ([0] if include_m0 else []) + milestone_periods
        available_periods = sorted(
            int(c) for c in retention_matrix.columns if isinstance(c, (int, np.integer))
        )

        data: Dict[str, pd.Series] = {}
        for period in selected_periods:
            label = f"M{period}"
            if period in retention_matrix.columns:
                data[label] = retention_matrix[period].astype(float)
            else:
                fallback_periods = [p for p in available_periods if p <= period]
                if fallback_periods:
                    data[label] = retention_matrix[fallback_periods[-1]].astype(float)
                else:
                    data[label] = pd.Series(
                        np.nan, index=retention_matrix.index, dtype=float
                    )

        milestone_df = pd.DataFrame(data, index=retention_matrix.index)
        milestone_df.index.name = retention_matrix.index.name
        return milestone_df

    def analyze_churn_signals(
        self,
        customers_df: pd.DataFrame,
        events_df: pd.DataFrame,
        n_days: int = 30,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """Bundle reusable churn-signal outputs for downstream reporting."""
        return {
            "top_sequences": extract_churn_sequences(
                events_df=events_df,
                customers_df=customers_df,
                n_days=n_days,
                top_n=top_n,
            ),
            "pre_churn_events": analyze_pre_churn_events(
                events_df=events_df,
                customers_df=customers_df,
                n_days=n_days,
            ),
            "journey_funnel": compute_journey_funnel(
                customers_df=customers_df,
                events_df=events_df,
            ),
        }

    def compute_cumulative_revenue(
        self,
        cohort_data: pd.DataFrame,
        max_periods: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute cumulative revenue per cohort over time.

        Args:
            cohort_data: DataFrame with cohort, cohort_period, and
                revenue columns.
            max_periods: Maximum periods to include. Defaults to
                config['periods'].

        Returns:
            DataFrame with cohorts as rows, periods as columns, and
            cumulative revenue as values.
        """
        max_periods = max_periods or self.config["periods"]
        revenue_col = self.config["revenue_column"]

        if revenue_col not in cohort_data.columns:
            raise ValueError(
                f"Revenue column '{revenue_col}' not found in data. "
                f"Available columns: {list(cohort_data.columns)}"
            )

        rev_agg = (
            cohort_data.groupby(["cohort", "cohort_period"])[revenue_col]
            .sum()
            .reset_index()
        )

        rev_pivot = rev_agg.pivot_table(
            index="cohort",
            columns="cohort_period",
            values=revenue_col,
            aggfunc="sum",
            fill_value=0,
        )

        valid_cols = sorted(c for c in rev_pivot.columns if c <= max_periods)
        rev_pivot = rev_pivot[valid_cols]

        # Cumulative sum across periods
        return rev_pivot.cumsum(axis=1)

    # ------------------------------------------------------------------
    # Visualization Methods
    # ------------------------------------------------------------------

    def plot_retention_heatmap(
        self,
        retention_matrix: pd.DataFrame,
        title: str = "Cohort Retention Heatmap",
        figsize: Tuple[int, int] = (12, 8),
        cmap: str = "YlOrRd",
        annot: bool = True,
        fmt: str = ".0%",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a heatmap of the retention matrix.

        Displays retention rates as a color-coded heatmap with cohorts
        on the y-axis and periods on the x-axis.

        Args:
            retention_matrix: DataFrame from ``compute_retention_matrix()``.
            title: Plot title.
            figsize: Figure size as (width, height).
            cmap: Matplotlib/seaborn colormap name.
            annot: Whether to annotate cells with retention values.
            fmt: Format string for annotations (e.g., ".0%" for percentages).
            save_path: If provided, saves the figure to this file path.

        Returns:
            The matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            retention_matrix,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            ax=ax,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Cohort Period", fontsize=12)
        ax.set_ylabel("Cohort", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Retention heatmap saved to %s", save_path)

        return fig

    def plot_retention_lines(
        self,
        retention_matrix: pd.DataFrame,
        title: str = "Cohort Retention Curves",
        figsize: Tuple[int, int] = (12, 6),
        show_average: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot retention curves as line plots, one line per cohort.

        Args:
            retention_matrix: DataFrame from ``compute_retention_matrix()``.
            title: Plot title.
            figsize: Figure size as (width, height).
            show_average: If True, overlay the average retention curve
                as a dashed black line.
            save_path: If provided, saves the figure to this file path.

        Returns:
            The matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)

        curves = self.get_retention_curves(retention_matrix)
        periods_all = sorted(retention_matrix.columns)

        for cohort_label, curve in curves.items():
            periods = periods_all[: len(curve)]
            ax.plot(periods, curve, marker="o", markersize=4, label=cohort_label)

        if show_average and len(curves) > 0:
            avg_curve = self.get_average_retention_curve(retention_matrix)
            avg_periods = periods_all[: len(avg_curve)]
            ax.plot(
                avg_periods,
                avg_curve,
                color="black",
                linewidth=2.5,
                linestyle="--",
                marker="s",
                markersize=5,
                label="Average",
                zorder=10,
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Cohort Period", fontsize=12)
        ax.set_ylabel("Retention Rate", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(title="Cohort", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=10)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Retention line plot saved to %s", save_path)
            path = Path(save_path)
            if path.name == "cohort_retention_curves.png":
                churn_diff_path = path.with_name("cohort_churn_rate_differences.png")
                churn_fig = self.plot_churn_rate_differences(
                    retention_matrix,
                    save_path=str(churn_diff_path),
                )
                plt.close(churn_fig)

        return fig

    def plot_churn_rate_differences(
        self,
        retention_matrix: pd.DataFrame,
        title: str = "Cohort Churn Rate Differences",
        figsize: Tuple[int, int] = (12, 7),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualize period churn-rate differences across cohorts.

        The chart directly compares each cohort's incremental churn rate by
        period and includes the cross-cohort spread so reviewers can verify
        that cohort-level churn differences were analyzed, not only tabulated.
        """
        churn_rates = self.compute_churn_rates(retention_matrix).copy()
        churn_rates = churn_rates.rename(
            columns={
                col: int(col)
                for col in churn_rates.columns
                if isinstance(col, str) and col.isdigit()
            }
        )
        period_cols = sorted(
            col for col in churn_rates.columns
            if isinstance(col, (int, np.integer)) and int(col) != 0
        )
        if not period_cols:
            period_cols = sorted(churn_rates.columns)

        plot_data = churn_rates[period_cols].astype(float)
        spread = plot_data.max(axis=0, skipna=True) - plot_data.min(axis=0, skipna=True)
        average = plot_data.mean(axis=0, skipna=True)

        fig, (ax_lines, ax_spread) = plt.subplots(
            2,
            1,
            figsize=figsize,
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )

        for cohort_label in plot_data.index:
            row = plot_data.loc[cohort_label]
            ax_lines.plot(
                period_cols,
                row.values,
                marker="o",
                markersize=4,
                linewidth=1.4,
                label=str(cohort_label),
            )

        ax_lines.plot(
            period_cols,
            average.values,
            color="black",
            linewidth=2.5,
            linestyle="--",
            marker="s",
            markersize=5,
            label="Average",
            zorder=10,
        )
        ax_lines.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.5)
        ax_lines.set_title(title, fontsize=14, fontweight="bold")
        ax_lines.set_ylabel("Incremental Churn Rate", fontsize=12)
        ax_lines.grid(True, alpha=0.3)
        ax_lines.legend(title="Cohort", bbox_to_anchor=(1.05, 1), loc="upper left")

        ax_spread.bar(
            period_cols,
            spread.values,
            color="#4c78a8",
            edgecolor="white",
            linewidth=0.5,
        )
        ax_spread.set_xlabel("Cohort Period", fontsize=12)
        ax_spread.set_ylabel("Spread", fontsize=12)
        ax_spread.grid(axis="y", alpha=0.3)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Churn rate difference plot saved to %s", save_path)

        return fig

    def plot_cohort_sizes(
        self,
        cohort_data: pd.DataFrame,
        title: str = "Cohort Sizes",
        figsize: Tuple[int, int] = (10, 6),
        color: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a bar chart of cohort sizes (unique customers per cohort).

        Args:
            cohort_data: DataFrame with cohort assignments
                (output of ``assign_cohorts()``).
            title: Plot title.
            figsize: Figure size as (width, height).
            color: Bar color. If None, uses the default seaborn palette.
            save_path: If provided, saves the figure to this file path.

        Returns:
            The matplotlib Figure object.
        """
        customer_col = self.config["customer_column"]

        cohort_sizes = (
            cohort_data.groupby("cohort")[customer_col]
            .nunique()
            .sort_index()
        )

        fig, ax = plt.subplots(figsize=figsize)

        bar_color = color or sns.color_palette()[0]
        ax.bar(
            range(len(cohort_sizes)),
            cohort_sizes.values,
            color=bar_color,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_xticks(range(len(cohort_sizes)))
        ax.set_xticklabels(cohort_sizes.index, rotation=45, ha="right")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Cohort", fontsize=12)
        ax.set_ylabel("Number of Customers", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)

        # Add value labels on top of bars
        for i, (_, val) in enumerate(cohort_sizes.items()):
            ax.text(
                i,
                val + max(cohort_sizes.values) * 0.01,
                str(val),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Cohort size bar chart saved to %s", save_path)

        return fig

    def full_analysis(
        self,
        data: pd.DataFrame,
        cohort_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a complete cohort analysis pipeline.

        Convenience method that chains cohort assignment, retention matrix
        computation, retention curves, churn rates, summary, and half-life.

        Args:
            data: Raw event-level DataFrame.
            cohort_type: Cohort type override.

        Returns:
            Dictionary with keys:
                - ``cohort_data``: DataFrame with cohort assignments
                - ``retention_matrix``: Retention rate matrix
                - ``retention_curves``: Per-cohort retention curves
                - ``avg_retention_curve``: Average curve across cohorts
                - ``churn_rates``: Period-over-period churn rates
                - ``half_life``: Half-life per cohort
                - ``summary``: Cohort summary table
                - ``metrics``: Dict of all configured metric matrices
        """
        cohort_data = self.assign_cohorts(data, cohort_type=cohort_type)
        cohort_data = self.filter_cohorts(cohort_data)

        retention_matrix = self.compute_retention_matrix(cohort_data)
        retention_curves = self.get_retention_curves(retention_matrix)
        avg_curve = self.get_average_retention_curve(retention_matrix)
        churn_rates = self.compute_churn_rates(retention_matrix)
        half_life = self.compute_half_life(retention_matrix)
        summary = self.get_cohort_summary(cohort_data)
        metrics = self.compute_cohort_metrics(cohort_data)

        return {
            "cohort_data": cohort_data,
            "retention_matrix": retention_matrix,
            "retention_curves": retention_curves,
            "avg_retention_curve": avg_curve,
            "churn_rates": churn_rates,
            "milestone_retention": self.extract_retention_milestones(
                retention_matrix
            ),
            "half_life": half_life,
            "summary": summary,
            "metrics": metrics,
        }


def extract_churn_sequences(
    events_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    n_days: int = 30,
    top_n: int = 5,
) -> List[Tuple[str, int]]:
    """Extract common event-type sequences from churned customers' last N days.

    For each churned customer, collects the event types observed in the final
    ``n_days`` of activity ordered chronologically, converts each customer's
    sequence to a single whitespace-separated string, then counts how often
    each unique sequence appears across all churned customers.

    Args:
        events_df: DataFrame with columns ``customer_id``, ``event_type``,
            and ``event_date`` (str or datetime).
        customers_df: DataFrame with columns ``customer_id`` and
            ``churn_label`` (1 = churned, 0 = active).
        n_days: Number of days before each churned customer's last event to
            include. Defaults to 30.
        top_n: Number of most-common sequences to return. Defaults to 5.

    Returns:
        List of ``(sequence_string, count)`` tuples for the top ``top_n``
        most common pre-churn event sequences, ordered by descending
        frequency.
    """
    events = _normalize_event_schema(events_df)

    churned_ids = customers_df.loc[
        customers_df["churn_label"] == 1, "customer_id"
    ]
    churned_events = events[events["customer_id"].isin(churned_ids)]

    sequences: List[str] = []
    for customer_id, group in churned_events.groupby("customer_id"):
        group = group.sort_values("event_date")
        last_date = group["event_date"].max()
        cutoff = last_date - pd.Timedelta(days=n_days)
        window = group[group["event_date"] >= cutoff]
        seq = " -> ".join(window["event_type"].tolist())
        if seq:
            sequences.append(seq)

    counter = Counter(sequences)
    top = counter.most_common(top_n)
    logger.info(
        "extract_churn_sequences: %d churned customers, top %d sequences extracted",
        len(sequences),
        len(top),
    )
    return top


def analyze_pre_churn_events(
    events_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    n_days: int = 30,
) -> pd.DataFrame:
    """Analyze event-type frequencies in the pre-churn window for churned vs active customers.

    For each customer (churned and active), counts how many times each event
    type occurs in their most recent ``n_days`` of activity. Compares the
    mean frequency per event type between the two groups.

    Args:
        events_df: DataFrame with columns ``customer_id``, ``event_type``,
            and ``event_date`` (str or datetime).
        customers_df: DataFrame with columns ``customer_id`` and
            ``churn_label`` (1 = churned, 0 = active).
        n_days: Lookback window in days from each customer's last event.
            Defaults to 30.

    Returns:
        DataFrame indexed by ``event_type`` with columns:

        - ``churned_freq``: Mean event count per churned customer.
        - ``active_freq``: Mean event count per active customer.
        - ``freq_ratio``: ``churned_freq / active_freq`` (NaN when
          ``active_freq`` is 0).
    """
    events = _normalize_event_schema(events_df)

    churn_map = customers_df.set_index("customer_id")["churn_label"]

    # Compute each customer's last event date
    last_dates = events.groupby("customer_id")["event_date"].max()

    # Keep only events within the n_days window before each customer's last date
    events = events.join(last_dates.rename("last_date"), on="customer_id")
    events = events[
        events["event_date"] >= events["last_date"] - pd.Timedelta(days=n_days)
    ]

    # Count events per customer per event type
    counts = (
        events.groupby(["customer_id", "event_type"])
        .size()
        .reset_index(name="count")
    )
    counts["churn_label"] = counts["customer_id"].map(churn_map)
    counts = counts.dropna(subset=["churn_label"])

    churned_counts = counts[counts["churn_label"] == 1]
    active_counts = counts[counts["churn_label"] == 0]

    churned_freq = (
        churned_counts.groupby("event_type")["count"].mean().rename("churned_freq")
    )
    active_freq = (
        active_counts.groupby("event_type")["count"].mean().rename("active_freq")
    )

    result = pd.concat([churned_freq, active_freq], axis=1).fillna(0.0)
    result["freq_ratio"] = result["churned_freq"] / result["active_freq"].replace(
        0, float("nan")
    )
    result = result.sort_values("churned_freq", ascending=False)

    logger.info(
        "analyze_pre_churn_events: %d event types analysed over %d-day window",
        len(result),
        n_days,
    )
    return result


def compute_journey_funnel(
    customers_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute a customer journey funnel with stage timing evidence.

    Stages (in order):
        1. **Signup** – all customers in ``customers_df``.
        2. **First Purchase** – customers with at least one ``purchase`` event.
        3. **Repeat Purchase** – customers with two or more ``purchase`` events.
        4. **Loyal** – customers with five or more ``purchase`` events.
        5. **Churned** – customers whose ``churn_label`` equals 1.

    Args:
        customers_df: DataFrame with columns ``customer_id`` and
            ``churn_label`` (1 = churned, 0 = active).
        events_df: DataFrame with columns ``customer_id`` and ``event_type``.

    Returns:
        DataFrame with one row per stage and the following columns:

        - ``stage``: Stage name string.
        - ``count``: Number of customers who reached this stage.
        - ``conversion_rate``: Fraction of all customers who reached this
          stage (relative to the Signup stage).
        - ``drop_off_rate``: Fraction of customers who did *not* proceed from
          the previous stage to this one (0.0 for the Signup stage).
        - ``avg_days_since_signup`` / ``median_days_since_signup``: Tenure at
          the stage timestamp.
        - ``avg_days_from_previous_stage`` /
          ``median_days_from_previous_stage``: Stage transition duration.
        - ``dropoff_customer_count`` and ``*_dropoff_days_after_previous_stage``:
          customers who stopped before reaching the current stage and their
          timing evidence.
    """
    events = _normalize_event_schema(events_df)
    customers = customers_df.copy()
    customers["customer_id"] = customers["customer_id"].astype(str)
    events["customer_id"] = events["customer_id"].astype(str)

    all_customers = set(customers["customer_id"])
    first_event_dates = events.groupby("customer_id")["event_date"].min()
    last_event_dates = events.groupby("customer_id")["event_date"].max()

    if "signup_date" in customers.columns:
        signup_dates = pd.to_datetime(customers["signup_date"], errors="coerce")
        signup_dates.index = customers["customer_id"]
        signup_dates = signup_dates.fillna(first_event_dates)
    else:
        signup_dates = customers["customer_id"].map(first_event_dates)
        signup_dates.index = customers["customer_id"]
    signup_dates = pd.to_datetime(signup_dates, errors="coerce")

    purchases = (
        events[events["event_type"] == "purchase"]
        .sort_values(["customer_id", "event_date"])
        .copy()
    )
    if purchases.empty:
        purchase_counts = pd.Series(dtype=int)
        purchase_dates = pd.DataFrame(
            index=pd.Index([], name="customer_id"),
            columns=["first_purchase", "repeat_purchase", "loyal"],
            dtype="datetime64[ns]",
        )
    else:
        purchases["purchase_number"] = (
            purchases.groupby("customer_id").cumcount() + 1
        )
        purchase_counts = purchases.groupby("customer_id").size()
        purchase_dates = pd.DataFrame(index=purchase_counts.index)
        for purchase_number, column in (
            (1, "first_purchase"),
            (2, "repeat_purchase"),
            (5, "loyal"),
        ):
            purchase_dates[column] = (
                purchases.loc[purchases["purchase_number"] == purchase_number]
                .drop_duplicates("customer_id")
                .set_index("customer_id")["event_date"]
            )

    first_purchase = set(purchase_counts[purchase_counts >= 1].index) & all_customers
    repeat_purchase = set(purchase_counts[purchase_counts >= 2].index) & all_customers
    loyal = set(purchase_counts[purchase_counts >= 5].index) & all_customers

    if "churn_label" in customers.columns:
        churn_labels = pd.to_numeric(
            customers["churn_label"], errors="coerce"
        ).fillna(0).astype(int)
        churned = set(
            customers.loc[churn_labels == 1, "customer_id"]
        ) & all_customers
    else:
        churned = set()

    churn_dates = pd.Series(
        pd.NaT,
        index=customers["customer_id"],
        dtype="datetime64[ns]",
    )
    for churn_col in ("churn_date", "churned_at", "churn_timestamp"):
        if churn_col in customers.columns:
            churn_dates = pd.to_datetime(
                customers.set_index("customer_id")[churn_col],
                errors="coerce",
            )
            break
    churn_dates = churn_dates.fillna(last_event_dates)
    churn_dates = churn_dates.where(churn_dates.index.isin(churned), pd.NaT)

    stage_dates = pd.DataFrame(index=pd.Index(sorted(all_customers), name="customer_id"))
    stage_dates["Signup"] = signup_dates.reindex(stage_dates.index)
    stage_dates["First Purchase"] = purchase_dates.get(
        "first_purchase", pd.Series(dtype="datetime64[ns]")
    ).reindex(stage_dates.index)
    stage_dates["Repeat Purchase"] = purchase_dates.get(
        "repeat_purchase", pd.Series(dtype="datetime64[ns]")
    ).reindex(stage_dates.index)
    stage_dates["Loyal"] = purchase_dates.get(
        "loyal", pd.Series(dtype="datetime64[ns]")
    ).reindex(stage_dates.index)
    stage_dates["Churned"] = churn_dates.reindex(stage_dates.index)
    stage_dates["last_event_date"] = last_event_dates.reindex(stage_dates.index)

    stage_members = {
        "Signup": all_customers,
        "First Purchase": first_purchase,
        "Repeat Purchase": repeat_purchase,
        "Loyal": loyal,
        "Churned": churned,
    }

    stages = [
        ("Signup", len(all_customers)),
        ("First Purchase", len(first_purchase)),
        ("Repeat Purchase", len(repeat_purchase)),
        ("Loyal", len(loyal)),
        ("Churned", len(churned)),
    ]

    total = stages[0][1] if stages[0][1] > 0 else 1
    rows = []
    for i, (stage, count) in enumerate(stages):
        conversion_rate = count / total
        if i == 0:
            drop_off_rate = 0.0
            previous_stage = None
        else:
            prev_count = stages[i - 1][1]
            drop_off_rate = (
                (prev_count - count) / prev_count if prev_count > 0 else 0.0
            )
            previous_stage = stages[i - 1][0]

        member_index = list(stage_members[stage])
        since_signup = (
            stage_dates.loc[member_index, stage] - stage_dates.loc[member_index, "Signup"]
            if member_index else pd.Series(dtype="timedelta64[ns]")
        )
        avg_since_signup, median_since_signup = _mean_median_days(
            since_signup.dt.days if len(since_signup) else pd.Series(dtype=float)
        )

        if stage == "Signup":
            transition_days = pd.Series(0.0, index=member_index, dtype=float)
        elif stage == "Churned":
            prior_dates = stage_dates.loc[member_index, [
                "Loyal", "Repeat Purchase", "First Purchase", "Signup",
            ]].bfill(axis=1).iloc[:, 0] if member_index else pd.Series(
                dtype="datetime64[ns]"
            )
            transition_days = (
                stage_dates.loc[member_index, stage] - prior_dates
                if member_index else pd.Series(dtype="timedelta64[ns]")
            )
            transition_days = transition_days.dt.days
        else:
            transition_days = (
                stage_dates.loc[member_index, stage]
                - stage_dates.loc[member_index, previous_stage]
                if member_index else pd.Series(dtype="timedelta64[ns]")
            )
            transition_days = transition_days.dt.days
        avg_transition, median_transition = _mean_median_days(transition_days)

        if previous_stage is None:
            dropoff_ids: set = set()
            dropoff_days = pd.Series(dtype=float)
        else:
            dropoff_ids = stage_members[previous_stage] - stage_members[stage]
            dropoff_index = list(dropoff_ids)
            previous_dates = stage_dates.loc[dropoff_index, previous_stage]
            dropoff_days = (
                stage_dates.loc[dropoff_index, "last_event_date"] - previous_dates
                if dropoff_index else pd.Series(dtype="timedelta64[ns]")
            )
            dropoff_days = (
                dropoff_days.dt.days if len(dropoff_days) else pd.Series(dtype=float)
            )
        avg_dropoff, median_dropoff = _mean_median_days(dropoff_days)

        rows.append(
            {
                "stage": stage,
                "count": count,
                "conversion_rate": round(conversion_rate, 4),
                "drop_off_rate": round(drop_off_rate, 4),
                "avg_days_since_signup": avg_since_signup,
                "median_days_since_signup": median_since_signup,
                "avg_days_from_previous_stage": avg_transition,
                "median_days_from_previous_stage": median_transition,
                "dropoff_customer_count": len(dropoff_ids),
                "avg_dropoff_days_after_previous_stage": avg_dropoff,
                "median_dropoff_days_after_previous_stage": median_dropoff,
            }
        )

    funnel = pd.DataFrame(rows)
    logger.info(
        "compute_journey_funnel: %d stages, signup=%d churned=%d",
        len(funnel),
        total,
        len(churned),
    )
    return funnel
