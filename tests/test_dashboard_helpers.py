"""
TDD Tests for Dashboard Helpers Module.

Tests cover:
- Configuration loading and validation
- Color/theme utility functions
- KPI formatting helpers
- Risk level classification
- Data validation utilities
- Chart configuration helpers
- Sidebar info rendering helpers
- Page routing validation
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Load simulator configuration from YAML."""
    import yaml
    config_path = PROJECT_ROOT / "config" / "simulator_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_predictions():
    """Create sample prediction DataFrame."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "churn_probability": np.random.beta(2, 5, n),
        "risk_level": np.random.choice(
            ["low", "medium", "high", "critical"], n,
            p=[0.4, 0.3, 0.2, 0.1],
        ),
        "segment": np.random.choice(
            ["vip_loyal", "regular_loyal", "bargain_hunter",
             "explorer", "dormant", "new_customer"], n,
        ),
        "clv_predicted": np.random.lognormal(11, 1, n),
    })


# ---------------------------------------------------------------------------
# Format helpers tests
# ---------------------------------------------------------------------------

class TestFormatHelpers:
    """Test formatting utility functions."""

    def test_format_currency_krw(self):
        """Must format KRW currency with commas."""
        from src.dashboard.utils.dashboard_helpers import format_currency
        result = format_currency(50000000, "KRW")
        assert "50,000,000" in result
        assert "KRW" in result

    def test_format_currency_zero(self):
        """Must handle zero values."""
        from src.dashboard.utils.dashboard_helpers import format_currency
        result = format_currency(0, "KRW")
        assert "0" in result

    def test_format_percentage(self):
        """Must format decimal as percentage."""
        from src.dashboard.utils.dashboard_helpers import format_percentage
        result = format_percentage(0.1234)
        assert "12.34%" == result

    def test_format_percentage_zero(self):
        """Must handle zero."""
        from src.dashboard.utils.dashboard_helpers import format_percentage
        result = format_percentage(0.0)
        assert "0.00%" == result

    def test_format_count(self):
        """Must format integer counts with commas."""
        from src.dashboard.utils.dashboard_helpers import format_count
        result = format_count(1234567)
        assert result == "1,234,567"


# ---------------------------------------------------------------------------
# Risk classification tests
# ---------------------------------------------------------------------------

class TestRiskClassification:
    """Test risk level classification helpers."""

    def test_classify_risk_low(self):
        """Probability <= 0.25 should be low risk."""
        from src.dashboard.utils.dashboard_helpers import classify_risk
        assert classify_risk(0.1) == "low"
        assert classify_risk(0.25) == "low"

    def test_classify_risk_medium(self):
        """Probability 0.25-0.5 should be medium risk."""
        from src.dashboard.utils.dashboard_helpers import classify_risk
        assert classify_risk(0.3) == "medium"
        assert classify_risk(0.5) == "medium"

    def test_classify_risk_high(self):
        """Probability 0.5-0.75 should be high risk."""
        from src.dashboard.utils.dashboard_helpers import classify_risk
        assert classify_risk(0.6) == "high"
        assert classify_risk(0.75) == "high"

    def test_classify_risk_critical(self):
        """Probability > 0.75 should be critical."""
        from src.dashboard.utils.dashboard_helpers import classify_risk
        assert classify_risk(0.8) == "critical"
        assert classify_risk(1.0) == "critical"

    def test_classify_risk_custom_thresholds(self):
        """Must support custom thresholds."""
        from src.dashboard.utils.dashboard_helpers import classify_risk
        result = classify_risk(0.15, thresholds=(0.1, 0.2, 0.3))
        assert result == "medium"

    def test_get_risk_color(self):
        """Must map risk level to appropriate color."""
        from src.dashboard.utils.dashboard_helpers import get_risk_color
        assert get_risk_color("low") == "#2ecc71"
        assert get_risk_color("critical") == "#e74c3c"


# ---------------------------------------------------------------------------
# Data validation tests
# ---------------------------------------------------------------------------

class TestDataValidation:
    """Test data validation utility functions."""

    def test_validate_predictions_valid(self, sample_predictions):
        """Valid predictions should pass validation."""
        from src.dashboard.utils.dashboard_helpers import validate_predictions
        is_valid, errors = validate_predictions(sample_predictions)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_predictions_missing_column(self):
        """Missing required columns should fail validation."""
        from src.dashboard.utils.dashboard_helpers import validate_predictions
        df = pd.DataFrame({"customer_id": ["C00001"]})
        is_valid, errors = validate_predictions(df)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_predictions_empty(self):
        """Empty DataFrame should fail validation."""
        from src.dashboard.utils.dashboard_helpers import validate_predictions
        df = pd.DataFrame()
        is_valid, errors = validate_predictions(df)
        assert is_valid is False

    def test_safe_get_column(self, sample_predictions):
        """Must safely get column with default."""
        from src.dashboard.utils.dashboard_helpers import safe_get_column
        result = safe_get_column(sample_predictions, "churn_probability")
        assert len(result) == len(sample_predictions)

        # Missing column returns default series
        result2 = safe_get_column(sample_predictions, "nonexistent", default=0)
        assert (result2 == 0).all()


# ---------------------------------------------------------------------------
# Chart configuration tests
# ---------------------------------------------------------------------------

class TestChartHelpers:
    """Test chart configuration helpers."""

    def test_get_color_palette(self):
        """Must return a list of colors."""
        from src.dashboard.utils.dashboard_helpers import get_color_palette
        colors = get_color_palette()
        assert isinstance(colors, list)
        assert len(colors) >= 6

    def test_get_segment_colors(self, config):
        """Must return colors for each segment from config."""
        from src.dashboard.utils.dashboard_helpers import get_segment_colors
        colors = get_segment_colors(config)
        assert isinstance(colors, dict)
        assert len(colors) >= 6

    def test_create_kpi_delta(self):
        """Must compute KPI delta for comparison."""
        from src.dashboard.utils.dashboard_helpers import compute_kpi_delta
        delta = compute_kpi_delta(current=100, previous=80)
        assert delta == 25.0  # 25% increase

        delta2 = compute_kpi_delta(current=80, previous=100)
        assert delta2 == -20.0  # 20% decrease


# ---------------------------------------------------------------------------
# Page routing tests
# ---------------------------------------------------------------------------

class TestPageRouting:
    """Test page routing and navigation utilities."""

    def test_get_page_list(self):
        """Must return the full set of dashboard pages."""
        from src.dashboard.utils.dashboard_helpers import get_page_list
        pages = get_page_list()
        # 시연(Demo) + Overview + 14 analytical pages + System Health = 17
        assert len(pages) == 17
        assert "Demo" in pages
        assert pages[0] == "Demo"  # Demo must be the first page
        assert "Overview" in pages
        assert "MLflow Experiments" in pages
        assert "CLV & Retention Campaign" in pages

    def test_get_page_icon(self):
        """Must return icon for each page."""
        from src.dashboard.utils.dashboard_helpers import get_page_icon
        icon = get_page_icon("Overview")
        assert isinstance(icon, str)
        assert len(icon) > 0

    def test_all_pages_have_icons(self):
        """Every page must have an assigned icon."""
        from src.dashboard.utils.dashboard_helpers import (
            get_page_list, get_page_icon,
        )
        for page in get_page_list():
            icon = get_page_icon(page)
            assert icon is not None, f"Missing icon for page: {page}"


# ---------------------------------------------------------------------------
# Config helpers tests
# ---------------------------------------------------------------------------

class TestConfigHelpers:
    """Test configuration helper functions."""

    def test_get_churn_definition(self, config):
        """Must extract churn definition from config."""
        from src.dashboard.utils.dashboard_helpers import get_churn_definition
        churn_def = get_churn_definition(config)
        assert churn_def["no_purchase_days"] == 30
        assert churn_def["no_login_days"] == 60
        assert churn_def["operator"] == "OR"

    def test_get_ensemble_weights(self, config):
        """Must extract ensemble weights from config."""
        from src.dashboard.utils.dashboard_helpers import get_ensemble_weights
        ml_w, dl_w = get_ensemble_weights(config)
        assert ml_w == 0.6
        assert dl_w == 0.4

    def test_get_budget_config(self, config):
        """Must extract budget configuration."""
        from src.dashboard.utils.dashboard_helpers import get_budget_config
        budget = get_budget_config(config)
        assert budget["total_krw"] == 50000000
        assert budget["currency"] == "KRW"


# ---------------------------------------------------------------------------
# Sidebar helpers tests
# ---------------------------------------------------------------------------

class TestSidebarHelpers:
    """Test sidebar rendering helpers."""

    def test_build_sidebar_info(self, config):
        """Must build sidebar info dictionary."""
        from src.dashboard.utils.dashboard_helpers import build_sidebar_info
        info = build_sidebar_info(config)
        assert "churn_definition" in info
        assert "budget" in info
        assert "ensemble_weights" in info

    def test_get_app_title(self):
        """Must return app title string."""
        from src.dashboard.utils.dashboard_helpers import get_app_title
        title = get_app_title()
        assert isinstance(title, str)
        assert len(title) > 0
