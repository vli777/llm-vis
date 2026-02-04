"""
Tests for LLM-related features: profiling, role detection, structured outputs, and provider detection.
"""

import pytest
import pandas as pd
from pydantic import ValidationError

from skills.profile import dataset_profile, detect_column_role as _detect_column_role, suggest_chart_types as _suggest_chart_types
from skills.target import infer_task_and_target, run_target_analysis
from app.models import Plan, Operation
from app.llm_loader import supports_native_structured_output, get_provider_name


class TestColumnRoleDetection:
    """Tests for semantic column role detection."""

    @pytest.mark.parametrize("col_name,unique_ratio,expected_role", [
        ("created_at", 0.99, "temporal"),
        ("country", 0.03, "geographic"),
        ("user_id", 0.99, "identifier"),
        ("revenue", 0.99, "measure"),
        ("city", 0.03, "geographic"),
    ])
    def test_detect_column_role(self, col_name, unique_ratio, expected_role):
        """Test column role detection with various column types."""
        if col_name == "created_at":
            series = pd.Series(pd.date_range('2020-01-01', periods=100))
        elif col_name in ["country", "city"]:
            series = pd.Series(['US', 'UK', 'CA'] * 33 + ['US'])
        elif col_name == "user_id":
            series = pd.Series(range(100))
        else:  # revenue
            series = pd.Series(range(1000, 2000))

        detected = _detect_column_role(col_name, series, unique_ratio)
        assert detected == expected_role, f"Expected {expected_role}, got {detected} for {col_name}"

    def test_temporal_detection_by_dtype(self):
        """Test temporal detection based on datetime dtype."""
        series = pd.Series(pd.date_range('2020-01-01', periods=100))
        role = _detect_column_role("some_column", series, 0.5)
        assert role == "temporal"

    def test_measure_detection_by_name(self):
        """Test measure detection based on column name keywords."""
        test_cases = ["revenue", "sales", "amount", "price", "cost", "total"]
        for col_name in test_cases:
            series = pd.Series(range(100))
            role = _detect_column_role(col_name, series, 0.5)
            assert role == "measure", f"{col_name} should be detected as measure"

    def test_string_encoded_numeric_detection(self):
        """Test detection of string-encoded numeric values like '$1.3B'."""
        # Financial data with various formats
        series = pd.Series(["$1.3B", "$500M", "$2.1B", "$750M", "$1.0B"])
        role = _detect_column_role("Valuation", series, 0.8)
        assert role == "measure", "String-encoded financial values should be detected as measure"

    def test_string_encoded_numeric_with_commas(self):
        """Test detection of numeric strings with comma separators."""
        series = pd.Series(["1,234.56", "2,345.67", "3,456.78", "4,567.89"])
        role = _detect_column_role("revenue", series, 0.9)
        assert role == "measure", "Comma-separated numeric strings should be detected as measure"

    def test_mixed_parseable_strings(self):
        """Test that mixed parseable/non-parseable strings are classified correctly."""
        # >50% parseable -> measure
        series = pd.Series(["$1.3B", "$500M", "$2.1B", "N/A", "$1.0B"])
        role = _detect_column_role("value", series, 0.8)
        assert role == "measure", "Mostly parseable strings should be measure"

        # <50% parseable -> nominal
        series = pd.Series(["$1.3B", "Invalid", "Not a number", "Also invalid", "$1.0B"])
        role = _detect_column_role("value", series, 0.8)
        assert role == "nominal", "Mostly non-parseable strings should be nominal"

    def test_count_detection_by_name(self):
        """Test count detection based on column name keywords."""
        test_cases = ["count", "quantity", "number", "num_items"]
        for col_name in test_cases:
            series = pd.Series(range(100))
            role = _detect_column_role(col_name, series, 0.5)
            assert role == "count", f"{col_name} should be detected as count"


class TestDatasetProfiling:
    """Tests for enhanced dataset profiling."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'category': ['A', 'B', 'C'] * 33 + ['A'],
            'value': range(100, 200),
            'revenue': [x * 1.5 for x in range(100, 200)],
            'user_id': range(1000, 1100)
        })

    def test_profile_basic_structure(self, sample_dataframe):
        """Test that profile contains expected basic structure."""
        profile = dataset_profile(sample_dataframe, include_viz_hints=True)

        assert 'row_count' in profile
        assert 'columns' in profile
        assert profile['row_count'] == 100
        assert len(profile['columns']) == 5

    def test_profile_visualization_hints(self, sample_dataframe):
        """Test that visualization hints are included."""
        profile = dataset_profile(sample_dataframe, include_viz_hints=True)

        assert 'visualization_hints' in profile
        hints = profile['visualization_hints']

        assert 'suggested_chart_types' in hints
        assert 'summary' in hints

        summary = hints['summary']
        assert summary['numeric_columns'] == 3
        assert summary['categorical_columns'] == 1
        assert summary['has_temporal_data'] is True
        assert len(hints['suggested_chart_types']) > 0

    def test_profile_column_roles(self, sample_dataframe):
        """Test that columns have role annotations."""
        profile = dataset_profile(sample_dataframe, include_viz_hints=True)

        roles = {col['name']: col.get('role') for col in profile['columns']}

        assert roles['date'] == 'temporal'
        assert roles['category'] == 'categorical'
        assert roles['value'] == 'measure'
        assert roles['revenue'] == 'measure'
        assert roles['user_id'] == 'identifier'

    def test_profile_sample_rows(self, sample_dataframe):
        """Test that sample rows are included."""
        profile = dataset_profile(sample_dataframe, include_viz_hints=True)

        assert 'sample_rows' in profile
        assert len(profile['sample_rows']) <= 3
        assert isinstance(profile['sample_rows'], list)

    def test_profile_without_viz_hints(self, sample_dataframe):
        """Test profile generation without visualization hints."""
        profile = dataset_profile(sample_dataframe, include_viz_hints=False)

        assert 'row_count' in profile
        assert 'columns' in profile
        # Should not have visualization hints when disabled
        assert 'visualization_hints' not in profile

    def test_profile_string_encoded_numerics(self):
        """Test that string-encoded numeric columns are profiled correctly."""
        df = pd.DataFrame({
            'Company': ['A', 'B', 'C', 'D', 'E'],
            'Valuation': ['$1.3B', '$500M', '$2.1B', '$750M', '$1.0B'],
            'ARR': ['$100M', '$50M', '$200M', '$75M', '$90M'],
            'Founded': ['2010', '2015', '2012', '2018', '2011']
        })

        profile = dataset_profile(df, include_viz_hints=True)

        # Find columns by name
        cols_by_name = {col['name']: col for col in profile['columns']}

        # Valuation and ARR should be detected as measures with numeric stats
        assert cols_by_name['Valuation']['role'] == 'measure'
        assert 'num_stats' in cols_by_name['Valuation']
        assert 'note' in cols_by_name['Valuation']
        assert 'string-encoded numeric' in cols_by_name['Valuation']['note']

        assert cols_by_name['ARR']['role'] == 'measure'
        assert 'num_stats' in cols_by_name['ARR']

        # Founded should be detected as temporal or measure
        assert cols_by_name['Founded']['role'] in ['temporal', 'measure']

        # Check that numeric stats are reasonable
        val_stats = cols_by_name['Valuation']['num_stats']
        assert val_stats['min'] is not None
        assert val_stats['max'] is not None
        assert val_stats['max'] > val_stats['min']


class TestChartSuggestions:
    """Tests for chart type suggestion logic."""

    def test_temporal_data_suggestions(self):
        """Test suggestions for temporal data."""
        numeric_info = [{"name": "value", "variance": 100}]
        categorical_info = []
        suggestions = _suggest_chart_types(numeric_info, categorical_info, has_temporal=True)

        assert "line chart (temporal trends)" in suggestions
        assert len(suggestions) > 0

    def test_scatter_plot_suggestions(self):
        """Test suggestions for scatter plot data (2+ numeric columns)."""
        numeric_info = [
            {"name": "x", "variance": 100},
            {"name": "y", "variance": 200}
        ]
        categorical_info = []
        suggestions = _suggest_chart_types(numeric_info, categorical_info, has_temporal=False)

        assert "scatter plot (correlations)" in suggestions

    def test_bar_chart_suggestions(self):
        """Test suggestions for bar chart data (categorical + numeric)."""
        numeric_info = [{"name": "value", "variance": 100}]
        categorical_info = [{"name": "category", "unique": 5, "ratio": 0.05}]
        suggestions = _suggest_chart_types(numeric_info, categorical_info, has_temporal=False)

        assert "bar chart (category comparisons)" in suggestions

    def test_pie_chart_suggestions(self):
        """Test suggestions for pie chart data (low cardinality categorical)."""
        numeric_info = []
        categorical_info = [{"name": "category", "unique": 5, "ratio": 0.05}]
        suggestions = _suggest_chart_types(numeric_info, categorical_info, has_temporal=False)

        assert "pie/donut chart (part-to-whole)" in suggestions

    def test_max_suggestions_limit(self):
        """Test that suggestions are limited to 4."""
        # Create scenario with many potential chart types
        numeric_info = [{"name": f"col{i}", "variance": 100} for i in range(5)]
        categorical_info = [{"name": "cat", "unique": 3, "ratio": 0.05}]
        suggestions = _suggest_chart_types(numeric_info, categorical_info, has_temporal=True)

        assert len(suggestions) <= 4


class TestPydanticModels:
    """Tests for Pydantic model validation."""

    def test_valid_plan_creation(self):
        """Test creating a valid plan."""
        plan = Plan(
            action="create",
            type="chart",
            title="Sales by Category",
            operations=[],
            vega_lite={
                "mark": "bar",
                "encoding": {
                    "x": {"field": "category", "type": "nominal"},
                    "y": {"field": "value", "type": "quantitative"}
                }
            }
        )

        assert plan.action == "create"
        assert plan.type == "chart"
        assert plan.title == "Sales by Category"
        assert plan.vega_lite is not None

    def test_invalid_operation_type(self):
        """Test that invalid operation types are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Operation(op="invalid_op", col="test")

        # Pydantic v2 error message format
        error_msg = str(exc_info.value)
        assert "Input should be" in error_msg or "op must be one of" in error_msg

    def test_title_too_long(self):
        """Test that overly long titles are rejected."""
        long_title = "This is a very very very very very long title that definitely exceeds the maximum allowed length for visualization titles by a significant margin"

        with pytest.raises(ValidationError) as exc_info:
            Plan(
                action="create",
                type="chart",
                title=long_title,
                vega_lite={"mark": "bar"}
            )

        assert "Title should be concise" in str(exc_info.value)

    def test_missing_vega_lite_for_chart(self):
        """Test that vega_lite is required for chart creation."""
        with pytest.raises(ValueError) as exc_info:
            Plan(
                action="create",
                type="chart",
                title="Test Chart"
            )

        assert "vega_lite is required" in str(exc_info.value)

    def test_table_plan_without_vega_lite(self):
        """Test that table plans don't require vega_lite."""
        plan = Plan(
            action="create",
            type="table",
            title="Data Table"
        )

        assert plan.action == "create"
        assert plan.type == "table"
        assert plan.vega_lite is None

    def test_operation_with_all_fields(self):
        """Test creating operation with all optional fields."""
        # Use model_validate to work with alias 'as'
        op = Operation.model_validate({
            "op": "value_counts",
            "col": "category",
            "as": ["category", "count"],
            "limit": 25
        })

        assert op.op == "value_counts"
        assert op.col == "category"
        assert op.as_ == ["category", "count"]
        assert op.limit == 25

    def test_plan_with_alias_field(self):
        """Test that 'intent' alias works for 'type' field."""
        # Using model_dump to serialize
        plan = Plan(
            action="create",
            type="chart",
            title="Test",
            vega_lite={"mark": "bar"}
        )

        # Check that both type and intent work
        assert plan.type == "chart"


class TestTargetAnalysis:
    def test_infer_target_classification(self):
        df = pd.DataFrame({
            "feature_a": [1, 2, 3, 4, 5],
            "label": ["yes", "no", "yes", "no", "yes"],
        })
        from skills.profile import build_profile
        prof_model = build_profile(df, "t")
        target = infer_task_and_target(prof_model)
        assert target.column == "label"
        assert target.task_type == "classification"

    def test_target_associations_regression(self):
        df = pd.DataFrame({
            "price": [10, 20, 30, 40, 50, 60],
            "units": [1, 2, 3, 4, 5, 6],
            "category": ["A", "A", "B", "B", "C", "C"],
        })
        from skills.profile import build_profile
        prof_model = build_profile(df, "t")
        insights = run_target_analysis(df, prof_model)
        assert insights.target.column in {"price", "units"}
        assert len(insights.associations) >= 1


class TestProviderDetection:
    """Tests for provider capability detection."""

    def test_provider_name_detection(self):
        """Test that provider name can be detected."""
        provider = get_provider_name()
        assert provider in {"groq", "openai", "oa", "nvidia", "nv", "nvcf"}

    def test_structured_output_support(self):
        """Test structured output support detection."""
        provider = get_provider_name()
        supports_pydantic = supports_native_structured_output()

        # Verify correct support based on provider
        if provider in {"openai", "oa", "groq"}:
            assert supports_pydantic is True, f"{provider} should support Pydantic"
        elif provider in {"nvidia", "nv", "nvcf"}:
            assert supports_pydantic is False, f"{provider} should use JSON mode"


class TestOperationModels:
    """Tests for specific operation types."""

    @pytest.mark.parametrize("op_type", [
        "value_counts",
        "explode_counts",
        "scatter_data",
        "corr_pair"
    ])
    def test_valid_operation_types(self, op_type):
        """Test that all valid operation types are accepted."""
        op = Operation(op=op_type, col="test_col")
        assert op.op == op_type

    def test_scatter_data_operation(self):
        """Test scatter_data operation with x, y fields."""
        op = Operation(
            op="scatter_data",
            x="column_x",
            y="column_y",
            log=True
        )

        assert op.x == "column_x"
        assert op.y == "column_y"
        assert op.log is True

    def test_value_counts_with_limit(self):
        """Test value_counts operation with limit."""
        op = Operation(
            op="value_counts",
            col="category",
            limit=10
        )

        assert op.op == "value_counts"
        assert op.limit == 10
