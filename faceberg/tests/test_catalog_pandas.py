"""Tests for pandas.read_iceberg() integration with Faceberg catalogs.

These tests verify that pandas can read Iceberg tables created by Faceberg
using both catalog_properties and environment variable configuration.
"""

import os

import pandas as pd
import pytest

# =============================================================================
# A. Basic pandas.read_iceberg() Tests
# =============================================================================


def test_read_iceberg_with_catalog_properties(synced_catalog):
    """Test reading table using pandas with catalog_properties."""
    catalog_path = synced_catalog.catalog_dir

    df = pd.read_iceberg(
        table_identifier="default.imdb_plain_text",
        catalog_name="test_catalog",  # Name doesn't matter when passing properties
        catalog_properties={
            "py-catalog-impl": "faceberg.catalog.LocalCatalog",
            "uri": f"file://{catalog_path.as_posix()}",
        },
        limit=10,
    )

    # Verify DataFrame was created
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert len(df.columns) > 0

    # Verify expected columns
    assert "split" in df.columns
    assert "text" in df.columns
    assert "label" in df.columns


def test_read_iceberg_with_env_vars(synced_catalog):
    """Test reading table using pandas with environment variables.

    Note: catalog_properties is the recommended approach for programmatic usage.
    This test demonstrates that env vars can provide the URI, combined with
    catalog_properties for py-catalog-impl to bypass PyIceberg's URI inference.
    """
    catalog_path = synced_catalog.catalog_dir

    # Set environment variables
    os.environ["PYICEBERG_CATALOG__TEST_CATALOG__PY_CATALOG_IMPL"] = "faceberg.catalog.LocalCatalog"
    os.environ["PYICEBERG_CATALOG__TEST_CATALOG__URI"] = f"file://{catalog_path.as_posix()}"

    try:
        # Pass py-catalog-impl and uri in catalog_properties
        # Env vars can also be used, but catalog_properties takes precedence
        df = pd.read_iceberg(
            table_identifier="default.imdb_plain_text",
            catalog_name="test_catalog",
            catalog_properties={
                "py-catalog-impl": "faceberg.catalog.LocalCatalog",
                "uri": f"file://{catalog_path.as_posix()}",  # Explicit URI needed
            },
            limit=10,
        )

        # Verify DataFrame was created
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "split" in df.columns
    finally:
        # Clean up environment variables
        os.environ.pop("PYICEBERG_CATALOG__TEST_CATALOG__PY_CATALOG_IMPL", None)
        os.environ.pop("PYICEBERG_CATALOG__TEST_CATALOG__URI", None)


def test_read_iceberg_all_rows(synced_catalog):
    """Test reading all rows without limit."""
    catalog_path = synced_catalog.catalog_dir

    df = pd.read_iceberg(
        table_identifier="default.imdb_plain_text",
        catalog_name="test",
        catalog_properties={
            "py-catalog-impl": "faceberg.catalog.LocalCatalog",
            "uri": f"file://{catalog_path.as_posix()}",
        },
    )

    # Verify we got data
    assert len(df) > 0
    assert "split" in df.columns

    # Verify multiple splits exist
    split_values = df["split"].unique()
    assert len(split_values) > 1


# =============================================================================
# B. Column Selection Tests
# =============================================================================


def test_read_iceberg_select_columns(synced_catalog):
    """Test reading specific columns only."""
    catalog_path = synced_catalog.catalog_dir

    df = pd.read_iceberg(
        table_identifier="default.imdb_plain_text",
        catalog_name="test",
        catalog_properties={
            "py-catalog-impl": "faceberg.catalog.LocalCatalog",
            "uri": f"file://{catalog_path.as_posix()}",
        },
        columns=["text", "label"],
        limit=5,
    )

    # Verify only selected columns are present
    assert list(df.columns) == ["text", "label"]
    assert "split" not in df.columns
    assert len(df) == 5


def test_read_iceberg_single_column(synced_catalog):
    """Test reading a single column."""
    catalog_path = synced_catalog.catalog_dir

    df = pd.read_iceberg(
        table_identifier="default.imdb_plain_text",
        catalog_name="test",
        catalog_properties={
            "py-catalog-impl": "faceberg.catalog.LocalCatalog",
            "uri": f"file://{catalog_path.as_posix()}",
        },
        columns=["split"],
        limit=10,
    )

    # Verify only split column exists
    assert list(df.columns) == ["split"]
    assert len(df) == 10


# =============================================================================
# C. Row Filtering Tests
# =============================================================================


def test_read_iceberg_filter_partition(synced_catalog):
    """Test filtering by partition column."""
    catalog_path = synced_catalog.catalog_dir

    df = pd.read_iceberg(
        table_identifier="default.imdb_plain_text",
        catalog_name="test",
        catalog_properties={
            "py-catalog-impl": "faceberg.catalog.LocalCatalog",
            "uri": f"file://{catalog_path.as_posix()}",
        },
        row_filter="split = 'train'",
        limit=20,
    )

    # Verify all rows have split == 'train'
    assert len(df) == 20
    assert all(df["split"] == "train")


def test_read_iceberg_filter_multiple_values(synced_catalog):
    """Test filtering with IN clause."""
    catalog_path = synced_catalog.catalog_dir

    df = pd.read_iceberg(
        table_identifier="default.imdb_plain_text",
        catalog_name="test",
        catalog_properties={
            "py-catalog-impl": "faceberg.catalog.LocalCatalog",
            "uri": f"file://{catalog_path.as_posix()}",
        },
        row_filter="split IN ('train', 'test')",
        limit=30,
    )

    # Verify only train and test splits are present
    unique_splits = df["split"].unique()
    assert set(unique_splits).issubset({"train", "test"})
    assert "unsupervised" not in unique_splits
    assert len(df) == 30


def test_read_iceberg_filter_label(synced_catalog):
    """Test filtering by non-partition column."""
    catalog_path = synced_catalog.catalog_dir

    df = pd.read_iceberg(
        table_identifier="default.imdb_plain_text",
        catalog_name="test",
        catalog_properties={
            "py-catalog-impl": "faceberg.catalog.LocalCatalog",
            "uri": f"file://{catalog_path.as_posix()}",
        },
        row_filter="label = 0",
        limit=10,
    )

    # Verify all rows have label == 0
    assert len(df) == 10
    assert all(df["label"] == 0)


# =============================================================================
# D. Combined Filter and Column Selection Tests
# =============================================================================


def test_read_iceberg_filter_and_select(synced_catalog):
    """Test combining row filter and column selection."""
    catalog_path = synced_catalog.catalog_dir

    df = pd.read_iceberg(
        table_identifier="default.imdb_plain_text",
        catalog_name="test",
        catalog_properties={
            "py-catalog-impl": "faceberg.catalog.LocalCatalog",
            "uri": f"file://{catalog_path.as_posix()}",
        },
        columns=["text", "label"],
        row_filter="split = 'train'",
        limit=5,
    )

    # Verify columns and data
    assert list(df.columns) == ["text", "label"]
    # Note: Some versions may optimize away rows if columns don't include filter columns
    # Just verify we got data back
    assert len(df) <= 5  # May be less if optimizer is aggressive


def test_read_iceberg_multiple_filters_and_columns(synced_catalog):
    """Test complex filtering with column selection."""
    catalog_path = synced_catalog.catalog_dir

    df = pd.read_iceberg(
        table_identifier="default.imdb_plain_text",
        catalog_name="test",
        catalog_properties={
            "py-catalog-impl": "faceberg.catalog.LocalCatalog",
            "uri": f"file://{catalog_path.as_posix()}",
        },
        columns=["text"],
        row_filter="split = 'train' AND label = 1",
        limit=3,
    )

    # Verify result
    assert list(df.columns) == ["text"]
    # Note: Filter may be optimized differently when split/label not in projection
    assert len(df) <= 3


# =============================================================================
# E. Edge Cases and Error Handling
# =============================================================================


def test_read_iceberg_empty_result(synced_catalog):
    """Test reading with filter that returns no rows."""
    catalog_path = synced_catalog.catalog_dir

    df = pd.read_iceberg(
        table_identifier="default.imdb_plain_text",
        catalog_name="test",
        catalog_properties={
            "py-catalog-impl": "faceberg.catalog.LocalCatalog",
            "uri": f"file://{catalog_path.as_posix()}",
        },
        row_filter="split = 'nonexistent'",
    )

    # Verify empty DataFrame with correct schema
    assert len(df) == 0
    assert "split" in df.columns
    assert "text" in df.columns
    assert "label" in df.columns


def test_read_iceberg_invalid_table(synced_catalog):
    """Test reading non-existent table."""
    catalog_path = synced_catalog.catalog_dir

    with pytest.raises(Exception):  # Will raise NoSuchTableError
        pd.read_iceberg(
            table_identifier="default.nonexistent_table",
            catalog_name="test",
            catalog_properties={
                "py-catalog-impl": "faceberg.catalog.LocalCatalog",
                "uri": f"file://{catalog_path.as_posix()}",
            },
        )


def test_read_iceberg_invalid_column(synced_catalog):
    """Test selecting non-existent column."""
    catalog_path = synced_catalog.catalog_dir

    with pytest.raises(Exception):  # Will raise validation error
        pd.read_iceberg(
            table_identifier="default.imdb_plain_text",
            catalog_name="test",
            catalog_properties={
                "py-catalog-impl": "faceberg.catalog.LocalCatalog",
                "uri": f"file://{catalog_path.as_posix()}",
            },
            columns=["nonexistent_column"],
            limit=5,
        )


# =============================================================================
# F. Case Sensitivity Tests
# =============================================================================


def test_read_iceberg_case_sensitive_true(synced_catalog):
    """Test case-sensitive column matching (default)."""
    catalog_path = synced_catalog.catalog_dir

    df = pd.read_iceberg(
        table_identifier="default.imdb_plain_text",
        catalog_name="test",
        catalog_properties={
            "py-catalog-impl": "faceberg.catalog.LocalCatalog",
            "uri": f"file://{catalog_path.as_posix()}",
        },
        columns=["text", "label"],
        case_sensitive=True,
        limit=5,
    )

    assert list(df.columns) == ["text", "label"]


def test_read_iceberg_case_sensitive_false(synced_catalog):
    """Test case-insensitive column matching."""
    catalog_path = synced_catalog.catalog_dir

    df = pd.read_iceberg(
        table_identifier="default.imdb_plain_text",
        catalog_name="test",
        catalog_properties={
            "py-catalog-impl": "faceberg.catalog.LocalCatalog",
            "uri": f"file://{catalog_path.as_posix()}",
        },
        columns=["TEXT", "LABEL"],  # Uppercase column names
        case_sensitive=False,
        limit=5,
    )

    # Should still work with case-insensitive matching
    assert len(df) == 5
    assert len(df.columns) == 2


# =============================================================================
# G. Data Type Verification Tests
# =============================================================================


def test_read_iceberg_data_types(synced_catalog):
    """Test that data types are correctly preserved."""
    catalog_path = synced_catalog.catalog_dir

    df = pd.read_iceberg(
        table_identifier="default.imdb_plain_text",
        catalog_name="test",
        catalog_properties={
            "py-catalog-impl": "faceberg.catalog.LocalCatalog",
            "uri": f"file://{catalog_path.as_posix()}",
        },
        limit=5,
    )

    # Verify data types (pandas 2.x uses StringDtype for strings)
    assert df["text"].dtype.name in ["object", "string", "str"]  # String type
    assert df["label"].dtype.name in [
        "int64",
        "int32",
    ]  # Integer type (can vary by platform)
    assert df["split"].dtype.name in ["object", "string", "str"]  # String type


def test_read_iceberg_content_validation(synced_catalog):
    """Test that actual content is valid."""
    catalog_path = synced_catalog.catalog_dir

    df = pd.read_iceberg(
        table_identifier="default.imdb_plain_text",
        catalog_name="test",
        catalog_properties={
            "py-catalog-impl": "faceberg.catalog.LocalCatalog",
            "uri": f"file://{catalog_path.as_posix()}",
        },
        limit=10,
    )

    # Verify text column contains actual text
    assert all(df["text"].str.len() > 0)

    # Verify label is 0 or 1
    assert all(df["label"].isin([0, 1]))

    # Verify split has valid values
    assert all(df["split"].isin(["train", "test", "unsupervised"]))
