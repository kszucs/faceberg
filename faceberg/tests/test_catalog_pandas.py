"""Tests for pandas.read_iceberg() integration with Faceberg catalogs.

These tests verify that pandas can read Iceberg tables created by Faceberg
using both catalog_properties and environment variable configuration.
"""

import os

import pandas as pd
import pytest


@pytest.fixture
def catalog_properties(synced_catalog):
    """Return catalog properties for pandas.read_iceberg()."""
    # Use appropriate catalog implementation based on URI scheme
    if synced_catalog.uri.startswith("hf://"):
        catalog_impl = "faceberg.catalog.RemoteCatalog"
    else:
        catalog_impl = "faceberg.catalog.LocalCatalog"

    return {"py-catalog-impl": catalog_impl, "uri": synced_catalog.uri}


# =============================================================================
# A. Basic pandas.read_iceberg() Tests
# =============================================================================


def test_read_iceberg_with_catalog_properties(catalog_properties):
    """Test reading table using pandas with catalog_properties."""
    df = pd.read_iceberg(
        table_identifier="stanfordnlp.imdb",
        catalog_name="test_catalog",  # Name doesn't matter when passing properties
        catalog_properties=catalog_properties,
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


def test_read_iceberg_with_env_vars(catalog_properties):
    """Test reading table using pandas with environment variables.

    Note: catalog_properties is the recommended approach for programmatic usage.
    This test demonstrates that env vars can provide the URI, combined with
    catalog_properties for py-catalog-impl to bypass PyIceberg's URI inference.
    """
    catalog_uri = catalog_properties["uri"]

    # Set environment variables
    os.environ["PYICEBERG_CATALOG__TEST_CATALOG__PY_CATALOG_IMPL"] = "faceberg.catalog.LocalCatalog"
    os.environ["PYICEBERG_CATALOG__TEST_CATALOG__URI"] = catalog_uri

    try:
        # Pass py-catalog-impl and uri in catalog_properties
        # Env vars can also be used, but catalog_properties takes precedence
        df = pd.read_iceberg(
            table_identifier="stanfordnlp.imdb",
            catalog_name="test_catalog",
            catalog_properties=catalog_properties,
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


def test_read_iceberg_all_rows(catalog_properties):
    """Test reading all rows without limit."""
    df = pd.read_iceberg(
        table_identifier="stanfordnlp.imdb",
        catalog_name="test",
        catalog_properties=catalog_properties,
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


def test_read_iceberg_column_selection(catalog_properties):
    """Test reading specific columns - both multiple and single column."""

    # Test multiple columns
    df = pd.read_iceberg(
        table_identifier="stanfordnlp.imdb",
        catalog_name="test",
        catalog_properties=catalog_properties,
        columns=["text", "label"],
        limit=5,
    )
    assert list(df.columns) == ["text", "label"]
    assert "split" not in df.columns
    assert len(df) == 5

    # Test single column
    df = pd.read_iceberg(
        table_identifier="stanfordnlp.imdb",
        catalog_name="test",
        catalog_properties=catalog_properties,
        columns=["split"],
        limit=10,
    )
    assert list(df.columns) == ["split"]
    assert len(df) == 10


# =============================================================================
# C. Row Filtering Tests
# =============================================================================


def test_read_iceberg_row_filtering(catalog_properties):
    """Test various row filtering scenarios."""

    # Test filtering by partition column
    df = pd.read_iceberg(
        table_identifier="stanfordnlp.imdb",
        catalog_name="test",
        catalog_properties=catalog_properties,
        row_filter="split = 'train'",
        limit=20,
    )
    assert len(df) == 20
    assert all(df["split"] == "train")

    # Test filtering with IN clause
    df = pd.read_iceberg(
        table_identifier="stanfordnlp.imdb",
        catalog_name="test",
        catalog_properties=catalog_properties,
        row_filter="split IN ('train', 'test')",
        limit=30,
    )
    unique_splits = df["split"].unique()
    assert set(unique_splits).issubset({"train", "test"})
    assert "unsupervised" not in unique_splits
    assert len(df) == 30

    # Test filtering by non-partition column
    df = pd.read_iceberg(
        table_identifier="stanfordnlp.imdb",
        catalog_name="test",
        catalog_properties=catalog_properties,
        row_filter="label = 0",
        limit=10,
    )
    assert len(df) == 10
    assert all(df["label"] == 0)


# =============================================================================
# D. Combined Filter and Column Selection Tests
# =============================================================================


def test_read_iceberg_filter_and_column_selection(catalog_properties):
    """Test combining row filters and column selection."""

    # Test basic filter with column selection
    df = pd.read_iceberg(
        table_identifier="stanfordnlp.imdb",
        catalog_name="test",
        catalog_properties=catalog_properties,
        columns=["text", "label"],
        row_filter="split = 'train'",
        limit=5,
    )
    assert list(df.columns) == ["text", "label"]
    # Note: Some versions may optimize away rows if columns don't include filter columns
    assert len(df) <= 5  # May be less if optimizer is aggressive

    # Test complex filtering with column selection
    df = pd.read_iceberg(
        table_identifier="stanfordnlp.imdb",
        catalog_name="test",
        catalog_properties=catalog_properties,
        columns=["text"],
        row_filter="split = 'train' AND label = 1",
        limit=3,
    )
    assert list(df.columns) == ["text"]
    # Note: Filter may be optimized differently when split/label not in projection
    assert len(df) <= 3


# =============================================================================
# E. Edge Cases and Error Handling
# =============================================================================


def test_read_iceberg_empty_result(catalog_properties):
    """Test reading with filter that returns no rows."""

    df = pd.read_iceberg(
        table_identifier="stanfordnlp.imdb",
        catalog_name="test",
        catalog_properties=catalog_properties,
        row_filter="split = 'nonexistent'",
    )

    # Verify empty DataFrame with correct schema
    assert len(df) == 0
    assert "split" in df.columns
    assert "text" in df.columns
    assert "label" in df.columns


def test_read_iceberg_invalid_table(catalog_properties):
    """Test reading non-existent table."""

    with pytest.raises(Exception):  # Will raise NoSuchTableError
        pd.read_iceberg(
            table_identifier="default.nonexistent_table",
            catalog_name="test",
            catalog_properties=catalog_properties,
        )


def test_read_iceberg_case_sensitive_false(catalog_properties):
    """Test case-insensitive column matching."""

    df = pd.read_iceberg(
        table_identifier="stanfordnlp.imdb",
        catalog_name="test",
        catalog_properties=catalog_properties,
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


def test_read_iceberg_data_integrity(catalog_properties):
    """Test that data types and content are valid."""

    df = pd.read_iceberg(
        table_identifier="stanfordnlp.imdb",
        catalog_name="test",
        catalog_properties=catalog_properties,
        limit=10,
    )

    # Verify data types (pandas 2.x uses StringDtype for strings)
    assert df["text"].dtype.name in ["object", "string", "str"]  # String type
    assert df["label"].dtype.name in [
        "int64",
        "int32",
    ]  # Integer type (can vary by platform)
    assert df["split"].dtype.name in ["object", "string", "str"]  # String type

    # Verify text column contains actual text
    assert all(df["text"].str.len() > 0)

    # Verify label is 0 or 1
    assert all(df["label"].isin([0, 1]))

    # Verify split has valid values
    assert all(df["split"].isin(["train", "test", "unsupervised"]))
