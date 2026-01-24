"""Tests for reading catalogs using DuckDB.

These tests verify that DuckDB can properly read Iceberg tables
created by the Faceberg catalog from HuggingFace datasets.

Note: DuckDB does not natively support the hf:// protocol for file reading.
Tests will be automatically skipped with informative messages when hf:// URIs
are encountered.
"""

import pytest


def duckdb_available():
    """Check if DuckDB is installed."""
    try:
        import duckdb

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not duckdb_available(), reason="DuckDB not installed (install with: pip install duckdb)"
)


@pytest.fixture
def duckdb_conn():
    """Create a DuckDB connection for testing."""
    import duckdb

    conn = duckdb.connect()
    yield conn
    conn.close()


@pytest.fixture
def imdb_metadata_path(synced_catalog_dir):
    """Return path to IMDB table metadata for DuckDB."""
    # DuckDB expects the metadata file path
    metadata_path = synced_catalog_dir / "default" / "imdb_plain_text" / "metadata"

    # Find the actual metadata file (v1.metadata.json, v2.metadata.json, etc.)
    metadata_files = sorted(metadata_path.glob("v*.metadata.json"))
    if metadata_files:
        # Return the latest version
        return str(metadata_files[-1])

    # Fallback - should not happen if catalog sync worked
    raise FileNotFoundError(f"No metadata files found in {metadata_path}")


@pytest.fixture
def rotten_tomatoes_metadata_path(synced_catalog_dir):
    """Return path to Rotten Tomatoes table metadata for DuckDB."""
    metadata_path = synced_catalog_dir / "default" / "rotten_tomatoes" / "metadata"

    # Find the actual metadata file
    metadata_files = sorted(metadata_path.glob("v*.metadata.json"))
    if metadata_files:
        # Return the latest version
        return str(metadata_files[-1])

    # Fallback - should not happen if catalog sync worked
    raise FileNotFoundError(f"No metadata files found in {metadata_path}")


# =============================================================================
# A. Basic Scanning Tests
# =============================================================================


def test_duckdb_iceberg_scan_basic(duckdb_conn, imdb_metadata_path):
    """Test basic DuckDB iceberg_scan functionality."""
    # Use iceberg_scan to read the table
    result = duckdb_conn.execute(
        f"""
        SELECT COUNT(*) as cnt
        FROM iceberg_scan('{imdb_metadata_path}')
    """
    ).fetchone()

    # Verify we got a count
    assert result is not None
    assert result[0] > 0



def test_duckdb_query_data(duckdb_conn, imdb_metadata_path):
    """Test querying data with WHERE clause."""

    # Query with WHERE clause on split column
    result = duckdb_conn.execute(
        f"""
        SELECT COUNT(*) as cnt, split
        FROM iceberg_scan('{imdb_metadata_path}')
        WHERE split = 'train'
        GROUP BY split
    """
    ).fetchall()

    # Verify we got results
    assert len(result) > 0
    assert result[0][1] == "train"  # Split column value
    assert result[0][0] > 0  # Count


def test_duckdb_aggregation(duckdb_conn, imdb_metadata_path):
    """Test aggregation queries (GROUP BY)."""

    # Run GROUP BY query on split column
    result = duckdb_conn.execute(
        f"""
        SELECT split, COUNT(*) as cnt
        FROM iceberg_scan('{imdb_metadata_path}')
        GROUP BY split
        ORDER BY split
    """
    ).fetchall()

    # Verify we got multiple splits
    assert len(result) > 0

    # Verify each split has a count
    for row in result:
        split_name, count = row
        assert split_name in ["train", "test", "unsupervised"]
        assert count > 0


# =============================================================================
# B. Schema and Metadata Tests
# =============================================================================


def test_duckdb_read_schema(duckdb_conn, imdb_metadata_path):
    """Test reading table schema via DuckDB."""

    # Use DESCRIBE to get schema information
    result = duckdb_conn.execute(
        f"""
        DESCRIBE SELECT * FROM iceberg_scan('{imdb_metadata_path}') LIMIT 0
    """
    ).fetchall()

    # Verify we got column information
    assert len(result) > 0

    # Extract column names
    column_names = [row[0] for row in result]

    # Verify expected columns
    assert "split" in column_names
    assert "text" in column_names
    assert "label" in column_names


def test_duckdb_table_info(duckdb_conn, rotten_tomatoes_metadata_path):
    """Test reading basic table information."""

    # Query to verify table can be opened and scanned
    result = duckdb_conn.execute(
        f"""
        SELECT COUNT(*)
        FROM iceberg_scan('{rotten_tomatoes_metadata_path}')
    """
    ).fetchone()

    # Verify we can read the table
    assert result is not None
    assert result[0] >= 0  # May be 0 if no data, but should execute


# =============================================================================
# C. Partition Pruning Tests
# =============================================================================


def test_duckdb_partition_filter(duckdb_conn, imdb_metadata_path):
    """Test partition pruning with WHERE clause."""

    # Query with and without filter to compare
    total_count = duckdb_conn.execute(
        f"""
        SELECT COUNT(*)
        FROM iceberg_scan('{imdb_metadata_path}')
    """
    ).fetchone()[0]

    train_count = duckdb_conn.execute(
        f"""
        SELECT COUNT(*)
        FROM iceberg_scan('{imdb_metadata_path}')
        WHERE split = 'train'
    """
    ).fetchone()[0]

    # Verify partition pruning occurred (train < total)
    assert train_count > 0
    assert train_count < total_count


@pytest.mark.skip(reason="PyIceberg FileIO does not support hf:// protocol for comparison")
def test_duckdb_partition_comparison(catalog, duckdb_conn, imdb_metadata_path):
    """Test that DuckDB partition filtering matches PyIceberg.

    Note: Skipped because PyIceberg doesn't support hf:// protocol, so we can't
    compare the results. DuckDB reads the Iceberg metadata correctly, but can't
    read the actual data files from HuggingFace.
    """
    # Get count from DuckDB
    duckdb_count = duckdb_conn.execute(
        f"""
        SELECT COUNT(*)
        FROM iceberg_scan('{imdb_metadata_path}')
        WHERE split = 'train'
    """
    ).fetchone()[0]

    # Get count from PyIceberg
    table = catalog.load_table("default.imdb_plain_text")
    scan = table.scan().filter("split = 'train'")
    arrow_table = scan.to_arrow()
    pyiceberg_count = arrow_table.num_rows

    # Verify counts match
    assert duckdb_count == pyiceberg_count