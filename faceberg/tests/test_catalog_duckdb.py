"""Tests for reading catalogs using DuckDB.

These tests verify that DuckDB can properly read Iceberg tables
created by the Faceberg catalog from HuggingFace datasets.

Note: DuckDB supports the hf:// protocol through the httpfs extension. The tests
automatically load the httpfs and iceberg extensions to enable reading Iceberg
tables with hf:// URIs in their manifest files.

Limitation: DuckDB's httpfs extension requires hf:// URLs in the format
hf://datasets/{org}/{dataset}/{file}. Datasets must have an organization/user
prefix (e.g., stanfordnlp/imdb or glue/mrpc work, but rotten_tomatoes fails).
"""

import pytest
import duckdb


@pytest.fixture
def duckdb_conn():
    """Create a DuckDB connection for testing with required extensions."""
    conn = duckdb.connect()

    # Load httpfs extension for hf:// protocol support
    try:
        conn.execute("INSTALL httpfs")
        conn.execute("LOAD httpfs")
    except Exception as e:
        pytest.skip(f"Could not load httpfs extension: {e}")

    # Load iceberg extension for iceberg_scan support
    try:
        conn.execute("INSTALL iceberg")
        conn.execute("LOAD iceberg")
    except Exception as e:
        pytest.skip(f"Could not load iceberg extension: {e}")

    yield conn
    conn.close()


@pytest.fixture
def imdb_metadata_path(synced_catalog):
    """Return path to IMDB table metadata for DuckDB."""
    # DuckDB expects the metadata file path
    # synced_catalog fixture ensures catalog.sync() has been called
    catalog_location = synced_catalog.catalog_dir
    metadata_path = catalog_location / "default" / "imdb_plain_text" / "metadata"

    # Find the actual metadata file (v1.metadata.json, v2.metadata.json, etc.)
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


def test_duckdb_table_info(duckdb_conn, imdb_metadata_path):
    """Test reading basic table information.

    Uses stanfordnlp/imdb which has an org prefix compatible with DuckDB's
    httpfs hf:// URL format requirements.
    """

    # Query to verify table can be opened and scanned
    result = duckdb_conn.execute(
        f"""
        SELECT COUNT(*)
        FROM iceberg_scan('{imdb_metadata_path}')
    """
    ).fetchone()

    # Verify we can read the table
    assert result is not None
    assert result[0] > 0  # IMDB dataset has data


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


def test_duckdb_partition_comparison(catalog, duckdb_conn, imdb_metadata_path):
    """Test that DuckDB partition filtering matches PyIceberg.

    Both DuckDB (via httpfs extension) and PyIceberg (via HfFileIO) can read
    hf:// URIs, allowing direct comparison of query results.
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