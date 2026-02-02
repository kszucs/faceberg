"""Tests for reading catalogs using DuckDB.

These tests verify that DuckDB can properly read Iceberg tables
created by the Faceberg catalog from HuggingFace datasets.

DuckDB supports both file:// and hf:// URIs:
- file:// URIs work with local catalogs
- hf:// URIs work through the httpfs extension for remote catalogs

The tests automatically load the httpfs and iceberg extensions to enable
reading Iceberg tables with both local and remote storage.

Note: DuckDB's httpfs extension requires hf:// URLs in the format
hf://datasets/{org}/{dataset}/{file}. Datasets must have an organization/user
prefix (e.g., google-research-datasets/mbpp or glue/mrpc work, but rotten_tomatoes fails).
"""

import duckdb
import pytest


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
def mbpp_metadata_path(session_mbpp):
    """Return path to MBPP table metadata for DuckDB.

    DuckDB supports both file:// and hf:// URIs through httpfs extension.
    """
    # Construct path to v1.metadata.json directly from catalog URI
    return f"{session_mbpp.uri}/google-research-datasets/mbpp/metadata/v1.metadata.json"


# =============================================================================
# A. Basic Scanning Tests
# =============================================================================


def test_duckdb_iceberg_scan_basic(duckdb_conn, mbpp_metadata_path):
    """Test basic DuckDB iceberg_scan functionality."""
    # Use iceberg_scan to read the table
    result = duckdb_conn.execute(
        f"""
        SELECT COUNT(*) as cnt
        FROM iceberg_scan('{mbpp_metadata_path}')
    """
    ).fetchone()

    # Verify we got a count
    assert result is not None
    assert result[0] > 0


def test_duckdb_query_data(duckdb_conn, mbpp_metadata_path):
    """Test querying data with WHERE clause."""

    # Query with WHERE clause on split column
    result = duckdb_conn.execute(
        f"""
        SELECT COUNT(*) as cnt, split
        FROM iceberg_scan('{mbpp_metadata_path}')
        WHERE split = 'train'
        GROUP BY split
    """
    ).fetchall()

    # Verify we got results
    assert len(result) > 0
    assert result[0][1] == "train"  # Split column value
    assert result[0][0] > 0  # Count


def test_duckdb_aggregation(duckdb_conn, mbpp_metadata_path):
    """Test aggregation queries (GROUP BY)."""

    # Run GROUP BY query on split column
    result = duckdb_conn.execute(
        f"""
        SELECT split, COUNT(*) as cnt
        FROM iceberg_scan('{mbpp_metadata_path}')
        GROUP BY split
        ORDER BY split
    """
    ).fetchall()

    # Verify we got multiple splits
    assert len(result) > 0

    # Verify each split has a count
    for row in result:
        split_name, count = row
        assert split_name in ["train", "test", "validation", "prompt"]
        assert count > 0


# =============================================================================
# B. Schema and Metadata Tests
# =============================================================================


def test_duckdb_read_schema(duckdb_conn, mbpp_metadata_path):
    """Test reading table schema via DuckDB."""

    # Use DESCRIBE to get schema information
    result = duckdb_conn.execute(
        f"""
        DESCRIBE SELECT * FROM iceberg_scan('{mbpp_metadata_path}') LIMIT 0
    """
    ).fetchall()

    # Verify we got column information
    assert len(result) > 0

    # Extract column names
    column_names = [row[0] for row in result]

    # Verify expected columns
    assert "split" in column_names
    assert "prompt" in column_names
    assert "code" in column_names


def test_duckdb_table_info(duckdb_conn, mbpp_metadata_path):
    """Test reading basic table information.

    Uses google-research-datasets/mbpp which has an org prefix compatible with DuckDB's
    httpfs hf:// URL format requirements.
    """

    # Query to verify table can be opened and scanned
    result = duckdb_conn.execute(
        f"""
        SELECT COUNT(*)
        FROM iceberg_scan('{mbpp_metadata_path}')
    """
    ).fetchone()

    # Verify we can read the table
    assert result is not None
    assert result[0] > 0  # IMDB dataset has data


# =============================================================================
# C. Partition Pruning Tests
# =============================================================================


def test_duckdb_partition_filter(duckdb_conn, mbpp_metadata_path):
    """Test partition pruning with WHERE clause."""

    # Query with and without filter to compare
    total_count = duckdb_conn.execute(
        f"""
        SELECT COUNT(*)
        FROM iceberg_scan('{mbpp_metadata_path}')
    """
    ).fetchone()[0]

    train_count = duckdb_conn.execute(
        f"""
        SELECT COUNT(*)
        FROM iceberg_scan('{mbpp_metadata_path}')
        WHERE split = 'train'
    """
    ).fetchone()[0]

    # Verify partition pruning occurred (train < total)
    assert train_count > 0
    assert train_count < total_count


def test_duckdb_partition_comparison(session_mbpp, duckdb_conn, mbpp_metadata_path):
    """Test that DuckDB partition filtering matches PyIceberg.

    Both DuckDB (via httpfs extension) and PyIceberg (via HfFileIO) can read
    hf:// URIs, allowing direct comparison of query results.
    """
    # Get count from DuckDB
    duckdb_count = duckdb_conn.execute(
        f"""
        SELECT COUNT(*)
        FROM iceberg_scan('{mbpp_metadata_path}')
        WHERE split = 'train'
    """
    ).fetchone()[0]

    # Get count from PyIceberg
    table = session_mbpp.load_table("google-research-datasets.mbpp")
    scan = table.scan().filter("split = 'train'")
    arrow_table = scan.to_arrow()
    pyiceberg_count = arrow_table.num_rows

    # Verify counts match
    assert duckdb_count == pyiceberg_count


# =============================================================================
# D. REST Catalog Tests
# =============================================================================


@pytest.fixture(scope="session")
def duckdb_rest_conn(session_rest_server):
    """Create a DuckDB connection configured to use REST catalog.

    Note: DuckDB REST catalog support is still evolving. As of DuckDB 1.4.3,
    the REST catalog configuration may not be fully supported. These tests
    are marked as expected to fail until DuckDB adds stable REST catalog support.
    """
    conn = duckdb.connect()

    # Load required extensions
    try:
        conn.execute("INSTALL httpfs")
        conn.execute("LOAD httpfs")
        conn.execute("INSTALL iceberg")
        conn.execute("LOAD iceberg")
    except Exception as e:
        pytest.skip(f"Could not load required extensions: {e}")

    # Attach REST catalog
    # Note: DuckDB REST catalog support requires specifying ENDPOINT in ATTACH
    # AUTHORIZATION_TYPE 'none' disables authentication for local test server
    conn.execute(f"""
        ATTACH 'warehouse' AS iceberg_catalog (
            TYPE ICEBERG,
            ENDPOINT '{session_rest_server}',
            AUTHORIZATION_TYPE 'none'
        )
    """)

    yield conn
    conn.close()


def test_duckdb_rest_list_tables(duckdb_rest_conn):
    """Test listing tables via REST catalog in DuckDB."""
    # List tables in the google-research-datasets namespace using SHOW TABLES
    result = duckdb_rest_conn.execute("""
        SHOW TABLES FROM "iceberg_catalog"."google-research-datasets"
    """).fetchall()

    # Verify we can list tables and mbpp is present
    assert len(result) > 0
    table_names = [row[0] for row in result]
    assert "mbpp" in table_names


def test_duckdb_rest_query_data(duckdb_rest_conn):
    """Test querying data via REST catalog in DuckDB."""
    # Query with WHERE clause
    result = duckdb_rest_conn.execute("""
        SELECT COUNT(*) as cnt, split
        FROM iceberg_catalog."google-research-datasets".mbpp
        WHERE split = 'train'
        GROUP BY split
    """).fetchall()

    # Verify we got results
    assert len(result) > 0
    assert result[0][1] == "train"
    assert result[0][0] > 0


def test_duckdb_rest_aggregation(duckdb_rest_conn):
    """Test aggregation queries via REST catalog."""
    # Run GROUP BY query
    result = duckdb_rest_conn.execute("""
        SELECT split, COUNT(*) as cnt
        FROM iceberg_catalog."google-research-datasets".mbpp
        GROUP BY split
        ORDER BY split
    """).fetchall()

    # Verify we got multiple splits
    assert len(result) > 0

    # Verify each split has a count
    for row in result:
        split_name, count = row
        assert split_name in ["train", "test", "validation", "prompt"]
        assert count > 0


def test_duckdb_rest_schema(duckdb_rest_conn):
    """Test reading schema via REST catalog in DuckDB."""
    # Use DESCRIBE to get schema
    result = duckdb_rest_conn.execute("""
        DESCRIBE SELECT * FROM iceberg_catalog."google-research-datasets".mbpp LIMIT 0
    """).fetchall()

    # Verify we got column information
    assert len(result) > 0

    # Extract column names
    column_names = [row[0] for row in result]

    # Verify expected columns
    assert "split" in column_names
    assert "prompt" in column_names
    assert "code" in column_names


def test_duckdb_rest_partition_filter(duckdb_rest_conn):
    """Test partition filtering via REST catalog."""
    # Query with and without filter
    total_count = duckdb_rest_conn.execute("""
        SELECT COUNT(*)
        FROM iceberg_catalog."google-research-datasets".mbpp
    """).fetchone()[0]

    train_count = duckdb_rest_conn.execute("""
        SELECT COUNT(*)
        FROM iceberg_catalog."google-research-datasets".mbpp
        WHERE split = 'train'
    """).fetchone()[0]

    # Verify partition pruning
    assert train_count > 0
    assert train_count < total_count
