"""
Demonstration: Multiple Iceberg tables with data in separate directories.

This script shows that Apache Iceberg allows you to:
- Store multiple tables' metadata in the same catalog location
- Store each table's data files in completely different directories

This is achieved using the 'write.data.path' table property per table.
"""

import os
import pyarrow as pa

from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import LongType, StringType, NestedField

# -------------------------------------------------------------------
# 1. Paths
# -------------------------------------------------------------------
CATALOG_DB = "/tmp/metadata/catalog.db"
METADATA_BASE = "/tmp/metadata/default"  # Shared metadata location for both tables
TABLE1_DATA = "file:///tmp/table1"  # Data location for taxi table
TABLE2_DATA = "file:///tmp/table2"  # Data location for weather table

os.makedirs("/tmp/metadata", exist_ok=True)
os.makedirs("/tmp/table1", exist_ok=True)
os.makedirs("/tmp/table2", exist_ok=True)

# -------------------------------------------------------------------
# 2. Load SQL catalog
# -------------------------------------------------------------------
catalog = load_catalog(
    "default",
    type="sql",
    uri=f"sqlite:///{CATALOG_DB}",
)

# -------------------------------------------------------------------
# 2b. Ensure namespace exists
# -------------------------------------------------------------------
try:
    catalog.create_namespace("default")
except Exception:
    pass  # ignore if namespace already exists

# -------------------------------------------------------------------
# 3. Define schemas for two tables
# -------------------------------------------------------------------
taxi_schema = Schema(
    NestedField(
        field_id=1,
        name="vendor_id",
        field_type=LongType(),
        required=True,
    ),
    NestedField(
        field_id=2,
        name="pickup_zone",
        field_type=StringType(),
        required=False,
    ),
)

weather_schema = Schema(
    NestedField(
        field_id=1,
        name="station_id",
        field_type=LongType(),
        required=True,
    ),
    NestedField(
        field_id=2,
        name="temperature",
        field_type=LongType(),
        required=False,
    ),
)

# -------------------------------------------------------------------
# 4. Create Table 1: Taxi (data in /tmp/table1)
# -------------------------------------------------------------------
try:
    taxi_table = catalog.create_table(
        identifier="default.taxi",
        schema=taxi_schema,
        location=f"file://{METADATA_BASE}/taxi",
        properties={
            "write.data.path": TABLE1_DATA,
        },
    )
except Exception:
    taxi_table = catalog.load_table("default.taxi")
    taxi_table = (
        taxi_table.transaction()
        .set_properties(**{"write.data.path": TABLE1_DATA})
        .commit_transaction()
    )

# Write taxi data
taxi_arrow_schema = pa.schema(
    [
        pa.field("vendor_id", pa.int64(), nullable=False),
        pa.field("pickup_zone", pa.string(), nullable=True),
    ]
)

taxi_data = pa.table(
    {
        "vendor_id": [1, 2, 3],
        "pickup_zone": ["A", "B", "C"],
    },
    schema=taxi_arrow_schema,
)

taxi_table.append(taxi_data)

# -------------------------------------------------------------------
# 5. Create Table 2: Weather (data in /tmp/table2)
# -------------------------------------------------------------------
try:
    weather_table = catalog.create_table(
        identifier="default.weather",
        schema=weather_schema,
        location=f"file://{METADATA_BASE}/weather",
        properties={
            "write.data.path": TABLE2_DATA,
        },
    )
except Exception:
    weather_table = catalog.load_table("default.weather")
    weather_table = (
        weather_table.transaction()
        .set_properties(**{"write.data.path": TABLE2_DATA})
        .commit_transaction()
    )

# Write weather data
weather_arrow_schema = pa.schema(
    [
        pa.field("station_id", pa.int64(), nullable=False),
        pa.field("temperature", pa.int64(), nullable=True),
    ]
)

weather_data = pa.table(
    {
        "station_id": [101, 102, 103],
        "temperature": [72, 68, 75],
    },
    schema=weather_arrow_schema,
)

weather_table.append(weather_data)

# -------------------------------------------------------------------
# 6. Verify separation of metadata and data for both tables
# -------------------------------------------------------------------
print("=" * 80)
print("DEMONSTRATION: Multiple Tables with Data in Separate Directories")
print("=" * 80)

# Table 1: Taxi
taxi_table = catalog.load_table("default.taxi")
taxi_snapshot = taxi_table.current_snapshot()

print("\n" + "=" * 80)
print("TABLE 1: Taxi")
print("=" * 80)
print(f"Metadata location: {taxi_table.location()}")
print(f"Data location:     {TABLE1_DATA}")
print(f"Snapshot ID:       {taxi_snapshot.snapshot_id}")

print(f"\nMetadata files in {METADATA_BASE}/taxi:")
for root, dirs, files in os.walk(f"{METADATA_BASE}/taxi"):
    for file in files:
        rel_path = os.path.join(root, file).replace(f"{METADATA_BASE}/taxi/", "")
        print(f"  - {rel_path}")

print(f"\nData files in /tmp/table1:")
for root, dirs, files in os.walk("/tmp/table1"):
    for file in files:
        rel_path = os.path.join(root, file).replace("/tmp/table1/", "")
        print(f"  - {rel_path}")

print(f"\nData files referenced in taxi manifest:")
for manifest in taxi_snapshot.manifests(taxi_table.io):
    for entry in manifest.fetch_manifest_entry(taxi_table.io):
        print(f"  - {entry.data_file.file_path}")

# Table 2: Weather
weather_table = catalog.load_table("default.weather")
weather_snapshot = weather_table.current_snapshot()

print("\n" + "=" * 80)
print("TABLE 2: Weather")
print("=" * 80)
print(f"Metadata location: {weather_table.location()}")
print(f"Data location:     {TABLE2_DATA}")
print(f"Snapshot ID:       {weather_snapshot.snapshot_id}")

print(f"\nMetadata files in {METADATA_BASE}/weather:")
for root, dirs, files in os.walk(f"{METADATA_BASE}/weather"):
    for file in files:
        rel_path = os.path.join(root, file).replace(f"{METADATA_BASE}/weather/", "")
        print(f"  - {rel_path}")

print(f"\nData files in /tmp/table2:")
for root, dirs, files in os.walk("/tmp/table2"):
    for file in files:
        rel_path = os.path.join(root, file).replace("/tmp/table2/", "")
        print(f"  - {rel_path}")

print(f"\nData files referenced in weather manifest:")
for manifest in weather_snapshot.manifests(weather_table.io):
    for entry in manifest.fetch_manifest_entry(weather_table.io):
        print(f"  - {entry.data_file.file_path}")

# Summary
print("\n" + "=" * 80)
print("SUCCESS: Two tables with metadata in shared location, data in separate dirs!")
print("=" * 80)
print(f"\nShared metadata location: {METADATA_BASE}/")
print(f"  - Taxi metadata:    {METADATA_BASE}/taxi/")
print(f"  - Weather metadata: {METADATA_BASE}/weather/")
print(f"\nSeparate data locations:")
print(f"  - Taxi data:    /tmp/table1/")
print(f"  - Weather data: /tmp/table2/")
print("=" * 80)
