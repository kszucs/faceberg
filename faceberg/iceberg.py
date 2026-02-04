"""Iceberg metadata generation utilities.

This module provides functions for creating Apache Iceberg table metadata from
Parquet files. The main entry point is write_snapshot(), which takes a complete
list of files and generates all required Iceberg metadata (manifests, snapshots,
table metadata).

Architecture
------------

    ┌─────────────────────────────────────────────────────────────────┐
    │                      write_snapshot()                            │
    │  Main entry point - receives complete file list for snapshot    │
    └───────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   diff_snapshot()     │
                    │                       │
                    │ Compare current files │
                    │ against previous      │
                    │ snapshot based on     │
                    │ uri/size/hash         │
                    │                       │
                    │ Returns: List of      │
                    │ (status, ParquetFile) │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  write_manifest()     │
                    │                       │
                    │ Convert ParquetFile   │
                    │ → DataFile via        │
                    │ parquet_file_to_      │
                    │ data_file()           │
                    │                       │
                    │ Write entries with    │
                    │ ADDED/EXISTING/       │
                    │ DELETED status        │
                    │                       │
                    │ Returns: ManifestFile │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  create_snapshot()    │
                    │                       │
                    │ Read manifest entries │
                    │ Build summary stats   │
                    │ via Snapshot          │
                    │ SummaryCollector      │
                    │                       │
                    │ Determine operation:  │
                    │ APPEND/DELETE/        │
                    │ OVERWRITE             │
                    │                       │
                    │ Returns: Snapshot     │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Write metadata files │
                    │                       │
                    │ - vN.metadata.json    │
                    │ - version-hint.text   │
                    └───────────────────────┘

Data Flow
---------
1. User provides List[ParquetFile] representing desired snapshot state
2. diff_snapshot() compares against previous to determine changes
3. write_manifest() converts to Iceberg DataFiles and writes single manifest
4. create_snapshot() reads manifest to build snapshot with statistics
5. Table metadata written to JSON files

Key Concepts
------------
ParquetFile: Simple dataclass with uri, size, and hash fields
ManifestEntry: Iceberg entry with status (ADDED/EXISTING/DELETED) + DataFile
Snapshot: Point-in-time view of table with summary statistics
Operation: Type of snapshot (APPEND/DELETE/OVERWRITE) determined by entry statuses

Public API
----------
write_snapshot(): Main entry point for creating Iceberg metadata
create_schema(): Convert PyArrow schema to Iceberg schema with field IDs
ParquetFile: Dataclass representing a parquet file to include in snapshot
"""

import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow as pa

from faceberg.discover import ParquetFile
from pyiceberg.io import FileIO
from pyiceberg.io.pyarrow import PyArrowFileIO, parquet_file_to_data_file, _pyarrow_to_schema_without_ids
from pyiceberg.manifest import (
    ManifestEntry,
    ManifestEntryStatus,
    ManifestFile,
    write_manifest as _write_manifest_avro,
    write_manifest_list,
)
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC, PartitionSpec
from pyiceberg.schema import Schema, assign_fresh_schema_ids
from pyiceberg.table.metadata import INITIAL_SEQUENCE_NUMBER, TableMetadataV2, new_table_metadata
from pyiceberg.table.name_mapping import create_mapping_from_schema

# Property key for schema name mapping
SCHEMA_NAME_MAPPING_DEFAULT = "schema.name-mapping.default"
from pyiceberg.table.refs import SnapshotRef, SnapshotRefType
from pyiceberg.table.snapshots import (
    Operation,
    Snapshot,
    SnapshotSummaryCollector,
    Summary,
    update_snapshot_summaries,
)
from pyiceberg.table.sorting import UNSORTED_SORT_ORDER


def diff_snapshot(
    current_files: List[ParquetFile],
    previous_metadata: Optional[TableMetadataV2],
    io: FileIO,
) -> List[Tuple[ManifestEntryStatus, ParquetFile]]:
    """Diff current files against previous snapshot.

    Compares files based on uri/size/hash to determine status:
    - ADDED: File exists in current but not in previous
    - EXISTING: File exists in both with same size and hash
    - REMOVED: File exists in previous but not in current
    - REMOVED + ADDED: File exists in both but size/hash changed

    Args:
        current_files: List of current ParquetFile objects
        previous_metadata: Previous table metadata (None for initial snapshot)
        io: FileIO for reading previous manifests

    Returns:
        List of (status, ParquetFile) tuples
    """
    # If no previous metadata, all files are ADDED
    if previous_metadata is None:
        return [(ManifestEntryStatus.ADDED, pf) for pf in current_files]

    # Build map of previous files: uri -> size
    previous_snapshot = previous_metadata.snapshot_by_id(previous_metadata.current_snapshot_id)
    if previous_snapshot is None:
        return [(ManifestEntryStatus.ADDED, pf) for pf in current_files]

    # Read all files from previous snapshot
    previous_files_map: Dict[str, int] = {}
    for manifest in previous_snapshot.manifests(io):
        for entry in manifest.fetch_manifest_entry(io=io, discard_deleted=True):
            df = entry.data_file
            previous_files_map[df.file_path] = df.file_size_in_bytes

    # Build map of current files
    current_files_map: Dict[str, ParquetFile] = {pf.uri: pf for pf in current_files}

    result: List[Tuple[ManifestEntryStatus, ParquetFile]] = []

    # Process current files
    for pf in current_files:
        if pf.uri not in previous_files_map:
            # New file
            result.append((ManifestEntryStatus.ADDED, pf))
        else:
            prev_size = previous_files_map[pf.uri]
            # Check if size changed (we don't have hash in DataFile, so use size as proxy)
            if pf.size == prev_size:
                # Same file
                result.append((ManifestEntryStatus.EXISTING, pf))
            else:
                # File changed: REMOVED (old) + ADDED (new)
                # Create ParquetFile for old version
                old_pf = ParquetFile(uri=pf.uri, path=pf.path, size=prev_size, blob_id="", split=None)
                result.append((ManifestEntryStatus.DELETED, old_pf))
                result.append((ManifestEntryStatus.ADDED, pf))

    # Process removed files (in previous but not in current)
    for uri, size in previous_files_map.items():
        if uri not in current_files_map:
            # File was removed - extract path from URI
            # URI format: hf://datasets/repo@revision/path
            path = uri.split("@", 1)[1].split("/", 1)[1] if "@" in uri else ""
            removed_pf = ParquetFile(uri=uri, path=path, size=size, blob_id="", split=None)
            result.append((ManifestEntryStatus.DELETED, removed_pf))

    return result


def create_schema(arrow_schema: pa.Schema) -> Schema:
    """Convert PyArrow schema to Iceberg Schema.

    Converts PyArrow schema to Iceberg Schema with globally unique field IDs
    assigned to all fields (including nested structures).

    Args:
        arrow_schema: PyArrow schema to convert

    Returns:
        Iceberg Schema with field IDs assigned
    """
    # Convert to schema without IDs, then assign fresh IDs
    schema_without_ids = _pyarrow_to_schema_without_ids(arrow_schema)
    return assign_fresh_schema_ids(schema_without_ids)


# TODO(kszucs): allow parallel calls to parquet_file_to_data_file
def write_manifest(
    files: List[Tuple[ManifestEntryStatus, ParquetFile]],
    metadata: TableMetadataV2,
    schema: Schema,
    spec: PartitionSpec,
    snapshot_id: int,
    sequence_number: int,
    io: FileIO,
    output_file,
    uri: str,
) -> ManifestFile:
    """Create and write a manifest file.

    Converts ParquetFile objects to DataFile objects and writes them
    to a single manifest with their respective statuses.

    Args:
        files: List of (status, ParquetFile) tuples
        metadata: Table metadata for reading parquet files
        schema: Iceberg schema
        spec: Partition specification
        snapshot_id: Snapshot ID for the entries
        sequence_number: Sequence number for the entries
        io: FileIO instance for reading files
        output_file: OutputFile to write to
        uri: URI path to use in the returned ManifestFile

    Returns:
        ManifestFile object
    """
    with _write_manifest_avro(
        format_version=2,
        spec=spec,
        schema=schema,
        output_file=output_file,
        snapshot_id=snapshot_id,
        avro_compression="deflate",
    ) as writer:
        for status, parquet_file in files:
            # Convert ParquetFile to DataFile
            data_file = parquet_file_to_data_file(
                io=io,
                table_metadata=metadata,
                file_path=parquet_file.uri,
            )

            # Create manifest entry with the appropriate status
            entry = ManifestEntry.from_args(
                status=status,
                snapshot_id=snapshot_id,
                sequence_number=sequence_number,
                file_sequence_number=sequence_number,
                data_file=data_file,
            )
            writer.add_entry(entry)
        manifest = writer.to_manifest_file()

    manifest_file = ManifestFile.from_args(
        manifest_path=uri,
        manifest_length=manifest.manifest_length,
        partition_spec_id=manifest.partition_spec_id,
        content=manifest.content,
        sequence_number=manifest.sequence_number,
        min_sequence_number=manifest.min_sequence_number,
        added_snapshot_id=manifest.added_snapshot_id,
        added_files_count=manifest.added_files_count,
        existing_files_count=manifest.existing_files_count,
        deleted_files_count=manifest.deleted_files_count,
        added_rows_count=manifest.added_rows_count,
        existing_rows_count=manifest.existing_rows_count,
        deleted_rows_count=manifest.deleted_rows_count,
        partitions=manifest.partitions,
        key_metadata=manifest.key_metadata,
    )

    return manifest_file


def create_snapshot(
    manifest: ManifestFile,
    manifest_list_path: str,
    snapshot_id: int,
    parent_snapshot_id: Optional[int],
    sequence_number: int,
    schema_id: int,
    spec: PartitionSpec,
    schema: Schema,
    io: FileIO,
    previous_summary: Optional[Summary] = None,
) -> Snapshot:
    """Create Snapshot object with proper summary.

    Uses SnapshotSummaryCollector and update_snapshot_summaries() to
    compute accurate statistics by reading entries from the manifest.

    Args:
        manifest: ManifestFile object
        manifest_list_path: Path to the manifest list
        snapshot_id: Snapshot ID
        parent_snapshot_id: Parent snapshot ID
        sequence_number: Sequence number
        schema_id: Schema ID
        spec: Partition specification
        schema: Iceberg schema
        io: FileIO for reading manifest entries
        previous_summary: Summary from previous snapshot (for totals)

    Returns:
        Snapshot object
    """
    # Build summary by processing manifest entries in a single pass
    ssc = SnapshotSummaryCollector(partition_summary_limit=0)
    has_added = False
    has_removed = False

    for entry in manifest.fetch_manifest_entry(io=io, discard_deleted=False):
        if entry.status == ManifestEntryStatus.ADDED:
            ssc.add_file(entry.data_file, schema=schema, partition_spec=spec)
            has_added = True
        elif entry.status == ManifestEntryStatus.DELETED:
            ssc.remove_file(entry.data_file, schema=schema, partition_spec=spec)
            has_removed = True

    # Determine operation type
    if has_removed and has_added:
        operation = Operation.OVERWRITE
    elif has_removed:
        operation = Operation.DELETE
    else:
        operation = Operation.APPEND

    summary = Summary(operation=operation, **ssc.build())
    summary = update_snapshot_summaries(summary, previous_summary)

    return Snapshot(
        snapshot_id=snapshot_id,
        parent_snapshot_id=parent_snapshot_id,
        sequence_number=sequence_number,
        timestamp_ms=int(uuid.uuid4().time_low),  # Use UUID time component
        manifest_list=manifest_list_path,
        summary=summary,
        schema_id=schema_id,
    )


def write_snapshot(
    files: List[ParquetFile],
    schema: pa.Schema,
    current_metadata: Optional[TableMetadataV2],
    output_dir: Path,
    base_uri: str,
    properties: Optional[Dict[str, str]] = None,
    partition_spec: Optional[PartitionSpec] = None,
) -> TableMetadataV2:
    """Write new snapshot metadata.

    This is the main entry point for creating Iceberg metadata. Compares the
    provided files against the previous snapshot to determine operations:
    - APPEND: only added files
    - DELETE: only removed files
    - OVERWRITE: both added and removed files

    Args:
        files: Complete list of ParquetFile objects for this snapshot
        schema: PyArrow schema
        current_metadata: Existing metadata (None for initial snapshot)
        output_dir: Directory to write metadata files
        base_uri: Base URI for paths in metadata
        properties: Table properties
        partition_spec: Optional partition spec (default: unpartitioned)

    Returns:
        Updated TableMetadataV2
    """
    properties = properties or {}
    partition_spec = partition_spec or UNPARTITIONED_PARTITION_SPEC
    io = PyArrowFileIO()

    # Ensure metadata directory exists
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Set up context based on whether this is initial or subsequent snapshot
    if current_metadata is None:
        table_uuid = uuid.UUID(str(uuid.uuid4()))
        snapshot_id = 1
        sequence_number = INITIAL_SEQUENCE_NUMBER
        parent_snapshot_id = None
        previous_summary = None

        # Convert schema and create name mapping
        iceberg_schema_obj = create_schema(schema)
        name_mapping = create_mapping_from_schema(iceberg_schema_obj)
        merged_properties = {
            **properties,
            SCHEMA_NAME_MAPPING_DEFAULT: name_mapping.model_dump_json(),
        }

        # Create preliminary metadata for reading parquet files
        file_metadata = new_table_metadata(
            schema=iceberg_schema_obj,
            partition_spec=partition_spec,
            sort_order=UNSORTED_SORT_ORDER,
            location=base_uri,
            properties=merged_properties,
            table_uuid=table_uuid,
        )
        spec = partition_spec
    else:
        table_uuid = current_metadata.table_uuid
        snapshot_id = max(s.snapshot_id for s in current_metadata.snapshots) + 1
        sequence_number = current_metadata.last_sequence_number + 1
        parent_snapshot_id = current_metadata.current_snapshot_id

        previous_snapshot = current_metadata.snapshot_by_id(parent_snapshot_id)
        previous_summary = previous_snapshot.summary if previous_snapshot else None

        iceberg_schema_obj = current_metadata.schema()
        file_metadata = current_metadata
        spec = current_metadata.spec()

        merged_properties = {**current_metadata.properties}
        if properties:
            merged_properties.update(properties)

    # Diff the provided files against previous snapshot
    diff_results = diff_snapshot(files, current_metadata, io)

    # Create single manifest with all entries (mixed statuses)
    manifest_filename = f"{uuid.uuid4()}.avro"
    manifest_path = metadata_dir / manifest_filename
    manifest_uri = f"{base_uri}/metadata/{manifest_filename}"

    output_file = io.new_output(str(manifest_path))
    manifest = write_manifest(
        diff_results,
        file_metadata,
        iceberg_schema_obj,
        spec,
        snapshot_id,
        sequence_number,
        io,
        output_file,
        manifest_uri,
    )
    all_manifests = [manifest]

    # Create manifest list
    manifest_list_filename = f"snap-{snapshot_id}-{sequence_number}-{uuid.uuid4()}.avro"
    manifest_list_path = metadata_dir / manifest_list_filename
    manifest_list_uri = f"{base_uri}/metadata/{manifest_list_filename}"

    manifest_list_output = io.new_output(str(manifest_list_path))
    with write_manifest_list(
        format_version=2,
        output_file=manifest_list_output,
        snapshot_id=snapshot_id,
        parent_snapshot_id=parent_snapshot_id,
        sequence_number=sequence_number,
        avro_compression="deflate",
    ) as writer:
        writer.add_manifests(all_manifests)

    # Create snapshot
    snapshot = create_snapshot(
        manifest,
        manifest_list_uri,
        snapshot_id,
        parent_snapshot_id,
        sequence_number,
        iceberg_schema_obj.schema_id,
        spec,
        iceberg_schema_obj,
        io,
        previous_summary=previous_summary,
    )

    # Create table metadata
    if current_metadata is None:
        metadata = TableMetadataV2(
            location=base_uri,
            table_uuid=table_uuid,
            last_updated_ms=snapshot.timestamp_ms,
            last_column_id=iceberg_schema_obj.highest_field_id,
            schemas=[iceberg_schema_obj],
            current_schema_id=iceberg_schema_obj.schema_id,
            partition_specs=[partition_spec],
            default_spec_id=partition_spec.spec_id,
            last_partition_id=partition_spec.last_assigned_field_id,
            properties=merged_properties,
            current_snapshot_id=snapshot.snapshot_id,
            snapshots=[snapshot],
            snapshot_log=[],
            metadata_log=[],
            sort_orders=[UNSORTED_SORT_ORDER],
            default_sort_order_id=UNSORTED_SORT_ORDER.order_id,
            refs={"main": SnapshotRef(snapshot_id=snapshot.snapshot_id, type=SnapshotRefType.BRANCH)},
            format_version=2,
            last_sequence_number=sequence_number,
        )
    else:
        metadata = TableMetadataV2(
            location=current_metadata.location,
            table_uuid=table_uuid,
            last_updated_ms=snapshot.timestamp_ms,
            last_column_id=current_metadata.last_column_id,
            schemas=current_metadata.schemas,
            current_schema_id=current_metadata.current_schema_id,
            partition_specs=current_metadata.partition_specs,
            default_spec_id=current_metadata.default_spec_id,
            last_partition_id=current_metadata.last_partition_id,
            properties=merged_properties,
            current_snapshot_id=snapshot.snapshot_id,
            snapshots=list(current_metadata.snapshots) + [snapshot],
            snapshot_log=current_metadata.snapshot_log,
            metadata_log=current_metadata.metadata_log,
            sort_orders=current_metadata.sort_orders,
            default_sort_order_id=current_metadata.default_sort_order_id,
            refs={"main": SnapshotRef(snapshot_id=snapshot.snapshot_id, type=SnapshotRefType.BRANCH)},
            format_version=2,
            last_sequence_number=sequence_number,
        )

    # Write metadata file and version hint
    version = len(metadata.snapshots)
    metadata_file = metadata_dir / f"v{version}.metadata.json"
    with open(metadata_file, "w") as f:
        f.write(metadata.model_dump_json(indent=2))

    version_hint_file = metadata_dir / "version-hint.text"
    with open(version_hint_file, "w") as f:
        f.write(str(version))

    return metadata
