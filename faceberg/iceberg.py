"""Iceberg metadata generation utilities.

This module provides functions for creating Apache Iceberg table metadata from
Parquet files. The main entry point is write_snapshot(), which takes a complete
list of files and generates all required Iceberg metadata (manifests, snapshots,
table metadata).

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
create_schema(): Convert PyArrow schema to Iceberg schema with field IDs (optionally
                 with split column)
create_partition_spec(): Create a partition spec with optional split partitioning
ParquetFile: Dataclass representing a parquet file to include in snapshot

File Structure and Type Hierarchy
----------------------------------

Physical Files Created:
    table/
    └── metadata/
        ├── v1.metadata.json          (TableMetadataV2)
        ├── v2.metadata.json          (TableMetadataV2) - for subsequent snapshots
        ├── version-hint.text         (current version number)
        ├── snap-1-0-<uuid>.avro      (ManifestList)
        ├── snap-2-1-<uuid>.avro      (ManifestList) - for subsequent snapshots
        ├── <uuid>.avro               (Manifest file)
        └── <uuid>.avro               (Manifest file) - one per snapshot

Type Hierarchy:
    TableMetadataV2                   # Root metadata object
    ├── schemas: List[Schema]         # Iceberg schema definitions
    ├── partition_specs: List[PartitionSpec]
    ├── snapshots: List[Snapshot]     # All table snapshots
    │   └── Snapshot
    │       ├── snapshot_id: int
    │       ├── manifest_list: str    # → snap-X-Y-<uuid>.avro
    │       └── summary: Summary      # Operation stats + HF metadata
    └── refs: Dict[str, SnapshotRef]  # Branch references (e.g., "main")

ManifestList (snap-X-Y-<uuid>.avro)   # Written to manifest_list path
└── manifests: List[ManifestFile]     # References to manifest files
    └── ManifestFile
        ├── manifest_path: str        # → <uuid>.avro
        ├── added_files_count: int
        ├── added_rows_count: int
        └── partition_spec_id: int

Manifest (<uuid>.avro)                # Written to manifest_path
└── entries: List[ManifestEntry]
    └── ManifestEntry
        ├── status: ManifestEntryStatus  # ADDED/EXISTING/DELETED
        ├── snapshot_id: int
        ├── sequence_number: int
        └── data_file: DataFile       # ↓

DataFile                              # References actual data
├── file_path: str                    # → hf://datasets/org/repo@rev/file.parquet
├── file_format: FileFormat           # PARQUET
├── partition: Dict[int, str]         # {0: "train"} for split partitioning
├── record_count: int                 # Number of rows
├── file_size_in_bytes: int
└── file_sequence_number: int         # Tracks when file was added

Note: DataFile objects reference external HuggingFace parquet files without
copying them. All metadata files use Iceberg's Avro format for manifests and
JSON for table metadata.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from pyiceberg.io import FileIO
from pyiceberg.io.pyarrow import (
    _pyarrow_to_schema_without_ids,
    compute_statistics_plan,
    data_file_statistics_from_parquet_metadata,
)
from pyiceberg.io.pyarrow import parquet_path_to_id_mapping as _parquet_path_to_id_mapping
from pyiceberg.manifest import (
    DataFile,
    DataFileContent,
    ManifestEntry,
    ManifestEntryStatus,
    ManifestFile,
    write_manifest_list,
)
from pyiceberg.manifest import write_manifest as _write_manifest
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC, PartitionField, PartitionSpec
from pyiceberg.schema import Schema, assign_fresh_schema_ids
from pyiceberg.table import TableProperties
from pyiceberg.table.metadata import INITIAL_SEQUENCE_NUMBER, TableMetadataV2, new_table_metadata
from pyiceberg.table.refs import SnapshotRef, SnapshotRefType
from pyiceberg.table.snapshots import (
    Operation,
    Snapshot,
    SnapshotSummaryCollector,
    Summary,
    update_snapshot_summaries,
)
from pyiceberg.table.sorting import UNSORTED_SORT_ORDER
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import NestedField, StringType

from .discover import ParquetFile


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
                old_pf = ParquetFile(
                    uri=pf.uri, path=pf.path, size=prev_size, blob_id="", split=None
                )
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


def create_schema(arrow_schema: pa.Schema, include_split_column: bool) -> Schema:
    """Convert PyArrow schema to Iceberg Schema.

    Converts PyArrow schema to Iceberg Schema with globally unique field IDs
    assigned to all fields (including nested structures).

    Args:
        arrow_schema: PyArrow schema to convert
        include_split_column: If True, adds a 'split' column as the first field (default: False)

    Returns:
        Iceberg Schema with field IDs assigned
    """
    # Convert to schema without IDs, then assign fresh IDs
    schema_without_ids = _pyarrow_to_schema_without_ids(arrow_schema)
    schema = assign_fresh_schema_ids(schema_without_ids)

    # Add split column as the first field if requested
    if include_split_column:
        # Create split field (will get ID 1 after reassignment)
        # Note: Although the schema uses StringType, the actual Parquet data
        # will use dictionary encoding (int8 indices) for compression efficiency
        # The split column is optional since it doesn't exist in the source Parquet files,
        # it's derived from partition metadata
        split_field = NestedField(
            field_id=-1,  # Temporary ID, will be reassigned
            name="split",
            field_type=StringType(),
            required=False,
        )
        # Prepend split field to existing fields
        new_fields = [split_field] + list(schema.fields)

        # Create new schema and reassign all field IDs globally
        # This ensures field IDs are globally unique across nested structures
        schema_with_split = Schema(*new_fields)
        schema = assign_fresh_schema_ids(schema_with_split)

    return schema


def create_partition_spec(schema: Schema, include_split_column: bool) -> PartitionSpec:
    """Build a partition spec with optional split partitioning.

    Creates an identity partition on the split field when requested.
    When False, returns an unpartitioned spec.

    Args:
        schema: Iceberg schema
        include_split_column: Whether to partition by split field (default: False)

    Returns:
        PartitionSpec with split partition key if include_split_column is True,
        or UNPARTITIONED_PARTITION_SPEC otherwise

    Raises:
        ValueError: If include_split_column is True but schema doesn't contain a 'split' field
    """
    if not include_split_column:
        return UNPARTITIONED_PARTITION_SPEC

    split_field = schema.find_field("split")
    if split_field is None:
        raise ValueError("Schema must contain a 'split' field to create split partition spec")

    return PartitionSpec(
        PartitionField(
            source_id=split_field.field_id,
            field_id=1000,  # Partition field IDs start at 1000
            transform=IdentityTransform(),
            name="split",
        ),
        spec_id=0,
    )


# TODO(kszucs): copied from pyiceberg.io.pyarrow with modifications to resolve list
# field mapping issues, remove once fixed in pyiceberg
def parquet_path_to_id_mapping(schema: Schema) -> dict[str, int]:
    """Build a field mapping that handles both 'element' and 'item' list conventions.

    Creates mappings for both PyArrow-compliant ('element') and actual Parquet
    schema paths. This handles cases where Parquet files use 'item' (Arrow convention)
    instead of 'element' (Parquet spec).
    """
    # Start with standard iceberg mapping (uses 'element')
    base_mapping = _parquet_path_to_id_mapping(schema)

    # Create alternative mappings by replacing 'element' with 'item'
    flexible_mapping = dict(base_mapping)
    for path, field_id in base_mapping.items():
        if ".list.element" in path:
            # Add mapping with 'item' convention
            alt_path = path.replace(".list.element", ".list.item")
            flexible_mapping[alt_path] = field_id

    return flexible_mapping


# TODO(kszucs): copied from pyiceberg.io.pyarrow with modifications to resolve list
# field mapping issues, remove once fixed in pyiceberg
def create_data_file(
    io: FileIO,
    table_metadata: "TableMetadataV2",
    parquet_file: ParquetFile,
    include_split_column: bool,
) -> DataFile:
    """Convert ParquetFile to DataFile using flexible field mapping.

    This implementation builds a flexible field mapping that supports both
    'element' (Parquet spec) and 'item' (Arrow convention) for list fields,
    handling Parquet files written by both spec-compliant and non-compliant writers.

    Args:
        io: FileIO for reading parquet files
        table_metadata: Table metadata containing schema and partition spec
        parquet_file: ParquetFile with uri, size, and optional split metadata
        include_split_column: If True, includes split from ParquetFile in partition

    Returns:
        DataFile with appropriate partition values
    """
    # TODO(kszucs): this is a port of the upstream parquet_file_to_data_file function
    # with modifications to handle list field mapping issues, nce the upstream issue
    # is resolved should use the original from pyiceberg.io.pyarrow directly
    input_file = io.new_input(parquet_file.uri)
    with input_file.open() as f:
        parquet_metadata = pq.read_metadata(f)

    schema = table_metadata.schema()
    spec = table_metadata.spec()

    # Use flexible mapping that handles both 'element' and 'item'
    statistics = data_file_statistics_from_parquet_metadata(
        parquet_metadata=parquet_metadata,
        stats_columns=compute_statistics_plan(schema, table_metadata.properties),
        parquet_column_mapping=parquet_path_to_id_mapping(schema),
    )

    # Get partition from statistics (handles columns present in parquet file)
    partition = statistics.partition(spec, schema)
    # Add split to partition if requested and we have split metadata
    # The split is not in the parquet file itself, it's metadata we know about the file
    if include_split_column:
        for i, field in enumerate(spec.fields):
            if field.name == "split":
                partition[i] = parquet_file.split

    return DataFile.from_args(
        content=DataFileContent.DATA,
        file_path=parquet_file.uri,
        file_format="PARQUET",
        partition=partition,
        file_size_in_bytes=parquet_file.size,
        sort_order_id=None,
        spec_id=table_metadata.default_spec_id,
        equality_ids=None,
        key_metadata=None,
        **statistics.to_serialized_dict(),
    )


def write_manifest(
    files: List[Tuple[ManifestEntryStatus, ParquetFile]],
    metadata: TableMetadataV2,
    schema: Schema,
    spec: PartitionSpec,
    snapshot_id: int,
    sequence_number: int,
    io: FileIO,
    output_file,
    manifest_uri: str,
    include_split_column: bool,
    progress_callback: Callable,
    max_workers: Optional[int] = None,
) -> Tuple[ManifestFile, List]:
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
        manifest_uri: URI path to use in the returned ManifestFile
        include_split_column: If True, includes split from ParquetFile in partition
        progress_callback: Callable for reporting progress (state, percent, stage)
        max_workers: Maximum number of threads for parallel DataFile conversion.
            If None, uses ThreadPoolExecutor default

    Returns:
        Tuple of (ManifestFile object, List of ManifestEntry objects)
    """
    progress_callback(state="in_progress", percent=20, stage="Creating metadata files")

    # Convert ParquetFiles to DataFiles in parallel (I/O bound operation)
    total_files = len(files)
    completed_files = 0
    lock = Lock()

    def convert_file(parquet_file: ParquetFile) -> DataFile:
        nonlocal completed_files
        result = create_data_file(
            io=io,
            table_metadata=metadata,
            parquet_file=parquet_file,
            include_split_column=include_split_column,
        )
        with lock:
            completed_files += 1
            fraction_done = completed_files / total_files
            percent = 20 + int(fraction_done * 70)  # Map to 20-90% range
            progress_callback(
                state="in_progress",
                percent=percent,
                stage=f"{parquet_file.path} ({completed_files}/{total_files})",
            )
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        parquet_files = [pf for _, pf in files]
        data_files = list(executor.map(convert_file, parquet_files))

    # Create a mapping from ParquetFile to DataFile
    data_file_map = {pf.uri: df for pf, df in zip(parquet_files, data_files)}

    # Write manifest entries with pre-computed DataFiles
    entries = []
    with _write_manifest(
        format_version=2,
        spec=spec,
        schema=schema,
        output_file=output_file,
        snapshot_id=snapshot_id,
        avro_compression="deflate",
    ) as writer:
        for status, parquet_file in files:
            data_file = data_file_map[parquet_file.uri]

            # Create manifest entry with the appropriate status
            entry = ManifestEntry.from_args(
                status=status,
                snapshot_id=snapshot_id,
                sequence_number=sequence_number,
                file_sequence_number=sequence_number,
                data_file=data_file,
            )
            writer.add_entry(entry)
            entries.append(entry)
        manifest = writer.to_manifest_file()

    manifest_file = ManifestFile.from_args(
        manifest_path=manifest_uri,
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

    return manifest_file, entries


def create_snapshot(
    manifest_entries: List,
    manifest_list_path: str,
    snapshot_id: int,
    parent_snapshot_id: Optional[int],
    sequence_number: int,
    schema_id: int,
    spec: PartitionSpec,
    schema: Schema,
    previous_summary: Optional[Summary] = None,
) -> Snapshot:
    """Create Snapshot object with proper summary.

    Uses SnapshotSummaryCollector and update_snapshot_summaries() to
    compute accurate statistics from the provided manifest entries.

    Args:
        manifest_entries: List of ManifestEntry objects. Must be provided to avoid
            file I/O issues with staging directories. The entries should be collected
            during manifest creation.
        manifest_list_path: Path to the manifest list
        snapshot_id: Snapshot ID
        parent_snapshot_id: Parent snapshot ID
        sequence_number: Sequence number
        schema_id: Schema ID
        spec: Partition specification
        schema: Iceberg schema
        previous_summary: Summary from previous snapshot (for totals)

    Returns:
        Snapshot object
    """
    # Build summary by processing manifest entries in a single pass
    ssc = SnapshotSummaryCollector(partition_summary_limit=0)
    has_added = False
    has_removed = False

    for entry in manifest_entries:
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
    properties: Dict[str, str],
    progress_callback: Callable,
    include_split_column: bool,
    io: FileIO,
    max_workers: Optional[int] = None,
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
        include_split_column: Whether to include a 'split' column in the schema and partition
            by it. When True, adds a split column to the schema and partitions by split.
            When False, uses unpartitioned spec (default: True)
        io: FileIO instance
        max_workers: Maximum number of threads for parallel DataFile conversion in manifest writing.
            If None, uses ThreadPoolExecutor default

    Returns:
        Updated TableMetadataV2
    """
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

        # Convert schema with optional split column
        iceberg_schema = create_schema(schema, include_split_column=include_split_column)
        merged_properties = {
            **properties,
            TableProperties.DEFAULT_NAME_MAPPING: iceberg_schema.name_mapping.model_dump_json(),
        }

        # Create partition spec (partition by split if split column is included)
        spec = create_partition_spec(iceberg_schema, include_split_column=include_split_column)

        # Create preliminary metadata for reading parquet files
        file_metadata = new_table_metadata(
            schema=iceberg_schema,
            partition_spec=spec,
            sort_order=UNSORTED_SORT_ORDER,
            location=base_uri,
            properties=merged_properties,
            table_uuid=table_uuid,
        )
    else:
        table_uuid = current_metadata.table_uuid
        snapshot_id = max(s.snapshot_id for s in current_metadata.snapshots) + 1
        sequence_number = current_metadata.last_sequence_number + 1
        parent_snapshot_id = current_metadata.current_snapshot_id

        previous_snapshot = current_metadata.snapshot_by_id(parent_snapshot_id)
        previous_summary = previous_snapshot.summary if previous_snapshot else None

        iceberg_schema = current_metadata.schema()
        file_metadata = current_metadata
        spec = current_metadata.spec()

        merged_properties = {**current_metadata.properties}
        if properties:
            merged_properties.update(properties)

    # Diff the provided files against previous snapshot
    progress_callback(state="in_progress", percent=15, stage="Diffing snapshots")
    diff_results = diff_snapshot(files, current_metadata, io)

    # Create single manifest with all entries (mixed statuses)
    manifest_filename = f"{uuid.uuid4()}.avro"
    manifest_path = metadata_dir / manifest_filename
    manifest_uri = f"{base_uri}/metadata/{manifest_filename}"

    output_file = io.new_output(str(manifest_path))
    # Write manifest with final URI and get entries

    progress_callback(state="in_progress", percent=20, stage="Writing manifest file")
    manifest, manifest_entries = write_manifest(
        files=diff_results,
        metadata=file_metadata,
        schema=iceberg_schema,
        spec=spec,
        snapshot_id=snapshot_id,
        sequence_number=sequence_number,
        io=io,
        output_file=output_file,
        manifest_uri=manifest_uri,
        include_split_column=include_split_column,
        max_workers=max_workers,
        progress_callback=progress_callback,
    )

    # Create manifest list
    manifest_list_filename = f"snap-{snapshot_id}-{sequence_number}-{uuid.uuid4()}.avro"
    manifest_list_path = metadata_dir / manifest_list_filename
    manifest_list_uri = f"{base_uri}/metadata/{manifest_list_filename}"

    progress_callback(state="in_progress", percent=90, stage="Writing manifest list")
    manifest_list_output = io.new_output(str(manifest_list_path))
    with write_manifest_list(
        format_version=2,
        output_file=manifest_list_output,
        snapshot_id=snapshot_id,
        parent_snapshot_id=parent_snapshot_id,
        sequence_number=sequence_number,
        avro_compression="deflate",
    ) as writer:
        writer.add_manifests([manifest])

    # Create snapshot using the collected manifest entries (avoids reading from file)
    snapshot = create_snapshot(
        manifest_entries,
        manifest_list_uri,
        snapshot_id,
        parent_snapshot_id,
        sequence_number,
        iceberg_schema.schema_id,
        spec,
        iceberg_schema,
        previous_summary=previous_summary,
    )

    # Create table metadata
    if current_metadata is None:
        metadata = TableMetadataV2(
            location=base_uri,
            table_uuid=table_uuid,
            last_updated_ms=snapshot.timestamp_ms,
            last_column_id=iceberg_schema.highest_field_id,
            schemas=[iceberg_schema],
            current_schema_id=iceberg_schema.schema_id,
            partition_specs=[spec],
            default_spec_id=spec.spec_id,
            last_partition_id=spec.last_assigned_field_id,
            properties=merged_properties,
            current_snapshot_id=snapshot.snapshot_id,
            snapshots=[snapshot],
            snapshot_log=[],
            metadata_log=[],
            sort_orders=[UNSORTED_SORT_ORDER],
            default_sort_order_id=UNSORTED_SORT_ORDER.order_id,
            refs={
                "main": SnapshotRef(snapshot_id=snapshot.snapshot_id, type=SnapshotRefType.BRANCH)
            },
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
            refs={
                "main": SnapshotRef(snapshot_id=snapshot.snapshot_id, type=SnapshotRefType.BRANCH)
            },
            format_version=2,
            last_sequence_number=sequence_number,
        )

    # Write metadata file and version hint
    progress_callback(state="in_progress", percent=95, stage="Writing metadata file and type hint")
    version = len(metadata.snapshots)
    metadata_file = metadata_dir / f"v{version}.metadata.json"
    with open(metadata_file, "w") as f:
        f.write(metadata.model_dump_json(indent=2))

    version_hint_file = metadata_dir / "version-hint.text"
    with open(version_hint_file, "w") as f:
        f.write(str(version))

    return metadata
