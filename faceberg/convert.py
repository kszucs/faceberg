"""Conversion from TableInfo to Iceberg metadata files.

This module takes TableInfo objects (created by the bridge layer) and converts them
into actual Iceberg metadata files in metadata-only mode, referencing the original
HuggingFace dataset files.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

from pyiceberg.io.pyarrow import PyArrowFileIO, parquet_file_to_data_file
from pyiceberg.manifest import (
    DataFile,
    ManifestEntry,
    ManifestEntryStatus,
    ManifestFile,
    write_manifest,
    write_manifest_list,
)
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC, PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table.metadata import INITIAL_SEQUENCE_NUMBER, TableMetadataV2, new_table_metadata
from pyiceberg.table.refs import SnapshotRef, SnapshotRefType
from pyiceberg.table.snapshots import Operation, Snapshot, Summary
from pyiceberg.table.sorting import UNSORTED_SORT_ORDER
from pyiceberg.transforms import IdentityTransform

# Import FileInfo (created by bridge layer)
from .bridge import FileInfo

logger = logging.getLogger(__name__)


# TODO(kszucs): parallelize metadata creation for large number of files


class IcebergMetadataWriter:
    """Writes Iceberg metadata files in metadata-only mode.

    This writer creates Iceberg metadata (manifest, manifest list, table metadata)
    that references existing HuggingFace dataset files without copying or modifying them.

    File Structure and Type Hierarchy:

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

    def __init__(
        self,
        table_path: Path,
        schema: Schema,
        partition_spec: PartitionSpec = UNPARTITIONED_PARTITION_SPEC,
        base_uri: str = None,
    ):
        """Initialize metadata writer.

        Args:
            table_path: Local directory for physically writing files (staging directory)
            schema: Iceberg schema
            partition_spec: Partition specification
            base_uri: Base URI for paths stored in metadata
                     (e.g., "file:///path/to/catalog" or "hf://datasets/org/repo")
        """
        self.table_path = table_path
        self.schema = schema
        self.partition_spec = partition_spec
        self.metadata_dir = table_path / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.file_io = PyArrowFileIO()

        # Store base URI for metadata references
        self.base_uri = base_uri.rstrip("/")

    def create_metadata_from_files(
        self,
        file_infos: List[FileInfo],
        table_uuid: str,
        properties: Optional[Dict[str, str]] = None,
        progress_callback: Optional[Callable] = None,
        identifier: Optional[str] = None,
    ) -> Path:
        """Create Iceberg metadata from data file information.

        This method creates all necessary Iceberg metadata files:
        - Manifest file (.avro)
        - Manifest list (.avro)
        - Table metadata (v1.metadata.json)
        - Version hint (version-hint.text)

        Args:
            file_infos: List of FileInfo objects describing data files
            table_uuid: UUID for the table
            properties: Optional table properties
            progress_callback: Optional callback for progress updates
            identifier: Optional table identifier for progress reporting

        Returns:
            Path to the metadata file
        """
        logger.info(f"Creating Iceberg metadata for {len(file_infos)} files")

        # Create preliminary TableMetadata (without snapshots) for parquet_file_to_data_file
        preliminary_metadata = self._create_preliminary_metadata(table_uuid, properties or {})

        # Create DataFile entries directly from FileInfo (reads parquet metadata)
        data_files = self._create_data_files_from_info(
            file_infos,
            preliminary_metadata,
            sequence_number=INITIAL_SEQUENCE_NUMBER,
            progress_callback=progress_callback,
            identifier=identifier,
        )

        # Write metadata files
        return self._write_metadata_files(data_files, table_uuid, properties or {})

    def _create_preliminary_metadata(
        self, table_uuid: str, properties: Dict[str, str]
    ) -> TableMetadataV2:
        """Create preliminary TableMetadata without snapshots.

        This is used by parquet_file_to_data_file to get schema and partition spec.

        Args:
            table_uuid: UUID for the table
            properties: Table properties

        Returns:
            TableMetadataV2 object without snapshots
        """
        metadata = new_table_metadata(
            schema=self.schema,
            partition_spec=self.partition_spec,
            sort_order=UNSORTED_SORT_ORDER,
            location=self.base_uri,
            properties=properties,
            table_uuid=uuid.UUID(table_uuid),
        )

        # Update partition spec with correct field IDs if partitioned
        if self.partition_spec != UNPARTITIONED_PARTITION_SPEC:
            reassigned_schema = metadata.schema()
            split_field = reassigned_schema.find_field("split")
            if split_field:
                partition_spec_with_correct_ids = PartitionSpec(
                    PartitionField(
                        source_id=split_field.field_id,
                        field_id=1000,
                        transform=IdentityTransform(),
                        name="split",
                    ),
                    spec_id=0,
                )
                metadata = TableMetadataV2(  # type: ignore[call-arg]
                    location=metadata.location,
                    table_uuid=metadata.table_uuid,
                    last_updated_ms=metadata.last_updated_ms,
                    last_column_id=metadata.last_column_id,
                    schemas=metadata.schemas,
                    current_schema_id=metadata.current_schema_id,
                    partition_specs=[partition_spec_with_correct_ids],
                    default_spec_id=0,
                    last_partition_id=1000,
                    properties=metadata.properties,
                    current_snapshot_id=None,
                    snapshots=[],
                    snapshot_log=[],
                    metadata_log=[],
                    sort_orders=metadata.sort_orders,
                    default_sort_order_id=metadata.default_sort_order_id,
                    refs={},
                    format_version=2,
                    last_sequence_number=INITIAL_SEQUENCE_NUMBER,
                )

        return metadata

    def _create_data_files_from_info(
        self,
        file_infos: List[FileInfo],
        table_metadata: TableMetadataV2,
        sequence_number: int = INITIAL_SEQUENCE_NUMBER,
        progress_callback: Optional[Callable] = None,
        identifier: Optional[str] = None,
    ) -> List[DataFile]:
        """Create Iceberg DataFile entries directly from FileInfo objects.

        This method uses pyiceberg's parquet_file_to_data_file to extract parquet
        metadata and populate DataFile fields with proper statistics.

        Args:
            file_infos: List of FileInfo objects describing data files
            table_metadata: TableMetadata for schema and partition spec
            sequence_number: Current sequence number (default: 0 for initial snapshot)
            progress_callback: Optional callback for progress updates
            identifier: Optional table identifier for progress reporting

        Returns:
            List of Iceberg DataFile objects with metadata populated from parquet files

        Raises:
            Exception: If metadata cannot be read from any file
        """
        data_files = []
        total_files = len(file_infos)

        for i, file_info in enumerate(file_infos):
            # Use pyiceberg's helper to create DataFile with extracted metadata
            data_file = parquet_file_to_data_file(
                io=self.file_io,
                table_metadata=table_metadata,
                file_path=file_info.uri,
            )

            data_files.append(data_file)

            # Report progress after processing each file
            if progress_callback and identifier:
                percent = 10 + int((i + 1) / total_files * 80)
                progress_callback(identifier, state="in_progress", percent=percent)

        return data_files

    def _get_previous_manifests(self, metadata: TableMetadataV2) -> Optional[List[ManifestFile]]:
        """Extract manifest file references from the current snapshot without reading
        their contents.

        This method is used for fast append operations to reuse existing manifest files
        without downloading and reading their contents. This significantly reduces
        bandwidth and improves performance for remote catalogs.

        Args:
            metadata: Current table metadata

        Returns:
            List of ManifestFile objects from current snapshot, or None if no snapshots
        """
        # Return None if there's no current snapshot
        if not metadata.current_snapshot_id or not metadata.snapshots:
            return None

        # Find the current snapshot
        current_snapshot = next(
            (s for s in metadata.snapshots if s.snapshot_id == metadata.current_snapshot_id),
            None,
        )
        if not current_snapshot:
            return None

        # Return manifest file references (without reading their contents)
        return list(current_snapshot.manifests(self.file_io))

    def _write_metadata_files(
        self,
        data_files: List[DataFile],
        table_uuid: str,
        properties: Dict[str, str],
    ) -> Path:
        """Write Iceberg table metadata, manifest, and manifest list.

        Args:
            data_files: List of data files
            table_uuid: UUID for the table
            properties: Table properties

        Returns:
            Path to the metadata file
        """
        # Step 1: Create and write manifest
        manifest_file = self._create_manifest(data_files)

        # Step 2: Create snapshot (with properties for summary fields)
        snapshot = self._create_snapshot(data_files, properties)

        # Step 3: Write manifest list
        self._write_manifest_list(snapshot, [manifest_file])

        # Step 4: Create table metadata
        metadata = self._create_table_metadata(snapshot, table_uuid, properties)

        # Step 5: Write metadata file
        return self._write_metadata_file(metadata)

    def _create_manifest(self, data_files: List[DataFile]):
        """Create and write manifest file.

        Args:
            data_files: List of data files

        Returns:
            ManifestFile object
        """
        manifest_filename = f"{uuid.uuid4()}.avro"
        manifest_write_path = self.metadata_dir / manifest_filename
        manifest_uri = f"{self.base_uri.rstrip('/')}/metadata/{manifest_filename}"
        output_file = self.file_io.new_output(str(manifest_write_path))

        with write_manifest(
            format_version=2,
            spec=self.partition_spec,
            schema=self.schema,
            output_file=output_file,
            snapshot_id=1,
            avro_compression="deflate",
        ) as writer:
            for data_file in data_files:
                entry = ManifestEntry.from_args(
                    status=ManifestEntryStatus.ADDED,
                    snapshot_id=1,
                    sequence_number=INITIAL_SEQUENCE_NUMBER,
                    file_sequence_number=INITIAL_SEQUENCE_NUMBER,
                    data_file=data_file,
                )
                writer.add_entry(entry)

            original_manifest = writer.to_manifest_file()

            # Create a new ManifestFile with URI path for metadata references
            return ManifestFile.from_args(
                manifest_path=manifest_uri,
                manifest_length=original_manifest.manifest_length,
                partition_spec_id=original_manifest.partition_spec_id,
                content=original_manifest.content,
                sequence_number=original_manifest.sequence_number,
                min_sequence_number=original_manifest.min_sequence_number,
                added_snapshot_id=original_manifest.added_snapshot_id,
                added_files_count=original_manifest.added_files_count,
                existing_files_count=original_manifest.existing_files_count,
                deleted_files_count=original_manifest.deleted_files_count,
                added_rows_count=original_manifest.added_rows_count,
                existing_rows_count=original_manifest.existing_rows_count,
                deleted_rows_count=original_manifest.deleted_rows_count,
                partitions=original_manifest.partitions,
                key_metadata=original_manifest.key_metadata,
            )

    def _create_snapshot(
        self, data_files: List[DataFile], properties: Optional[Dict[str, str]] = None
    ) -> Snapshot:
        """Create snapshot object.

        Args:
            data_files: List of data files
            properties: Optional table properties (used to populate snapshot summary)

        Returns:
            Snapshot object
        """
        properties = properties or {}
        total_records = sum(df.record_count for df in data_files)
        manifest_filename = f"snap-1-1-{uuid.uuid4()}.avro"

        # Build summary with standard fields + huggingface metadata
        summary_fields = {
            "added-data-files": str(len(data_files)),
            "added-records": str(total_records),
            "total-data-files": str(len(data_files)),
            "total-records": str(total_records),
        }

        # Add hf.* fields from properties to snapshot summary
        if "hf.dataset.repo" in properties:
            summary_fields["hf.dataset.repo"] = properties["hf.dataset.repo"]
        if "hf.dataset.config" in properties:
            summary_fields["hf.dataset.config"] = properties["hf.dataset.config"]
        if "hf.dataset.revision" in properties:
            revision = properties["hf.dataset.revision"]
            summary_fields["hf.dataset.revision"] = revision
            # Add short revision (first 7 chars)
            summary_fields["hf.dataset.revision.short"] = revision[:7]

        return Snapshot(  # type: ignore[call-arg]
            snapshot_id=1,
            parent_snapshot_id=None,
            sequence_number=INITIAL_SEQUENCE_NUMBER,
            timestamp_ms=1,
            manifest_list=f"{self.base_uri.rstrip('/')}/metadata/{manifest_filename}",
            summary=Summary(
                operation=Operation.APPEND,
                **summary_fields,
            ),
            schema_id=self.schema.schema_id,
        )

    def _write_manifest_list(self, snapshot: Snapshot, manifest_files: List):
        """Write manifest list file.

        Args:
            snapshot: Snapshot object
            manifest_files: List of manifest files
        """
        # Get filename from the snapshot's manifest_list path and write to staging directory
        manifest_list_filename = Path(snapshot.manifest_list).name
        manifest_list_write_path = self.metadata_dir / manifest_list_filename
        manifest_list_output = self.file_io.new_output(str(manifest_list_write_path))

        with write_manifest_list(
            format_version=2,
            output_file=manifest_list_output,
            snapshot_id=snapshot.snapshot_id,
            parent_snapshot_id=snapshot.parent_snapshot_id,
            sequence_number=snapshot.sequence_number,
            avro_compression="deflate",
        ) as manifest_list_writer:
            manifest_list_writer.add_manifests(manifest_files)

    def _create_table_metadata(
        self,
        snapshot: Snapshot,
        table_uuid: str,
        properties: Dict[str, str],
    ) -> TableMetadataV2:
        """Create table metadata object.

        Args:
            snapshot: Snapshot object
            table_uuid: UUID for the table
            properties: Table properties

        Returns:
            TableMetadataV2 object
        """
        # Create initial metadata
        metadata = new_table_metadata(
            schema=self.schema,
            partition_spec=self.partition_spec,
            sort_order=UNSORTED_SORT_ORDER,
            location=self.base_uri,
            properties=properties,
            table_uuid=uuid.UUID(table_uuid),
        )

        # Update partition spec with correct field IDs if partitioned
        if self.partition_spec != UNPARTITIONED_PARTITION_SPEC:
            # Get the reassigned schema from metadata
            reassigned_schema = metadata.schema()
            split_field = reassigned_schema.find_field("split")
            if split_field:
                # Create partition spec with correct source_id
                partition_spec_with_correct_ids = PartitionSpec(
                    PartitionField(
                        source_id=split_field.field_id,
                        field_id=1000,
                        transform=IdentityTransform(),
                        name="split",
                    ),
                    spec_id=0,
                )
                # Update metadata with correct partition spec
                metadata = TableMetadataV2(  # type: ignore[call-arg]
                    location=metadata.location,
                    table_uuid=metadata.table_uuid,
                    last_updated_ms=metadata.last_updated_ms,
                    last_column_id=metadata.last_column_id,
                    schemas=metadata.schemas,
                    current_schema_id=metadata.current_schema_id,
                    partition_specs=[partition_spec_with_correct_ids],
                    default_spec_id=0,
                    last_partition_id=1000,
                    properties=metadata.properties,
                    current_snapshot_id=None,
                    snapshots=[],
                    snapshot_log=[],
                    metadata_log=[],
                    sort_orders=metadata.sort_orders,
                    default_sort_order_id=metadata.default_sort_order_id,
                    refs={},
                    format_version=2,
                    last_sequence_number=INITIAL_SEQUENCE_NUMBER,
                )

        # Update metadata with snapshot
        return TableMetadataV2(  # type: ignore[call-arg]
            location=metadata.location,
            table_uuid=metadata.table_uuid,
            last_updated_ms=metadata.last_updated_ms,
            last_column_id=metadata.last_column_id,
            schemas=metadata.schemas,
            current_schema_id=metadata.current_schema_id,
            partition_specs=metadata.partition_specs,
            default_spec_id=metadata.default_spec_id,
            last_partition_id=metadata.last_partition_id,
            properties=metadata.properties,
            current_snapshot_id=snapshot.snapshot_id,
            snapshots=[snapshot],
            snapshot_log=[],
            metadata_log=[],
            sort_orders=metadata.sort_orders,
            default_sort_order_id=metadata.default_sort_order_id,
            refs={
                "main": SnapshotRef(  # type: ignore[call-arg]
                    snapshot_id=snapshot.snapshot_id,
                    type=SnapshotRefType.BRANCH,
                )
            },
            format_version=2,
            last_sequence_number=INITIAL_SEQUENCE_NUMBER,
        )

    def _write_metadata_file(self, metadata: TableMetadataV2) -> Path:
        """Write metadata file and version hint.

        Args:
            metadata: Table metadata object

        Returns:
            Path to the metadata file
        """
        # Write metadata file - DuckDB expects v1.metadata.json format
        metadata_file = self.metadata_dir / "v1.metadata.json"
        with open(metadata_file, "w") as f:
            f.write(metadata.model_dump_json(indent=2))

        # Write version hint - contains the version number (1)
        version_hint_file = self.metadata_dir / "version-hint.text"
        with open(version_hint_file, "w") as f:
            f.write("1")

        logger.info(f"Wrote metadata to {metadata_file}")
        return metadata_file

    def append_snapshot_from_files(
        self,
        file_infos: List[FileInfo],
        current_metadata: TableMetadataV2,
        properties: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Append a new snapshot to existing table metadata.

        This method creates a new snapshot with updated files and writes
        the new metadata version.

        Args:
            file_infos: List of FileInfo objects describing new data files
            current_metadata: Current TableMetadataV2 object
            properties: Optional updated table properties

        Returns:
            Path to the new metadata file
        """
        logger.info(f"Appending snapshot with {len(file_infos)} files")

        # Calculate next IDs
        next_snapshot_id = max(snap.snapshot_id for snap in current_metadata.snapshots) + 1
        next_sequence_number = current_metadata.last_sequence_number + 1

        # Create DataFile entries directly from FileInfo (reads parquet metadata)
        data_files = self._create_data_files_from_info(
            file_infos, current_metadata, next_sequence_number
        )

        # Merge properties first (needed for snapshot summary)
        merged_properties = {**current_metadata.properties}
        if properties:
            merged_properties.update(properties)

        # Write manifest for new files only
        new_manifest_file = self._write_manifest_with_ids(
            data_files, next_snapshot_id, next_sequence_number
        )

        # Create snapshot (with properties for summary fields)
        snapshot = self._create_snapshot_with_ids(
            data_files, next_snapshot_id, next_sequence_number, merged_properties
        )

        # Fast append: reuse previous manifests + add new manifest
        previous_manifests = self._get_previous_manifests(current_metadata)
        if previous_manifests:
            all_manifest_files = previous_manifests + [new_manifest_file]
        else:
            all_manifest_files = [new_manifest_file]

        # Write manifest list with all manifests
        self._write_manifest_list(snapshot, all_manifest_files)

        # Create updated metadata
        updated_metadata = TableMetadataV2(  # type: ignore[call-arg]
            location=current_metadata.location,
            table_uuid=current_metadata.table_uuid,
            last_updated_ms=int(time.time() * 1000),
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
                "main": SnapshotRef(  # type: ignore[call-arg]
                    snapshot_id=snapshot.snapshot_id,
                    type=SnapshotRefType.BRANCH,
                )
            },
            format_version=2,
            last_sequence_number=next_sequence_number,
        )

        # Write new metadata file
        return self._write_metadata_version(updated_metadata, next_sequence_number)

    def _write_manifest_with_ids(
        self, data_files: List[DataFile], snapshot_id: int, sequence_number: int
    ):
        """Write manifest file with specific IDs.

        Args:
            data_files: List of DataFile objects
            snapshot_id: Snapshot ID
            sequence_number: Sequence number

        Returns:
            ManifestFile object
        """
        manifest_filename = f"{uuid.uuid4()}.avro"
        manifest_write_path = self.metadata_dir / manifest_filename
        manifest_uri = f"{self.base_uri.rstrip('/')}/metadata/{manifest_filename}"
        output_file = self.file_io.new_output(str(manifest_write_path))

        with write_manifest(
            format_version=2,
            spec=self.partition_spec,
            schema=self.schema,
            output_file=output_file,
            snapshot_id=snapshot_id,
            avro_compression="deflate",
        ) as writer:
            for data_file in data_files:
                entry = ManifestEntry.from_args(
                    status=ManifestEntryStatus.ADDED,
                    snapshot_id=snapshot_id,
                    sequence_number=sequence_number,
                    file_sequence_number=sequence_number,
                    data_file=data_file,
                )
                writer.add_entry(entry)

            original_manifest = writer.to_manifest_file()

            # Create a new ManifestFile with URI path for metadata references
            return ManifestFile.from_args(
                manifest_path=manifest_uri,
                manifest_length=original_manifest.manifest_length,
                partition_spec_id=original_manifest.partition_spec_id,
                content=original_manifest.content,
                sequence_number=original_manifest.sequence_number,
                min_sequence_number=original_manifest.min_sequence_number,
                added_snapshot_id=original_manifest.added_snapshot_id,
                added_files_count=original_manifest.added_files_count,
                existing_files_count=original_manifest.existing_files_count,
                deleted_files_count=original_manifest.deleted_files_count,
                added_rows_count=original_manifest.added_rows_count,
                existing_rows_count=original_manifest.existing_rows_count,
                deleted_rows_count=original_manifest.deleted_rows_count,
                partitions=original_manifest.partitions,
                key_metadata=original_manifest.key_metadata,
            )

    def _create_snapshot_with_ids(
        self,
        data_files: List[DataFile],
        snapshot_id: int,
        sequence_number: int,
        properties: Optional[Dict[str, str]] = None,
    ) -> Snapshot:
        """Create snapshot with specific IDs.

        Args:
            data_files: List of DataFile objects
            snapshot_id: Snapshot ID
            sequence_number: Sequence number
            properties: Optional table properties (used to populate snapshot summary)

        Returns:
            Snapshot object
        """
        properties = properties or {}
        total_records = sum(df.record_count for df in data_files)

        manifest_filename = f"snap-{snapshot_id}-{sequence_number}-{uuid.uuid4()}.avro"

        # Build summary with standard fields + huggingface metadata
        summary_fields = {
            "added-data-files": str(len(data_files)),
            "added-records": str(total_records),
            "total-data-files": str(len(data_files)),
            "total-records": str(total_records),
        }

        # Add hf.* fields from properties to snapshot summary
        if "hf.dataset.repo" in properties:
            summary_fields["hf.dataset.repo"] = properties["hf.dataset.repo"]
        if "hf.dataset.config" in properties:
            summary_fields["hf.dataset.config"] = properties["hf.dataset.config"]
        if "hf.dataset.revision" in properties:
            revision = properties["hf.dataset.revision"]
            summary_fields["hf.dataset.revision"] = revision
            # Add short revision (first 7 chars)
            summary_fields["hf.dataset.revision.short"] = revision[:7]

        return Snapshot(  # type: ignore[call-arg]
            snapshot_id=snapshot_id,
            parent_snapshot_id=snapshot_id - 1,
            sequence_number=sequence_number,
            timestamp_ms=int(uuid.uuid4().time_low),
            manifest_list=f"{self.base_uri.rstrip('/')}/metadata/{manifest_filename}",
            summary=Summary(
                operation=Operation.APPEND,
                **summary_fields,
            ),
            schema_id=self.schema.schema_id,
        )

    def _write_metadata_version(self, metadata: TableMetadataV2, version: int) -> Path:
        """Write a specific metadata version.

        Args:
            metadata: TableMetadataV2 object
            version: Version number

        Returns:
            Path to the metadata file
        """
        # Write metadata file
        metadata_file = self.metadata_dir / f"v{version}.metadata.json"
        with open(metadata_file, "w") as f:
            f.write(metadata.model_dump_json(indent=2))

        # Update version hint
        version_hint_file = self.metadata_dir / "version-hint.text"
        with open(version_hint_file, "w") as f:
            f.write(str(version))

        logger.info(f"Wrote metadata version {version} to {metadata_file}")
        return metadata_file
