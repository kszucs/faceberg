"""Conversion from TableInfo to Iceberg metadata files.

This module takes TableInfo objects (created by the bridge layer) and converts them
into actual Iceberg metadata files in metadata-only mode, referencing the original
HuggingFace dataset files.
"""

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow.parquet as pq
from pyiceberg.io.pyarrow import PyArrowFileIO
from pyiceberg.manifest import (
    DataFile,
    DataFileContent,
    FileFormat,
    ManifestEntry,
    ManifestEntryStatus,
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
from faceberg.bridge import FileInfo

logger = logging.getLogger(__name__)


class IcebergMetadataWriter:
    """Writes Iceberg metadata files in metadata-only mode.

    This writer creates Iceberg metadata (manifest, manifest list, table metadata)
    that references existing HuggingFace dataset files without copying or modifying them.
    """

    def __init__(
        self,
        table_path: Path,
        schema: Schema,
        partition_spec: PartitionSpec = UNPARTITIONED_PARTITION_SPEC,
    ):
        """Initialize metadata writer.

        Args:
            table_path: Base path for the table
            schema: Iceberg schema
            partition_spec: Partition specification
        """
        self.table_path = table_path
        self.schema = schema
        self.partition_spec = partition_spec
        self.metadata_dir = table_path / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.file_io = PyArrowFileIO()

    def create_metadata_from_files(
        self,
        file_infos: List[FileInfo],
        table_uuid: str,
        properties: Optional[Dict[str, str]] = None,
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

        Returns:
            Path to the metadata file
        """
        logger.info(f"Creating Iceberg metadata for {len(file_infos)} files")

        # Step 1: Read file metadata from HuggingFace Hub
        enriched_files = self._read_file_metadata(file_infos)

        # Step 2: Create DataFile entries
        data_files = self._create_data_files(enriched_files)

        # Step 3: Write metadata files
        return self._write_metadata_files(data_files, table_uuid, properties or {})

    def _read_file_metadata(self, file_infos: List[FileInfo]) -> List[FileInfo]:
        """Read metadata from HuggingFace Hub files without downloading.

        Args:
            file_infos: List of FileInfo objects (may have size/row_count = 0)

        Returns:
            List of FileInfo objects with enriched metadata
        """
        enriched = []

        for file_info in file_infos:
            try:
                # Read metadata directly from HF Hub without downloading the file
                metadata = pq.read_metadata(file_info.path)
                row_count = metadata.num_rows

                # Use provided size if available, otherwise read from metadata
                file_size = file_info.size_bytes
                if file_size == 0:
                    # Try to get file size (this might still be 0 for some filesystems)
                    file_size = metadata.serialized_size

                enriched.append(
                    FileInfo(
                        path=file_info.path,
                        size_bytes=file_size,
                        row_count=row_count,
                        split=file_info.split,
                    )
                )

            except Exception as e:
                logger.warning(f"Could not read metadata from {file_info.path}: {e}")
                # Keep original file info if we can't read metadata
                enriched.append(file_info)

        return enriched

    def _create_data_files(self, file_infos: List[FileInfo]) -> List[DataFile]:
        """Create Iceberg DataFile entries from file information.

        Args:
            file_infos: List of FileInfo objects with metadata

        Returns:
            List of Iceberg DataFile objects
        """
        data_files = []

        for file_info in file_infos:
            # Build partition values based on the partition spec
            # Partition dict maps from partition field position to the partition value
            if self.partition_spec != UNPARTITIONED_PARTITION_SPEC and file_info.split:
                # Use position 0 for the first (and only) partition field
                partition = {0: file_info.split}
            else:
                partition = {}

            data_file = DataFile.from_args(
                content=DataFileContent.DATA,
                file_path=file_info.path,
                file_format=FileFormat.PARQUET,
                partition=partition,
                record_count=file_info.row_count,
                file_size_in_bytes=file_info.size_bytes,
                column_sizes={},
                value_counts={},
                null_value_counts={},
                nan_value_counts={},
                lower_bounds={},
                upper_bounds={},
                key_metadata=None,
                split_offsets=None,
                equality_ids=None,
                sort_order_id=None,
            )
            data_files.append(data_file)

        return data_files

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

        # Step 2: Create snapshot
        snapshot = self._create_snapshot(data_files)

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
        manifest_path = self.metadata_dir / f"{uuid.uuid4()}.avro"
        output_file = self.file_io.new_output(str(manifest_path))

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

            return writer.to_manifest_file()

    def _create_snapshot(self, data_files: List[DataFile]) -> Snapshot:
        """Create snapshot object.

        Args:
            data_files: List of data files

        Returns:
            Snapshot object
        """
        total_records = sum(df.record_count for df in data_files)
        return Snapshot(  # type: ignore[call-arg]
            snapshot_id=1,
            parent_snapshot_id=None,
            sequence_number=INITIAL_SEQUENCE_NUMBER,
            timestamp_ms=1,
            manifest_list=str(self.metadata_dir / f"snap-1-1-{uuid.uuid4()}.avro"),
            summary=Summary(
                operation=Operation.APPEND,
                **{
                    "added-data-files": str(len(data_files)),
                    "added-records": str(total_records),
                    "total-data-files": str(len(data_files)),
                    "total-records": str(total_records),
                },
            ),
            schema_id=self.schema.schema_id,
        )

    def _write_manifest_list(self, snapshot: Snapshot, manifest_files: List):
        """Write manifest list file.

        Args:
            snapshot: Snapshot object
            manifest_files: List of manifest files
        """
        manifest_list_path = Path(snapshot.manifest_list)
        manifest_list_output = self.file_io.new_output(str(manifest_list_path))

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
            location=str(self.table_path),
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
