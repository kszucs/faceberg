"""Faceberg catalog implementation with HuggingFace Hub support."""

import logging
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Set, Union
from urllib.parse import urlparse

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, HfFileSystem
from pyiceberg.catalog import Catalog, PropertiesUpdateSummary
from pyiceberg.exceptions import (
    NamespaceAlreadyExistsError,
    NamespaceNotEmptyError,
    NoSuchNamespaceError,
    NoSuchTableError,
    TableAlreadyExistsError,
)
from pyiceberg.io import load_file_io
from pyiceberg.io.fsspec import FsspecFileIO
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC, PartitionKey, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.serializers import FromInputFile
from pyiceberg.table import CommitTableRequest, CommitTableResponse, Table
from pyiceberg.table.locations import LocationProvider
from pyiceberg.table.metadata import new_table_metadata
from pyiceberg.table.sorting import UNSORTED_SORT_ORDER, SortOrder
from pyiceberg.table.update import update_table_metadata
from pyiceberg.typedef import EMPTY_DICT, Properties

from faceberg.bridge import DatasetInfo
from faceberg.config import Config, Identifier
from faceberg.config import Table as ConfigTable
from faceberg.convert import IcebergMetadataWriter

if TYPE_CHECKING:
    import pyarrow as pa

logger = logging.getLogger(__name__)


# =============================================================================
# Custom FileIO for HuggingFace Hub Support
# =============================================================================


def _hf_with_skip_cache(properties: Properties):
    """Create HfFileSystem with instance caching disabled.

    This avoids deepcopy issues when PyIceberg uses threading. HfFileSystem's cache
    can contain HfHubHTTPError exceptions that fail to deepcopy due to missing
    'response' parameter in their __init__.

    Args:
        properties: FileIO properties that may contain hf.endpoint and hf.token

    Returns:
        HfFileSystem instance with skip_instance_cache=True
    """
    return HfFileSystem(
        endpoint=properties.get("hf.endpoint"),
        token=properties.get("hf.token"),
        skip_instance_cache=True,  # Critical: prevents deepcopy issues
    )


class HfFileIO(FsspecFileIO):
    """Custom FileIO with HuggingFace Hub support and proper caching behavior.

    This FileIO implementation extends FsspecFileIO and overrides the HuggingFace
    filesystem factory to use skip_instance_cache=True. This eliminates the need
    for monkey patching HfFileSystem at the module level.

    The skip_instance_cache flag prevents HfFileSystem from caching instances in
    a way that causes deepcopy issues when PyIceberg's threading tries to copy
    filesystem instances across threads.

    Example:
        ```python
        from faceberg.catalog import HfFileIO

        io = HfFileIO(properties={
            "hf.endpoint": "https://huggingface.co",
            "hf.token": "hf_...",
        })
        ```
    """

    def __init__(self, properties: Properties):
        """Initialize HfFileIO with custom HuggingFace filesystem factory.

        Args:
            properties: FileIO properties (hf.endpoint, hf.token, etc.)
        """
        super().__init__(properties=properties)

        # Override the HuggingFace filesystem factory with our custom version
        self._scheme_to_fs["hf"] = _hf_with_skip_cache


# =============================================================================
# Custom LocationProvider for HuggingFace Hub
# =============================================================================


class HfLocationProvider(LocationProvider):
    """LocationProvider for HuggingFace datasets with configurable naming.

    Generates paths using configurable patterns:
    - Default: {split}-{index:05d}-iceberg.parquet
    - UUID mode: {split}-{uuid}-iceberg.parquet

    Pattern placeholders:
    - {split}: Partition split name (from partition key or default)
    - {index:05d}: Auto-incremented file index (zero-padded)
    - {uuid}: Random UUID for concurrent-safe writes

    Table properties:
    - write.py-location-provider.impl: faceberg.catalog.HfLocationProvider
    - huggingface.write.split: Default split name (default: "train")
    - huggingface.write.pattern: File naming pattern
    - huggingface.write.use-uuid: Use UUID instead of index (default: false)
    - huggingface.write.next-index: Starting index for file counter

    Example:
        ```python
        table_properties = {
            "write.py-location-provider.impl": "faceberg.catalog.HfLocationProvider",
            "write.data.path": "hf://datasets/my-org/my-dataset",
            "huggingface.write.split": "train",
            "huggingface.write.pattern": "{split}-{index:05d}-iceberg.parquet",
        }
        ```
    """

    def __init__(self, table_location: str, table_properties: Properties):
        """Initialize HfLocationProvider.

        Args:
            table_location: Base location of the table
            table_properties: Table properties for configuration
        """
        import itertools

        super().__init__(table_location, table_properties)
        self._default_split = table_properties.get("huggingface.write.split", "train")
        self._pattern = table_properties.get(
            "huggingface.write.pattern", "{split}-{index:05d}-iceberg.parquet"
        )
        self._use_uuid = (
            table_properties.get("huggingface.write.use-uuid", "false").lower() == "true"
        )
        # Index read from table properties (stored there by commit_table)
        self._start_index = int(table_properties.get("huggingface.write.next-index", "0"))
        self._file_counter = itertools.count(self._start_index)

    def new_data_location(
        self,
        _data_file_name: str,
        partition_key: Optional[PartitionKey] = None,
    ) -> str:
        """Generate a new data file location.

        Args:
            _data_file_name: Original filename (ignored, we use our pattern)
            partition_key: Optional partition key for extracting split

        Returns:
            Full path for the new data file
        """
        split = self._extract_split_from_partition(partition_key) or self._default_split

        if self._use_uuid:
            file_id = str(uuid.uuid4())
            filename = self._pattern.format(split=split, uuid=file_id)
        else:
            index = next(self._file_counter)
            filename = self._pattern.format(split=split, index=index)

        return f"{self.data_path}/{filename}"

    def _extract_split_from_partition(self, partition_key: Optional[PartitionKey]) -> Optional[str]:
        """Extract split value from partition key if present.

        Args:
            partition_key: Partition key that may contain a split field

        Returns:
            Split value if found, None otherwise
        """
        if partition_key is None:
            return None
        # Look for 'split' field in partition key
        # PartitionKey stores values as a Record with field access
        try:
            partition_data = partition_key.partition
            if hasattr(partition_data, "__iter__"):
                for field_name, value in partition_data.items():
                    if field_name.lower() == "split":
                        return str(value)
        except (AttributeError, TypeError):
            pass
        return None


# =============================================================================
# Utility for Staging Context Manager
# =============================================================================


class StagingContext:
    def __init__(self, staging_dir):
        self.path = Path(staging_dir)
        self.changes = []

    def add(self, path_in_repo: Union[str, Path]) -> None:
        """Record a file addition in staged changes.

        Args:
            path_in_repo: Relative path in repository
        """
        self.changes.append(("add", str(path_in_repo)))

    def delete(self, path_in_repo: Union[str, Path]) -> None:
        """Record a file/directory deletion in staged changes.

        Args:
            path_in_repo: Relative path in repository to delete
        """
        self.changes.append(("delete", str(path_in_repo)))

    def write(self, filename: str, content: str) -> None:
        """Write content to a file in the staging directory.

        Args:
            filename: Name of the file to write
            content: Content to write to the file
        """
        file_path = self.path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        self.changes.append(("add", filename))

    # TODO(kszucs): maybe add move

    def __iter__(self):
        return iter(self.changes)

    def __truediv__(self, other):
        return self.path / other


# =============================================================================
# Catalog Implementations
# =============================================================================


class BaseCatalog(Catalog):
    """Base Iceberg catalog with common catalog operations.

    A JSON-backed Iceberg catalog that stages changes in a temporary directory
    before persisting to the final location.

    Features:
    - Temporary staging for atomic operations
    - Automatic dataset discovery and table creation
    - On-demand namespace creation

    Stores table metadata locations in format:
    {
        "namespace.table_name": "path/to/metadata"
    }

    Subclasses must implement:
    - __init__: Initialize catalog-specific attributes
    - _init(): Initialize storage and save empty config
    - _commit(): Persist staged changes to final storage
    - _checkout(): Get local path to a file in the catalog
    """

    def __init__(self, name: str, uri: str, hf_token: Optional[str] = None, **properties: str):
        """Initialize base catalog.

        Args:
            name: Catalog name (identifier)
            uri: Full catalog URI (e.g., "file:///path/to/catalog" or "hf://datasets/org/repo")
            hf_token: Optional HuggingFace token for private datasets
            **properties: Additional catalog properties
        """
        # Set default properties for HuggingFace hf:// protocol support
        # These can be overridden by user-provided properties
        default_properties = {
            # Use custom HfFileIO that creates HfFileSystem with skip_instance_cache=True
            # This avoids deepcopy issues without needing monkey patching or single-threaded mode
            "py-io-impl": "faceberg.catalog.HfFileIO",
            "hf.endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        }

        # Merge default properties with user properties (user properties take precedence)
        merged_properties = {**default_properties, **properties}

        super().__init__(name=name, **merged_properties)
        self.uri = uri
        self._hf_token = hf_token

    # =========================================================================
    # Catalog initialization
    # =========================================================================

    def init(self, config: Optional[Config] = None) -> None:
        """Initialize the catalog storage.

        Creates the necessary storage structures and optionally populates it with
        tables from the provided config.
        For LocalCatalog, ensures directory exists and creates faceberg.yml.
        For RemoteCatalog, creates a new HF dataset repository with faceberg.yml.

        Args:
            config: Optional initial Config object to populate the catalog.
                   If None, creates an empty catalog.
                   If provided, adds each dataset table to the catalog.

        Raises:
            Exception: Implementation-specific exceptions (e.g., repository already exists)
        """
        # Initialize catalog storage (repository/directory creation) and save empty config
        self._init()

        # Initialize all tables from provided config
        if config is not None:
            for identifier, entry in config.items():
                # Use add_dataset to create each table
                self.add_dataset(identifier, entry.dataset, entry.config)

    # =========================================================================
    # Internal helper methods (catalog persistence and utilities)
    # =========================================================================
    # Subclasses must implement these methods

    def _init(self) -> None:
        """Initialize catalog-specific storage.

        Creates the storage (directory/repository) and saves an empty config.

        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement _init()")

    def config(self) -> Config:
        """Load catalog from storage.

        Returns:
            Config object loaded from faceberg.yml

        Raises:
            FileNotFoundError: If faceberg.yml doesn't exist
        """
        config_path = self._checkout("faceberg.yml")
        return Config.from_yaml(config_path)

    def _checkout(self, path: str) -> Path:
        """Get local path to a file in the catalog.

        For LocalCatalog, returns path in catalog directory.
        For RemoteCatalog, downloads from HF Hub and returns cached path.

        Args:
            path: Relative path within the catalog (e.g., "faceberg.yml" or "namespace/table/metadata/version-hint.text")

        Returns:
            Local path to the file

        Raises:
            FileNotFoundError: If file doesn't exist

        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement _checkout()")

    def _commit(self, changes) -> None:
        """Persist staged changes to final storage.

        Must be called within _staging() context.
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement _commit()")

    # TODO(kszucs): maybe rename it to committing or something indicating all or nothing
    @contextmanager
    def _staging(self):
        """Context manager for staging catalog operations.

        Creates a temporary staging directory and provides staging context
        for modifications. On exit, persists changes and cleans up.
        """
        staging_dir = tempfile.mkdtemp(prefix="faceberg_staging_")
        staging_ctx = StagingContext(staging_dir)
        try:
            yield staging_ctx
            self._commit(staging_ctx)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    # =========================================================================
    # Iceberg Catalog Interface (implementation of pyiceberg.catalog.Catalog)
    # =========================================================================

    # -------------------------------------------------------------------------
    # Namespace operations
    # -------------------------------------------------------------------------

    def create_namespace(
        self, namespace: Union[str, Identifier], properties: Properties = EMPTY_DICT
    ) -> None:
        """Create a namespace.

        Args:
            namespace: Namespace identifier
            properties: Namespace properties (ignored for now)

        Raises:
            NamespaceAlreadyExistsError: If namespace already exists
        """
        namespace = Identifier(namespace)
        config = self.config()
        if namespace in config:
            raise NamespaceAlreadyExistsError(f"Namespace {namespace} already exists")

        with self._staging() as staging:
            # add namespace to config
            config[namespace] = {}

            # save config since we added a namespace
            config.to_yaml(staging / "faceberg.yml")

            # create the directory in staging
            (staging / namespace.path).mkdir(parents=True, exist_ok=True)

            # stage the changes
            staging.add("faceberg.yml")
            staging.add(namespace.path)

    def drop_namespace(self, namespace: Union[str, Identifier]) -> None:
        """Drop a namespace.

        Args:
            namespace: Namespace identifier

        Raises:
            NoSuchNamespaceError: If namespace doesn't exist
            NamespaceNotEmptyError: If namespace contains tables
        """
        namespace = Identifier(namespace)
        config = self.config()
        if namespace not in config:
            raise NoSuchNamespaceError(f"Namespace {namespace} does not exist")

        # Check if namespace has tables
        namespace_obj = config[namespace]
        if isinstance(namespace_obj, dict) and len(namespace_obj) > 0:
            raise NamespaceNotEmptyError(f"Namespace {namespace} is not empty")

        with self._staging() as staging:
            # remove namespace from config
            del config[namespace]

            # save config since we removed a namespace
            config.to_yaml(staging / "faceberg.yml")

            # stage the changes
            staging.add("faceberg.yml")
            staging.delete(namespace.path)

    def list_namespaces(self, namespace: Union[str, Identifier] = ()) -> List[Identifier]:
        """List namespaces.

        Args:
            namespace: Parent namespace (not used for flat namespace structure)

        Returns:
            List of namespace identifiers
        """
        if namespace:
            raise NotImplementedError("Nested namespaces are not supported")
        config = self.config()
        return config.namespaces

    def load_namespace_properties(self, _namespace: Union[str, Identifier]) -> Properties:
        """Load namespace properties.

        Note: Namespace properties are not currently stored, returns empty dict.
        """
        return {}

    def update_namespace_properties(
        self,
        namespace: Union[str, Identifier],
        removals: Optional[Set[str]] = None,
        updates: Properties = EMPTY_DICT,
    ) -> PropertiesUpdateSummary:
        """Update namespace properties.

        Note: Namespace properties are not currently stored, returns empty summary.
        """
        return PropertiesUpdateSummary(removed=[], updated=[], missing=[])

    # -------------------------------------------------------------------------
    # Table operations
    # -------------------------------------------------------------------------

    def create_table(
        self,
        identifier: Union[str, Identifier],
        schema: Union[Schema, "pa.Schema"],
        location: Optional[str] = None,
        partition_spec: PartitionSpec = UNPARTITIONED_PARTITION_SPEC,
        sort_order: SortOrder = UNSORTED_SORT_ORDER,
        properties: Properties = EMPTY_DICT,
    ) -> Table:
        """Create a new table.

        Args:
            identifier: Table identifier
            schema: Table schema
            location: Table location (if None, uses default)
            partition_spec: Partition specification
            sort_order: Sort order
            properties: Table properties

        Returns:
            Created table

        Raises:
            TableAlreadyExistsError: If table already exists
        """
        identifier = Identifier(identifier)
        config = self.config()

        # Check if table already exists
        if identifier in config:
            raise TableAlreadyExistsError(f"Table {'.'.join(identifier)} already exists")

        if location is not None:
            raise NotImplementedError("Custom table locations are not supported yet")

        # Convert schema if needed
        schema = self._convert_schema_if_needed(schema)

        with self._staging() as staging:
            # Create table URI for metadata
            table_uri = f"{self.uri.rstrip('/')}/{'/'.join(identifier)}"

            # Create table metadata with URI location
            metadata = new_table_metadata(
                schema=schema,
                partition_spec=partition_spec,
                sort_order=sort_order,
                location=table_uri,
                properties=properties,
            )

            # Create relative paths for metadata and version hint
            metadata_dir = identifier.path / "metadata"
            metadata_path = metadata_dir / f"v{metadata.last_sequence_number}.metadata.json"
            version_hint_path = metadata_dir / "version-hint.text"

            # Write metadata and version hing files
            (staging / metadata_dir).mkdir(parents=True, exist_ok=True)
            (staging / metadata_path).write_text(metadata.model_dump_json(indent=2))
            (staging / version_hint_path).write_text(str(metadata.last_sequence_number))

            # Add table to config
            config[identifier] = ConfigTable()
            config.to_yaml(staging / "faceberg.yml")

            # Record metadata and version hint additions
            staging.add(metadata_path)
            staging.add(version_hint_path)
            staging.add("faceberg.yml")

        # Load and return table (after staging context exits and persists)
        return self.load_table(identifier)

    def load_table(self, identifier: Union[str, Identifier]) -> Table:
        """Load a table.

        Args:
            identifier: Table identifier

        Returns:
            Loaded table

        Raises:
            NoSuchTableError: If table doesn't exist
        """
        identifier = Identifier(identifier)
        config = self.config()

        # Check if table exists in catalog
        if identifier not in config:
            raise NoSuchTableError(f"Table {'.'.join(identifier)} not found")

        # Compute table location from catalog URI
        table_uri = f"{self.uri.rstrip('/')}/{'/'.join(identifier)}"

        # Use version-hint.text for Iceberg-native discovery
        version_hint_path = f"{table_uri}/metadata/version-hint.text"

        # Load FileIO to read version hint
        io = load_file_io(properties=self.properties, location=table_uri)

        # Read version hint to find current metadata version
        try:
            with io.new_input(version_hint_path).open() as f:
                version = int(f.read().decode("utf-8").strip())
        except FileNotFoundError as e:
            raise NoSuchTableError(
                f"Table {'.'.join(identifier)} metadata file not found: {version_hint_path}"
            ) from e

        # Read the actual metadata file
        metadata_uri = f"{table_uri}/metadata/v{version}.metadata.json"
        try:
            metadata_file = io.new_input(metadata_uri)
            metadata = FromInputFile.table_metadata(metadata_file)
        except FileNotFoundError as e:
            raise NoSuchTableError(
                f"Table {'.'.join(identifier)} metadata file not found: {metadata_uri}"
            ) from e

        return Table(
            identifier=identifier,
            metadata=metadata,
            metadata_location=metadata_uri,
            io=io,
            catalog=self,
        )

    def register_table(self, identifier: Union[str, Identifier], metadata_location: str) -> Table:
        """Register existing table.

        Args:
            identifier: Table identifier
            metadata_location: Path to table metadata file or directory

        Returns:
            Registered table

        Raises:
            TableAlreadyExistsError: If table already exists
        """
        raise NotImplementedError("register_table is not supported yet")

    def list_tables(self, namespace: Union[str, Identifier]) -> List[Identifier]:
        """List tables in namespace.

        Args:
            namespace: Namespace identifier

        Returns:
            List of table identifiers in namespace

        Raises:
            NoSuchNamespaceError: If namespace doesn't exist
        """
        namespace = Identifier(namespace)
        if not namespace.is_namespace():
            raise ValueError(f"Invalid namespace identifier: {namespace}")

        config = self.config()
        if namespace not in config:
            raise NoSuchNamespaceError(f"Namespace {namespace} does not exist")
        return [Identifier((*namespace, table)) for table in config[namespace]]

    def drop_table(self, identifier: Union[str, Identifier]) -> None:
        """Drop a table.

        Args:
            identifier: Table identifier

        Raises:
            NoSuchTableError: If table doesn't exist
        """
        identifier = Identifier(identifier)
        config = self.config()

        # Check if table exists
        if identifier not in config:
            raise NoSuchTableError(f"Table {'.'.join(identifier)} not found")

        with self._staging() as staging:
            # Remove table from config
            del config[identifier]
            config.to_yaml(staging / "faceberg.yml")

            # Mark the directory for deletion and stage config change
            staging.add("faceberg.yml")
            staging.delete(identifier.path)

    def rename_table(
        self, from_identifier: Union[str, Identifier], to_identifier: Union[str, Identifier]
    ) -> Table:
        """Rename a table.

        Args:
            from_identifier: Current table identifier
            to_identifier: New table identifier

        Returns:
            Renamed table

        Raises:
            NoSuchTableError: If source table doesn't exist
            TableAlreadyExistsError: If destination table already exists
        """
        from_identifier = Identifier(from_identifier)
        to_identifier = Identifier(to_identifier)

        with self._staging() as staging:
            config = self.config()

            # Check if source table exists
            if from_identifier not in config:
                raise NoSuchTableError(f"Table {'.'.join(from_identifier)} not found")

            from_entry = config[from_identifier]

            # Check if destination table already exists
            if to_identifier in config:
                raise TableAlreadyExistsError(f"Table {'.'.join(to_identifier)} already exists")

            # Get source table directory
            source_table_dir = self._checkout(from_identifier.path)
            new_table_dir = staging / to_identifier.path

            if source_table_dir.exists():
                # Copy from source to new location in staging
                shutil.copytree(source_table_dir, new_table_dir)

                # Record all files in the new table directory
                for path in new_table_dir.rglob("*"):
                    if path.is_file():
                        staging.add(path.relative_to(staging.path))

                # Add new table to config (minimal entry - table is self-contained)
                config[to_identifier] = ConfigTable(
                    dataset=from_entry.dataset,  # Preserve dataset info
                    config=from_entry.config,  # Preserve config
                )

            # Record deletion of old table directory
            staging.delete(from_identifier.path)

            # Remove old table from config
            del config[from_identifier]

            # Save config since we renamed a table (modified config structure)
            config.to_yaml(staging / "faceberg.yml")
            staging.add("faceberg.yml")

        return self.load_table(to_identifier)

    def table_exists(self, identifier: Union[str, Identifier]) -> bool:
        """Check if table exists.

        Args:
            identifier: Table identifier

        Returns:
            True if table exists
        """
        return identifier in self.config()

    def purge_table(self, _identifier: Union[str, Identifier]) -> None:
        """Drop table and delete all files."""
        raise NotImplementedError("Purge table not supported")

    def commit_table(self, request: CommitTableRequest) -> CommitTableResponse:
        """Commit table updates (data files and metadata).

        Called by pyiceberg after data files are written. This method:
        1. Validates requirements against current metadata
        2. Applies updates to create new metadata
        3. Writes new metadata file atomically
        4. Updates huggingface.write.next-index in table properties

        Args:
            request: Commit request with requirements and updates

        Returns:
            Commit response with updated metadata
        """
        # Parse identifier
        identifier = Identifier(request.identifier)

        # Load current table
        table = self.load_table(identifier)
        base_metadata = table.metadata

        # Validate requirements
        for requirement in request.requirements:
            requirement.validate(base_metadata)

        # Apply updates to create new metadata
        updated_metadata = update_table_metadata(
            base_metadata=base_metadata,
            updates=request.updates,
            enforce_validation=True,
            metadata_location=table.metadata_location,
        )

        # Persist new metadata
        with self._staging() as staging:
            # Determine new version number
            new_version = updated_metadata.last_sequence_number

            # Define metadata file and version hint paths
            metadata_dir = identifier.path / "metadata"
            metadata_file_path = metadata_dir / f"v{new_version}.metadata.json"
            version_hint_path = metadata_dir / "version-hint.text"

            # Write new metadata file and version hint
            (staging.path / metadata_dir).mkdir(parents=True, exist_ok=True)
            (staging.path / metadata_file_path).write_text(updated_metadata.model_dump_json(indent=2))
            (staging.path / version_hint_path).write_text(str(new_version))

            # Stage changes (must use relative paths from staging root)
            staging.add(metadata_file_path)
            staging.add(version_hint_path)

        # Compute metadata URI (no config update needed - table is self-contained)
        metadata_uri = f"{self.uri.rstrip('/')}/{metadata_file_path}"

        return CommitTableResponse(
            metadata=updated_metadata,
            metadata_location=metadata_uri,
        )

    # -------------------------------------------------------------------------
    # View operations (not implemented)
    # -------------------------------------------------------------------------

    def view_exists(self, identifier: Union[str, Identifier]) -> bool:
        """Check if view exists."""
        raise NotImplementedError("Views not supported")

    def drop_view(self, identifier: Union[str, Identifier]) -> None:
        """Drop a view."""
        raise NotImplementedError("Views not supported")

    def list_views(self, namespace: Union[str, Identifier]) -> List[Identifier]:
        """List views in namespace."""
        raise NotImplementedError("Views not supported")

    def create_table_transaction(
        self,
        identifier: Union[str, Identifier],
        schema: Union[Schema, "pa.Schema"],
        location: Optional[str] = None,
        partition_spec: PartitionSpec = UNPARTITIONED_PARTITION_SPEC,
        sort_order: SortOrder = UNSORTED_SORT_ORDER,
        properties: Properties = EMPTY_DICT,
    ) -> Any:
        """Create table transaction (not implemented)."""
        raise NotImplementedError("Table transactions not yet supported")

    # =========================================================================
    # Faceberg-specific Methods (dataset synchronization)
    # =========================================================================

    def sync_tables(self) -> List[Table]:
        """Sync all Iceberg tables with HuggingFace datasets in config.

        Discovers datasets and either creates new tables or updates existing ones
        with new snapshots if the dataset revision has changed.

        Returns:
            List of synced Table objects (created or updated)
        """
        # Load the config to get all tables
        config = self.config()

        tables = []
        for table in config.tables:
            table = self.sync_table(table)
            tables.append(table)

        return tables

    def add_dataset(
        self, identifier: Union[str, Identifier], dataset: str, config: str = "default"
    ) -> Table:
        """Add a dataset to the catalog and create the Iceberg table.

        This discovers the HuggingFace dataset, converts it to an Iceberg table,
        and adds it to the catalog in a single operation.

        Args:
            identifier: Table identifier in format "namespace.table"
            dataset: HuggingFace dataset in format "org/repo"
            config: Dataset configuration name (default: "default")

        Returns:
            Created Table object

        Raises:
            ValueError: If identifier format is invalid
            TableAlreadyExistsError: If table already exists with metadata
        """
        identifier = Identifier(identifier)
        catalog_config = self.config()

        if identifier in catalog_config:
            # Check if metadata files exist
            table_uri = f"{self.uri.rstrip('/')}/{'/'.join(identifier)}"
            version_hint_uri = f"{table_uri}/metadata/version-hint.text"
            io = load_file_io(properties=self.properties, location=version_hint_uri)

            try:
                with io.new_input(version_hint_uri).open():
                    pass  # File exists, table has been synced
                # Table has both config entry and metadata - it's truly a duplicate
                raise TableAlreadyExistsError(
                    f"Table {'.'.join(identifier)} already exists in catalog"
                )
            except FileNotFoundError:
                pass  # Config entry exists but no metadata - we can proceed

        # Discover dataset
        dataset_info = DatasetInfo.discover(
            repo_id=dataset,
            configs=[config],
            token=self._hf_token,
        )

        # Convert to TableInfo
        table_info = dataset_info.to_table_info(
            namespace=identifier[0],
            table_name=identifier[1],
            config=config,
            token=self._hf_token,
        )

        # Create the table with full metadata in staging context
        with self._staging() as staging:
            # Define table directory in the staging area
            # Note: IcebergMetadataWriter will create the metadata subdirectory
            table_dir = staging / identifier.path
            table_dir.mkdir(parents=True, exist_ok=True)

            # Create table URI for metadata
            table_uri = f"{self.uri.rstrip('/')}/{'/'.join(identifier)}"

            # Create metadata writer
            metadata_writer = IcebergMetadataWriter(
                table_path=table_dir,
                schema=table_info.schema,
                partition_spec=table_info.partition_spec,
                base_uri=table_uri,
            )

            # Generate table UUID
            table_uuid = str(uuid.uuid4())

            # Write Iceberg metadata files (manifest, manifest list, table metadata)
            metadata_writer.create_metadata_from_files(
                file_infos=table_info.files,
                table_uuid=table_uuid,
                properties=table_info.get_table_properties(),
            )

            # TODO(kszucs): metadata writer should return with the affected file paths
            # Record all created files in the table directory
            for path in table_dir.rglob("*"):
                if path.is_file():
                    staging.add(path.relative_to(staging.path))

            # Register table in config if not already there
            if identifier not in catalog_config:
                catalog_config[identifier] = ConfigTable(
                    dataset=table_info.source_repo,
                    config=table_info.source_config,
                )
                # Save config since we added a dataset table
                catalog_config.to_yaml(staging / "faceberg.yml")
                staging.add("faceberg.yml")

        # Load and return table after persistence
        return self.load_table(identifier)

    def sync_table(self, identifier: Union[str, Identifier]) -> Table:
        """Sync a single dataset table by adding or updating it.

        Reads dataset and config information from the catalog configuration,
        then either creates the table (if no metadata exists) or updates it
        with a new snapshot (if metadata exists).

        Args:
            identifier: Table identifier in format "namespace.table"

        Returns:
            Synced Table object (created or updated)

        Raises:
            ValueError: If identifier format is invalid or table not in config
        """
        identifier = Identifier(identifier)

        # Load config to get dataset info
        config = self.config()

        # Check if table exists in config
        if identifier not in config:
            raise ValueError(f"Table {'.'.join(identifier)} not found in config")

        config_entry = config[identifier]

        # Check if table has been synced (has metadata files)
        table_uri = f"{self.uri.rstrip('/')}/{'/'.join(identifier)}"
        version_hint_uri = f"{table_uri}/metadata/version-hint.text"

        io = load_file_io(properties=self.properties, location=table_uri)
        has_metadata = False
        try:
            with io.new_input(version_hint_uri).open():
                pass  # File exists, table has been synced
            has_metadata = True
        except Exception:
            pass  # Table hasn't been synced yet

        if not has_metadata:
            # First sync - call add_dataset which will create metadata
            # (add_dataset is now smart enough to handle existing config entries)
            return self.add_dataset(identifier, config_entry.dataset, config_entry.config)

        # Update existing table with new snapshot
        # Load table first to get old revision
        table = self.load_table(identifier)

        # Get old revision from table properties (required)
        old_revision = table.metadata.properties.get("huggingface.dataset.revision")
        if not old_revision:
            raise ValueError(
                f"Table {'.'.join(identifier)} missing 'huggingface.dataset.revision' property. "
                "This table was created before revision tracking was implemented. "
                "Please recreate the table to enable incremental sync."
            )

        # Discover dataset at current revision
        dataset_info = DatasetInfo.discover(
            repo_id=config_entry.dataset,
            configs=[config_entry.config],
            token=self._hf_token,
        )

        # Check if already up to date
        if old_revision == dataset_info.revision:
            logger.info(f"Table {'.'.join(identifier)} already at revision {old_revision}")
            return table

        # Get only new files since old revision (incremental update)
        table_info = dataset_info.to_table_info_incremental(
            namespace=identifier[0],
            table_name=identifier[1],
            config=config_entry.config,
            old_revision=old_revision,
            token=self._hf_token,
        )

        # If no new files, table is already up to date
        if not table_info.files:
            logger.info(f"No new files for {'.'.join(identifier)}")
            return table

        # Append new snapshot with only new files
        with self._staging() as staging:
            # Create local metadata directory
            metadata_dir = staging / identifier.path / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)

            # Create table URI for metadata
            table_uri = f"{self.uri.rstrip('/')}/{'/'.join(identifier)}"

            # Create metadata writer
            metadata_writer = IcebergMetadataWriter(
                table_path=metadata_dir,
                schema=table_info.schema,
                partition_spec=table_info.partition_spec,
                base_uri=table_uri,
            )

            # Append new snapshot with updated files
            metadata_writer.append_snapshot_from_files(
                file_infos=table_info.files,
                current_metadata=table.metadata,
                properties=table_info.get_table_properties(),
            )

            # Record all files in the table directory (including new manifest/metadata files)
            for path in metadata_dir.rglob("*"):
                if path.is_file():
                    staging.add(path.relative_to(staging.path))

            # Note: No need to save config since table entry hasn't changed

        # Load and return table after persistence
        return self.load_table(identifier)


class LocalCatalog(BaseCatalog):
    """Local Iceberg catalog with file system storage.

    Stores catalog metadata in a local directory.
    """

    def __init__(
        self,
        name: str,
        uri: str,
        *,
        hf_token: Optional[str] = None,
        **properties: str,
    ):
        """Initialize local catalog.

        Args:
            name: Catalog name
            uri: Catalog URI in file:// format (e.g., "file:///path/to/catalog")
            hf_token: Optional HuggingFace token for private datasets
            **properties: Additional catalog properties
        """
        # Parse file:// URI to extract path
        # Expected format: file:///path/to/catalog
        if not uri.startswith("file://"):
            raise ValueError(f"LocalCatalog requires file:// URI, got: {uri}")

        # Convert file:// URI to filesystem path
        parsed = urlparse(uri)
        path = parsed.path

        # Convert to absolute path
        self.catalog_dir = Path(path).resolve()

        # Set warehouse property to absolute path (not file:// URI)
        # PyIceberg expects warehouse to be a path, not a URI
        properties_with_warehouse = {
            "warehouse": str(self.catalog_dir),
            **properties,
        }
        super().__init__(name=name, uri=uri, hf_token=hf_token, **properties_with_warehouse)

    def _init(self) -> None:
        """Initialize local catalog storage.

        Ensures the catalog directory exists and creates an empty faceberg.yml file.
        """
        # Ensure catalog directory exists
        self.catalog_dir.mkdir(parents=True, exist_ok=True)

        # Create and save empty config
        config = Config(uri=self.uri, data={})
        config.to_yaml(self.catalog_dir / "faceberg.yml")

    def _checkout(self, path: str) -> Path:
        """Get local path to a file/directory in the catalog.

        Args:
            path: Relative path within the catalog

        Returns:
            Path in catalog directory (may or may not exist)
        """
        return self.catalog_dir / path

    def _commit(self, staging_ctx) -> None:
        """Persist staged changes to catalog directory.

        Applies staged changes to catalog_dir:
        - add: moves file from staging to catalog_dir
        - delete: removes directory from catalog_dir

        Must be called within _staging() context.
        """
        # Apply each operation
        for action, path_in_repo in staging_ctx:
            if action == "add":
                # Move file from staging to catalog_dir
                src = staging_ctx.path / path_in_repo
                dest = self.catalog_dir / path_in_repo

                # Ensure destination directory exists
                dest.parent.mkdir(parents=True, exist_ok=True)

                # Remove existing file/dir if present
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()

                # Move file from staging to catalog_dir
                shutil.move(str(src), str(dest))

            elif action == "delete":
                # Remove directory from catalog_dir
                path_to_delete = self.catalog_dir / path_in_repo
                if path_to_delete.exists():
                    if path_to_delete.is_dir():
                        shutil.rmtree(path_to_delete)
                    else:
                        path_to_delete.unlink()

            else:
                raise ValueError(f"Unknown action: {action}")


class RemoteCatalog(BaseCatalog):
    """Remote Iceberg catalog with HuggingFace Hub integration.

    Uses HuggingFace Hub for catalog storage with automatic local caching.
    All operations work on a local staging directory (cache), and changes
    are uploaded to the hub by _commit().

    Features:
    - HuggingFace Hub storage with local caching
    - Deletion tracking for removed tables
    - Atomic commits with all changes
    """

    def __init__(
        self,
        name: str,
        uri: str,
        *,
        hf_token: Optional[str] = None,
        **properties: str,
    ):
        """Initialize remote catalog.

        Args:
            name: Catalog name
            uri: HuggingFace Hub URI (e.g., "hf://datasets/org/repo" or "hf://spaces/org/repo")
            hf_token: HuggingFace authentication token (optional)
            **properties: Additional catalog properties
        """
        # Parse HuggingFace Hub URI to extract repo type and repo ID
        # Expected format: hf://{repo_type}/{org}/{repo}
        if not uri.startswith("hf://"):
            raise ValueError(f"RemoteCatalog requires hf:// URI, got: {uri}")

        # Remove "hf://" prefix and split into parts
        parts = uri[5:].split("/", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid HuggingFace URI format: {uri}. "
                f"Expected format: hf://{{repo_type}}/{{org}}/{{repo}}"
            )

        # Hub-specific attributes
        self._hf_api = HfApi(token=hf_token)

        repo_type, self._hf_repo = parts
        if repo_type == "spaces":
            self._hf_repo_type = "space"
        elif repo_type == "datasets":
            self._hf_repo_type = "dataset"
        else:
            raise ValueError(f"Unsupported repo_type in URI: {repo_type}")

        # Set warehouse property to hf:// URI for remote catalogs
        properties_with_warehouse = {"warehouse": uri, **properties}
        super().__init__(name=name, uri=uri, hf_token=hf_token, **properties_with_warehouse)

    def _init(self) -> None:
        """Initialize remote catalog storage.

        Creates a new HuggingFace repository with an empty faceberg.yml.

        Raises:
            ValueError: If repository already exists
        """
        # Create the repository
        self._hf_api.create_repo(
            repo_id=self._hf_repo,
            repo_type=self._hf_repo_type,
            space_sdk="docker" if self._hf_repo_type == "space" else None,
            exist_ok=False,
        )

        # Create and commit initial files
        with self._staging() as staging:
            # Create empty config
            config = Config(uri=self.uri, data={})
            config.to_yaml(staging / "faceberg.yml")
            staging.add("faceberg.yml")

            # For spaces, add README and Dockerfile
            if self._hf_repo_type == "space":
                spaces_dir = Path(__file__).parent / "spaces"
                staging.write("README.md", (spaces_dir / "README.md").read_text())
                staging.write("Dockerfile", (spaces_dir / "Dockerfile").read_text())

    def _commit(self, staging_ctx) -> None:
        """Persist staged changes to HuggingFace Hub.

        Uses staged changes to create atomic commit with all operations.
        Files are automatically cached by HuggingFace Hub's download mechanism.

        Must be called within _staging() context.
        """
        # Convert staging changes to HF operations
        operations = []
        for action, path_in_repo in staging_ctx:
            if action == "add":
                # File is in staging directory
                path_or_fileobj = staging_ctx.path / path_in_repo
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=path_in_repo, path_or_fileobj=str(path_or_fileobj)
                    )
                )
            elif action == "delete":
                operations.append(CommitOperationDelete(path_in_repo=path_in_repo))
            else:
                raise ValueError(f"Unknown staging action: {action}")

        # Create commit with all operations
        self._hf_api.create_commit(
            repo_id=self._hf_repo,
            repo_type=self._hf_repo_type,
            operations=operations,
            commit_message="Sync catalog metadata",
        )

    def _checkout(self, path: str) -> Path:
        """Get local path to a file in the catalog.

        Downloads file from HF Hub and returns cached path.

        Args:
            path: Relative path within the catalog

        Returns:
            Path to the file in HF cache

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        try:
            local_path = self._hf_api.hf_hub_download(
                repo_id=self._hf_repo,
                filename=path,
                repo_type=self._hf_repo_type,
            )
            return Path(local_path)
        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                raise FileNotFoundError(
                    f"File '{path}' not found in repository {self._hf_repo}. "
                    f"Initialize the catalog first by calling catalog.init()"
                ) from e
            raise


def catalog(
    uri: str, *, hf_token: Optional[str] = None, **properties: str
) -> Union[LocalCatalog, RemoteCatalog]:
    """Create a catalog instance based on URI.

    Factory function that determines catalog type from URI and creates
    the appropriate LocalCatalog or RemoteCatalog instance.

    Args:
        uri: Catalog URI
            - HuggingFace Hub: "hf://{repo_type}/org/repo" (e.g., "hf://datasets/org/repo", "hf://spaces/org/repo")
            - HuggingFace Hub (shorthand): "org/repo" (defaults to spaces repo type)
            - Local file system: "/path/to/catalog" or "file:///path/to/catalog"
        hf_token: HuggingFace API token (optional for public repos)
        **properties: Additional catalog properties

    Returns:
        LocalCatalog or RemoteCatalog instance

    Examples:
        >>> # Local catalog
        >>> cat = catalog("/path/to/catalog")
        >>> cat = catalog("file:///path/to/catalog")

        >>> # Remote catalog on HuggingFace Hub (datasets)
        >>> cat = catalog("hf://datasets/org/repo", hf_token="hf_...")
        >>> cat = catalog("org/repo", hf_token="hf_...")  # defaults to spaces

        >>> # Remote catalog on HuggingFace Hub (spaces)
        >>> cat = catalog("hf://spaces/org/repo", hf_token="hf_...")
    """
    if uri.startswith("hf://"):
        # HuggingFace Hub with explicit protocol
        # Extract repo name from URI for catalog name
        # Format: hf://{repo_type}/{org}/{repo}
        parts = uri[5:].split("/", 1)
        if len(parts) == 2:
            hf_repo = parts[1]  # org/repo
        else:
            hf_repo = uri
        return RemoteCatalog(name=hf_repo, uri=uri, hf_token=hf_token, **properties)
    elif uri.startswith("file://"):
        # Local catalog with explicit file:// protocol
        return LocalCatalog(name=uri, uri=uri, hf_token=hf_token, **properties)
    elif Path(uri).is_dir():
        # Local catalog with directory path - convert to file:// URI
        # Convert to absolute path and file:// URI
        abs_path = Path(uri).resolve()
        path_str = abs_path.as_posix()
        # file:// URI format: file:// + empty host + absolute path
        # For /path/to/file -> file:///path/to/file (3 slashes total)
        file_uri = f"file://{path_str}"
        return LocalCatalog(name=uri, uri=file_uri, hf_token=hf_token, **properties)
    else:
        # Assume it's a HuggingFace repo ID (org/repo format)
        return RemoteCatalog(name=uri, uri=f"hf://spaces/{uri}", hf_token=hf_token, **properties)


# Alias for main API
FacebergCatalog = LocalCatalog
