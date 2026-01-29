"""Faceberg catalog implementation with HuggingFace Hub support."""

import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Set, Union
from urllib.parse import urlparse

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from huggingface_hub import HfFileSystem
from pyiceberg.catalog import Catalog, PropertiesUpdateSummary
from pyiceberg.exceptions import (
    NamespaceAlreadyExistsError,
    NamespaceNotEmptyError,
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
from pyiceberg.typedef import EMPTY_DICT, Identifier, Properties

from faceberg.bridge import DatasetInfo, TableInfo
from faceberg.config import Config, Entry
from faceberg.convert import IcebergMetadataWriter

if TYPE_CHECKING:
    import pyarrow as pa


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
    - _load_config(): Load catalog from storage
    - _stage_config(): Save catalog to staging
    - _persist_changes(): Persist staged changes to final storage
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

        # Temporary staging attributes (set within context manager)
        self._staging_dir = None
        self._staged_changes = None  # List of CommitOperation objects

    # =========================================================================
    # Catalog initialization
    # =========================================================================

    # TODO(kszucs): allow passing a config object which is an incomplete
    # cfg.Config instance without uri and revisions set, then initialization
    # should sync as well given the provided config; this should be accessible
    # through the cli as well
    def init(self) -> None:
        """Initialize the catalog storage.

        Creates the necessary storage structures and empty faceberg.yml.
        For LocalCatalog, ensures directory exists and creates empty faceberg.yml.
        For RemoteCatalog, creates a new HF dataset repository with empty faceberg.yml.

        Raises:
            Exception: Implementation-specific exceptions (e.g., repository already exists)
        """
        with self._staging_changes():
            # Initialize catalog storage
            self._init_catalog()
            # Create empty catalog config
            config = Config(uri=self.uri, data={})
            # Save the new config
            self._stage_config(config)

    # =========================================================================
    # Internal helper methods (catalog persistence and utilities)
    # =========================================================================
    # Subclasses must implement these methods

    def _init_catalog(self) -> None:
        """Initialize catalog-specific storage.

        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement _init_catalog()")

    def _load_config(self) -> Config:
        """Load catalog from storage.

        Always sets self._config and returns it.
        Subclasses must implement this method.

        Returns:
            Catalog store object
        """
        raise NotImplementedError("Subclasses must implement _load_config()")

    def _stage_config(self, config: Config) -> None:
        """Save catalog to staging directory and record the change.

        Serializes config to staging_dir/faceberg.yml and appends the
        change to self._staged_changes.
        Must be called within _staging_changes() context.

        Args:
            config: Config object to save
        """
        if self._staging_dir is None:
            raise RuntimeError("_stage_config() must be called within _staging_changes() context")

        catalog_file = self._staging_dir / "faceberg.yml"

        # Save catalog store to YAML
        config.to_yaml(catalog_file)

        # Record the change
        self._stage_add("faceberg.yml", catalog_file)

    def _persist_changes(self) -> None:
        """Persist staged changes to final storage.

        Must be called within _staging_changes() context.
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement _persist_changes()")

    def _load_table_locally(self, namespace: str, table_name: str) -> Path:
        """Get local path where table directory can be accessed.

        Used only for operations that need to copy table files (e.g., rename_table).
        For loading tables, use URIs directly - PyIceberg's FileIO handles remote access.

        For local catalogs, returns the direct path in catalog_dir.
        For remote catalogs, downloads from HF Hub and returns cached path.

        Args:
            namespace: Table namespace
            table_name: Table name

        Returns:
            Path to locally accessible table directory

        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement _load_table_locally()")

    @contextmanager
    def _staging_changes(self):
        """Context manager for staging catalog operations.

        Creates a temporary staging directory and provides staging context
        for modifications. On exit, persists changes and cleans up.

        The caller is responsible for:
        - Loading the config before or within the staging context
        - Modifying self._config as needed
        - Calling self._stage_config() if config changes (e.g., adding/removing tables)
        - Persisting changes happens automatically on exit

        Usage:
            self._load_config()  # Load config first
            with self._staging_changes():
                # self._staging_dir is available
                # self._config contains loaded catalog
                # self._staged_changes tracks all modifications
                # Use helper methods: self._set_table('ns', 'table', cfg.Entry(...))
                # Write metadata files to self._staging_dir
                # Append changes to self._staged_changes
                # Call self._stage_config() when config is modified
        """
        # Create temporary staging directory
        temp_dir = tempfile.mkdtemp(prefix="faceberg_staging_")
        self._staging_dir = Path(temp_dir)

        # Initialize staged changes list
        self._staged_changes = []

        try:
            # Defer execution to caller
            yield

            # Persist changes to storage
            self._persist_changes()
        finally:
            # Clean up both the staging directory and internal state
            self._staged_changes = None
            self._staging_dir = None
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _normalize_identifier(self, identifier: Union[str, Identifier]) -> Identifier:
        """Convert string or tuple to validated Identifier tuple.

        Args:
            identifier: Table identifier (string or tuple)

        Returns:
            Identifier tuple (namespace, table_name)

        Raises:
            ValueError: If identifier format is invalid
        """
        if isinstance(identifier, str):
            parts = identifier.split(".")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid identifier: {identifier}. "
                    f"Expected format: 'namespace.table' (e.g., 'deepmind.code_contests')"
                )
            return tuple(parts)
        return tuple(identifier)

    def _identifier_to_path(self, identifier: Identifier) -> Path:
        """Convert identifier to relative path for file operations.

        Args:
            identifier: Identifier tuple

        Returns:
            Path object (namespace/table_name)
        """
        return Path(*identifier)

    def _stage_add(self, path_in_repo: Union[str, Path], path_or_fileobj: Union[str, Path]) -> None:
        """Record a file addition in staged changes.

        Must be called within _staging_changes() context.

        Args:
            path_in_repo: Relative path in repository
            path_or_fileobj: Path to file to add
        """
        if self._staged_changes is None:
            raise RuntimeError("_stage_add() must be called within _staging_changes() context")

        self._staged_changes.append(
            CommitOperationAdd(path_in_repo=str(path_in_repo), path_or_fileobj=str(path_or_fileobj))
        )

    def _stage_delete(self, path_in_repo: Union[str, Path]) -> None:
        """Record a file/directory deletion in staged changes.

        Must be called within _staging_changes() context.

        Args:
            path_in_repo: Relative path in repository to delete
        """
        if self._staged_changes is None:
            raise RuntimeError("_stage_delete() must be called within _staging_changes() context")

        self._staged_changes.append(CommitOperationDelete(path_in_repo=str(path_in_repo)))

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
        # Convert to string
        if isinstance(namespace, str):
            ns_str = namespace
        else:
            ns_str = ".".join(namespace)

        with self._staging_changes():
            config = self._load_config()

            # Check if namespace already exists
            if ns_str in config._data:
                raise NamespaceAlreadyExistsError(f"Namespace {ns_str} already exists")

            # Add empty namespace to config
            config._data[ns_str] = {}

            # Create the directory in staging
            ns_dir = self._staging_dir / ns_str
            ns_dir.mkdir(parents=True, exist_ok=True)

            # Save config since we added a namespace
            self._stage_config(config)

    def drop_namespace(self, namespace: Union[str, Identifier]) -> None:
        """Drop a namespace.

        Args:
            namespace: Namespace identifier

        Raises:
            NoSuchNamespaceError: If namespace doesn't exist
            NamespaceNotEmptyError: If namespace contains tables
        """
        # Convert to string
        if isinstance(namespace, str):
            ns_str = namespace
        else:
            ns_str = ".".join(namespace)

        with self._staging_changes():
            config = self._load_config()

            # Check if namespace exists and has tables
            ns_tables = config._data.get(ns_str, {})
            if ns_tables:
                raise NamespaceNotEmptyError(f"Namespace {ns_str} is not empty")

            # Remove namespace from config
            config._data.pop(ns_str, None)

            # Remove the directory
            ns_dir = self._staging_dir / ns_str
            if ns_dir.exists():
                ns_dir.rmdir()

            # Save config since we removed a namespace
            self._stage_config(config)

    def list_namespaces(self, namespace: Union[str, Identifier] = ()) -> List[Identifier]:
        """List namespaces.

        Args:
            namespace: Parent namespace (not used for flat namespace structure)

        Returns:
            List of namespace identifiers
        """
        config = self._load_config()
        return [tuple([ns_name]) for ns_name in config._data.keys()]

    def load_namespace_properties(self, namespace: Union[str, Identifier]) -> Properties:
        """Load namespace properties.

        Args:
            namespace: Namespace identifier

        Returns:
            Empty dict (properties not supported yet)
        """
        return {}

    def update_namespace_properties(
        self,
        namespace: Union[str, Identifier],
        removals: Optional[Set[str]] = None,
        updates: Properties = EMPTY_DICT,
    ) -> PropertiesUpdateSummary:
        """Update namespace properties.

        Args:
            namespace: Namespace identifier
            removals: Properties to remove
            updates: Properties to update

        Returns:
            Summary of changes (empty for now)
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
        identifier = self._normalize_identifier(identifier)

        with self._staging_changes():
            config = self._load_config()

            # Check if table already exists
            if identifier in config:
                raise TableAlreadyExistsError(f"Table {'.'.join(identifier)} already exists")

            # Convert schema if needed
            schema = self._convert_schema_if_needed(schema)

            # Determine table directory path
            if location is None:
                table_path = self._staging_dir / self._identifier_to_path(identifier)
            else:
                table_path = Path(location)

            # Ensure table directory exists
            table_path.mkdir(parents=True, exist_ok=True)

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

            # Write metadata file
            metadata_file_path = (
                table_path / "metadata" / f"v{metadata.last_sequence_number}.metadata.json"
            )
            metadata_file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(metadata_file_path, "w") as f:
                f.write(metadata.model_dump_json(indent=2))

            # Record metadata file change
            rel_metadata_path = metadata_file_path.relative_to(self._staging_dir)
            self._stage_add(rel_metadata_path, metadata_file_path)

            # Write version hint
            version_hint_path = table_path / "metadata" / "version-hint.text"
            with open(version_hint_path, "w") as f:
                f.write(str(metadata.last_sequence_number))

            # Record version hint change
            rel_version_hint = version_hint_path.relative_to(self._staging_dir)
            self._stage_add(rel_version_hint, version_hint_path)

            # Add table to config (minimal entry - table is self-contained via metadata files)
            config[identifier] = Entry(
                dataset="",  # Empty for manually created tables
                config="default",
            )

            # Save config since we added a table
            self._stage_config(config)

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
        identifier = self._normalize_identifier(identifier)
        config = self._load_config()

        # Check if table exists in catalog
        if identifier not in config:
            raise NoSuchTableError(f"Table {'.'.join(identifier)} not found")

        # Compute table location from catalog URI
        table_location = f"{self.uri.rstrip('/')}/{'/'.join(identifier)}"

        # Use version-hint.text for Iceberg-native discovery
        version_hint_path = f"{table_location}/metadata/version-hint.text"

        # Load FileIO to read version hint
        io = load_file_io(properties=self.properties, location=table_location)

        try:
            # Read version hint to find current metadata version
            with io.new_input(version_hint_path).open() as f:
                version = int(f.read().decode("utf-8").strip())
            metadata_uri = f"{table_location}/metadata/v{version}.metadata.json"
        except Exception:
            # Fallback: try v1.metadata.json if version hint doesn't exist
            metadata_uri = f"{table_location}/metadata/v1.metadata.json"

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
        identifier = self._normalize_identifier(identifier)

        with self._staging_changes():
            config = self._load_config()

            # Check if table already exists
            if identifier in config:
                raise TableAlreadyExistsError(f"Table {'.'.join(identifier)} already exists")

            # Register in config (minimal entry - table is self-contained via metadata files)
            # Note: We no longer store metadata_location in the config
            config[identifier] = Entry(
                dataset="",  # Empty for registered tables
                config="default",
            )

            # Save config since we registered a table
            self._stage_config(config)
            # Note: No files are added to staged_changes - only faceberg.yml will be updated

        return self.load_table(identifier)

    def list_tables(self, namespace: Union[str, Identifier]) -> List[Identifier]:
        """List tables in namespace.

        Args:
            namespace: Namespace identifier

        Returns:
            List of table identifiers in namespace
        """
        # Convert to string
        if isinstance(namespace, str):
            ns_str = namespace
        else:
            ns_str = ".".join(namespace)

        config = self._load_config()
        ns_tables = config._data.get(ns_str, {})
        return [tuple([ns_str, table_name]) for table_name in ns_tables.keys()]

    def drop_table(self, identifier: Union[str, Identifier]) -> None:
        """Drop a table.

        Args:
            identifier: Table identifier

        Raises:
            NoSuchTableError: If table doesn't exist
        """
        identifier = self._normalize_identifier(identifier)

        with self._staging_changes():
            config = self._load_config()

            # Check if table exists and remove it
            if identifier not in config:
                raise NoSuchTableError(f"Table {'.'.join(identifier)} not found")

            del config[identifier]

            # Record deletion of table directory
            table_dir = f"{'/'.join(identifier)}/"
            self._stage_delete(table_dir)

            # Save config since we dropped a table
            self._stage_config(config)

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
        from_identifier = self._normalize_identifier(from_identifier)
        to_identifier = self._normalize_identifier(to_identifier)

        with self._staging_changes():
            config = self._load_config()

            # Check if source table exists
            if from_identifier not in config:
                raise NoSuchTableError(f"Table {'.'.join(from_identifier)} not found")

            from_entry = config[from_identifier]

            # Check if destination table already exists
            if to_identifier in config:
                raise TableAlreadyExistsError(f"Table {'.'.join(to_identifier)} already exists")

            # Get source table directory
            source_table_dir = self._load_table_locally(*from_identifier)
            new_table_dir = self._staging_dir / self._identifier_to_path(to_identifier)

            if source_table_dir.exists():
                # Copy from source to new location in staging
                shutil.copytree(source_table_dir, new_table_dir)

                # Record all files in the new table directory
                for file_path in new_table_dir.rglob("*"):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(self._staging_dir)
                        self._stage_add(rel_path, file_path)

                # Add new table to config (minimal entry - table is self-contained)
                config[to_identifier] = Entry(
                    dataset=from_entry.dataset,  # Preserve dataset info
                    config=from_entry.config,  # Preserve config
                )

            # Record deletion of old table directory
            old_table_dir = f"{'/'.join(from_identifier)}/"
            self._stage_delete(old_table_dir)

            # Remove old table from config
            del config[from_identifier]

            # Save config since we renamed a table (modified config structure)
            self._stage_config(config)

        return self.load_table(to_identifier)

    def table_exists(self, identifier: Union[str, Identifier]) -> bool:
        """Check if table exists.

        Args:
            identifier: Table identifier

        Returns:
            True if table exists
        """
        try:
            identifier = self._normalize_identifier(identifier)
        except ValueError:
            return False
        config = self._load_config()
        return identifier in config

    def purge_table(self, identifier: Union[str, Identifier]) -> None:
        """Drop table and delete all files.

        Args:
            identifier: Table identifier
        """
        # For now, just drop the table (don't delete files)
        self.drop_table(identifier)

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
        from pyiceberg.table.update import update_table_metadata

        # Parse identifier
        identifier = self._normalize_identifier(request.identifier)

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
        with self._staging_changes():
            # Create metadata directory in staging
            metadata_path = self._staging_dir / self._identifier_to_path(identifier) / "metadata"
            metadata_path.mkdir(parents=True, exist_ok=True)

            # Determine new version number
            new_version = updated_metadata.last_sequence_number

            # Write new metadata file
            metadata_file = metadata_path / f"v{new_version}.metadata.json"
            with open(metadata_file, "w") as f:
                f.write(updated_metadata.model_dump_json(indent=2))

            # Record metadata file change
            rel_metadata = metadata_file.relative_to(self._staging_dir)
            self._stage_add(rel_metadata, metadata_file)

            # Write version hint
            version_hint = metadata_path / "version-hint.text"
            with open(version_hint, "w") as f:
                f.write(str(new_version))

            rel_hint = version_hint.relative_to(self._staging_dir)
            self._stage_add(rel_hint, version_hint)

            # Compute metadata URI (no config update needed - table is self-contained)
            metadata_uri = f"{self.uri.rstrip('/')}/{rel_metadata}"

        return CommitTableResponse(
            metadata=updated_metadata,
            metadata_location=metadata_uri,
        )

    # -------------------------------------------------------------------------
    # View operations (not implemented)
    # -------------------------------------------------------------------------

    def view_exists(self, identifier: Union[str, Identifier]) -> bool:
        """Check if view exists."""
        return False

    def drop_view(self, identifier: Union[str, Identifier]) -> None:
        """Drop a view."""
        raise NotImplementedError("Views not supported")

    def list_views(self, namespace: Union[str, Identifier]) -> List[Identifier]:
        """List views in namespace."""
        return []

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

    def sync_datasets(self, table_name: Optional[str] = None) -> List[Table]:
        """Sync Iceberg tables with HuggingFace datasets in config.

        Discovers datasets and either creates new tables or updates existing ones
        with new snapshots if the dataset revision has changed.

        Args:
            table_name: Specific table to sync (None for all), format: "namespace.table"

        Returns:
            List of synced Table objects (created or updated)

        Raises:
            ValueError: If table is invalid
        """
        # Load the config once at the start
        config = self._load_config()

        # Determine which tables to process
        if table_name:
            # Parse table name (format: namespace.table)
            identifier = self._normalize_identifier(table_name)

            # Find the specific table in config
            if identifier not in config:
                raise ValueError(f"Table {table_name} not found in config")

            config_entry = config[identifier]
            tables_to_process = [(identifier, config_entry)]
        else:
            # Process all tables
            tables_to_process = [(identifier, config[identifier]) for identifier in config]

        tables = []

        # Process each table with its own staging context
        for identifier, config_entry in tables_to_process:
            # Discover dataset (only for the specific config needed)
            dataset_info = DatasetInfo.discover(
                repo_id=config_entry.dataset,
                configs=[config_entry.config],  # Only discover the specific config
                token=self._hf_token,
            )

            # Convert to TableInfo using explicit identifier
            table_info = dataset_info.to_table_info(
                namespace=identifier[0],
                table_name=identifier[1],
                config=config_entry.config,
                token=self._hf_token,
            )

            # Sync table in its own staging context
            with self._staging_changes():
                # Load config inside staging context
                staging_config = self._load_config()
                # Sync table (create new or update existing)
                # Metadata is self-contained in Iceberg files (version-hint.text, snapshots)
                self._sync_dataset(staging_config, table_info)

            # Load the table after persistence
            table = self.load_table(identifier)
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
            TableAlreadyExistsError: If table already exists
        """
        identifier = self._normalize_identifier(identifier)

        # Load config to check if table exists
        catalog_config = self._load_config()

        # Check if table already exists
        if identifier in catalog_config:
            raise TableAlreadyExistsError(f"Table {'.'.join(identifier)} already exists in catalog")

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
        with self._staging_changes():
            staging_config = self._load_config()
            self._add_dataset(staging_config, table_info)

        # Load and return table after persistence
        return self.load_table(identifier)

    def _sync_dataset(self, config: Config, table_info: TableInfo) -> str:
        """Sync a table by creating it or checking if it needs updates.

        Must be called within a _staging_changes() context.

        Args:
            config: Config object loaded in staging context
            table_info: TableInfo containing all metadata needed for table creation/update

        Returns:
            Metadata URI (either existing or newly created)
        """
        identifier = self._normalize_identifier(table_info.identifier)

        # Check if table entry exists in config
        if identifier in config:
            # Table exists in config - check if it has been synced (has metadata files)
            table_location = f"{self.uri.rstrip('/')}/{'/'.join(identifier)}"
            version_hint_path = f"{table_location}/metadata/version-hint.text"

            # Check if table has been synced by looking for version-hint.text
            io = load_file_io(properties=self.properties, location=table_location)
            try:
                with io.new_input(version_hint_path).open():
                    pass  # File exists, table has been synced
                # Table has been synced - update it with new snapshot
                return self._update_dataset(table_info)
            except Exception:
                # Table hasn't been synced yet - create it
                return self._add_dataset(config, table_info)
        else:
            # Table doesn't exist - create it
            return self._add_dataset(config, table_info)

    def _add_dataset(self, config: Config, table_info: TableInfo) -> str:
        """Create Iceberg table from TableInfo using metadata-only mode.

        This method creates an Iceberg table by:
        1. Creating the namespace on-demand if it doesn't exist
        2. Creating the table metadata directory
        3. Using IcebergMetadataWriter to write Iceberg metadata files
        4. Registering the table in the catalog

        Must be called within a _staging_changes() context.

        Args:
            config: Config object loaded in staging context
            table_info: TableInfo containing all metadata needed for table creation

        Returns:
            Metadata URI of the created table

        Raises:
            TableAlreadyExistsError: If table already exists with metadata
        """
        # Parse identifier
        identifier = self._normalize_identifier(table_info.identifier)

        # Create namespace directory on-demand
        ns_dir = self._staging_dir / identifier[0]
        ns_dir.mkdir(parents=True, exist_ok=True)

        # Create local metadata directory
        metadata_path = self._staging_dir / self._identifier_to_path(identifier)
        metadata_path.mkdir(parents=True, exist_ok=True)

        # Create table URI for metadata
        table_uri = f"{self.uri.rstrip('/')}/{'/'.join(identifier)}"

        # Create metadata writer
        metadata_writer = IcebergMetadataWriter(
            table_path=metadata_path,
            schema=table_info.schema,
            partition_spec=table_info.partition_spec,
            base_uri=table_uri,
        )

        # Generate table UUID
        table_uuid = str(uuid.uuid4())

        # Write Iceberg metadata files (manifest, manifest list, table metadata)
        metadata_file_path = metadata_writer.create_metadata_from_files(
            file_infos=table_info.files,
            table_uuid=table_uuid,
            properties=table_info.get_table_properties(),
        )

        # Record all created files in the table directory
        for file_path in metadata_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(self._staging_dir)
                self._stage_add(rel_path, file_path)

        # Register table in config with full metadata URI
        rel_path = metadata_file_path.relative_to(self._staging_dir)
        metadata_uri = f"{self.uri.rstrip('/')}/{rel_path}"

        # Create and set table entry
        config[identifier] = Entry(
            dataset=table_info.source_repo,
            config=table_info.source_config,
        )

        # Save config since we added a dataset table
        self._stage_config(config)

        return metadata_uri

    def _update_dataset(self, table_info: TableInfo) -> str:
        """Update existing table with a new snapshot for the updated dataset revision.

        Must be called within a _staging_changes() context.

        Args:
            table_info: TableInfo with updated revision and file information

        Returns:
            Metadata URI of the updated table
        """
        # Parse identifier and load existing table
        identifier = self._normalize_identifier(table_info.identifier)
        table = self.load_table(identifier)

        # Create local metadata directory
        metadata_path = self._staging_dir / self._identifier_to_path(identifier)
        metadata_path.mkdir(parents=True, exist_ok=True)

        # Create table URI for metadata
        table_uri = f"{self.uri.rstrip('/')}/{'/'.join(identifier)}"

        # Create metadata writer
        metadata_writer = IcebergMetadataWriter(
            table_path=metadata_path,
            schema=table_info.schema,
            partition_spec=table_info.partition_spec,
            base_uri=table_uri,
        )

        # Append new snapshot with updated files
        metadata_file_path = metadata_writer.append_snapshot_from_files(
            file_infos=table_info.files,
            current_metadata=table.metadata,
            properties=table_info.get_table_properties(),
        )

        # Record all files in the table directory (including new manifest/metadata files)
        for file_path in metadata_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(self._staging_dir)
                self._stage_add(rel_path, file_path)

        # Update config with full metadata URI
        rel_path = metadata_file_path.relative_to(self._staging_dir)
        metadata_uri = f"{self.uri.rstrip('/')}/{rel_path}"

        # Note: No need to save config since table entry (dataset, config) hasn't changed

        return metadata_uri


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

    def _init_catalog(self) -> None:
        """Initialize local catalog storage.

        Ensures the catalog directory exists and creates an empty faceberg.yml file.
        """
        # Ensure catalog directory exists
        self.catalog_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> Config:
        """Load catalog from catalog directory.

        Returns:
            Config object

        Raises:
            FileNotFoundError: If faceberg.yml doesn't exist
        """
        catalog_file = self.catalog_dir / "faceberg.yml"
        if not catalog_file.exists():
            raise FileNotFoundError(
                f"Catalog not found at {catalog_file}. "
                f"Initialize the catalog first by calling catalog.init()"
            )

        return Config.from_yaml(catalog_file)

    def _persist_changes(self) -> None:
        """Persist staged changes to catalog directory.

        Applies staged changes (CommitOperations) to catalog_dir.
        - CommitOperationAdd: moves file from staging to catalog_dir
        - CommitOperationDelete: removes directory from catalog_dir

        Must be called within _staging_changes() context.
        """
        if self._staging_dir is None:
            raise RuntimeError(
                "_persist_changes() must be called within _staging_changes() context"
            )

        # Apply each operation
        for operation in self._staged_changes:
            if isinstance(operation, CommitOperationAdd):
                # Move file from staging to catalog_dir
                src = Path(operation.path_or_fileobj)
                dest = self.catalog_dir / operation.path_in_repo

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

            elif isinstance(operation, CommitOperationDelete):
                # Remove directory from catalog_dir
                path_to_delete = self.catalog_dir / operation.path_in_repo
                if path_to_delete.exists():
                    if path_to_delete.is_dir():
                        shutil.rmtree(path_to_delete)
                    else:
                        path_to_delete.unlink()

            else:
                raise ValueError(f"Unknown CommitOperation type: {type(operation)}")

    def _load_table_locally(self, namespace: str, table_name: str) -> Path:
        """Get local path where table directory can be accessed.

        Returns the direct path in catalog_dir (no copying needed).

        Args:
            namespace: Table namespace
            table_name: Table name

        Returns:
            Path to table directory in catalog_dir
        """
        return self.catalog_dir / namespace / table_name


class RemoteCatalog(BaseCatalog):
    """Remote Iceberg catalog with HuggingFace Hub integration.

    Uses HuggingFace Hub for catalog storage with automatic local caching.
    All operations work on a local staging directory (cache), and changes
    are uploaded to the hub by _persist_changes().

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
        self._hf_repo_type, self._hf_repo = parts
        if self._hf_repo_type == "spaces":
            self._hf_repo_type = "space"

        # Set warehouse property to hf:// URI for remote catalogs
        properties_with_warehouse = {"warehouse": uri, **properties}
        super().__init__(name=name, uri=uri, hf_token=hf_token, **properties_with_warehouse)

    def _init_catalog(self) -> None:
        """Initialize remote catalog storage.

        Creates a new HuggingFace dataset repository with an empty faceberg.yml.

        Raises:
            ValueError: If repository already exists
        """
        if self._hf_repo_type == "space":
            # Initialize Spaces repository with README and Dockerfile
            spaces_dir = Path(__file__).parent / "spaces"
            self._stage_add("README.md", spaces_dir / "README.md")
            self._stage_add("Dockerfile", spaces_dir / "Dockerfile")

        # Create the repository
        self._hf_api.create_repo(
            repo_id=self._hf_repo,
            repo_type=self._hf_repo_type,
            space_sdk="docker",
            exist_ok=False,
        )

    def _load_config(self) -> Config:
        """Load catalog from HuggingFace Hub.

        Downloads faceberg.yml from hub using HfApi.

        Returns:
            Config object

        Raises:
            FileNotFoundError: If faceberg.yml doesn't exist
        """
        # Download faceberg.yml from the latest revision
        try:
            local_path = self._hf_api.hf_hub_download(
                repo_id=self._hf_repo,
                filename="faceberg.yml",
                repo_type=self._hf_repo_type,
            )
        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                raise FileNotFoundError(
                    f"Catalog not found in repository {self._hf_repo}. "
                    f"Initialize the catalog first by calling catalog.init()"
                ) from e
            raise

        # Load and return the catalog
        return Config.from_yaml(local_path)

    def _persist_changes(self) -> None:
        """Persist staged changes to HuggingFace Hub.

        Uses staged changes to create atomic commit with all operations.
        Files are automatically cached by HuggingFace Hub's download mechanism.
        """
        # Create commit with all staged operations
        self._hf_api.create_commit(
            repo_id=self._hf_repo,
            repo_type=self._hf_repo_type,
            operations=self._staged_changes,
            commit_message="Sync catalog metadata",
        )

    def _load_table_locally(self, namespace: str, table_name: str) -> Path:
        """Get local path where table directory can be accessed.

        Downloads table directory from HF Hub and returns cached path.

        Args:
            namespace: Table namespace
            table_name: Table name

        Returns:
            Path to table directory in HF cache
        """
        # Download the table's metadata directory containing iceberg metadata files
        # Download one file from the table metadata directory to get the cached path
        metadata_file = self._hf_api.hf_hub_download(
            repo_id=self._hf_repo,
            filename=f"{namespace}/{table_name}/metadata/version-hint.text",
            repo_type=self._hf_repo_type,
        )
        # The metadata file is in the table directory
        return Path(metadata_file).parent.parent


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
        file_uri = f"file:///{path_str.lstrip('/')}"
        return LocalCatalog(name=uri, uri=file_uri, hf_token=hf_token, **properties)
    else:
        # Assume it's a HuggingFace repo ID (org/repo format)
        return RemoteCatalog(name=uri, uri=f"hf://spaces/{uri}", hf_token=hf_token, **properties)


# Alias for main API
FacebergCatalog = LocalCatalog
