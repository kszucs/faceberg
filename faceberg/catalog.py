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
from pyiceberg.catalog import Catalog, PropertiesUpdateSummary
from pyiceberg.exceptions import (
    NamespaceAlreadyExistsError,
    NamespaceNotEmptyError,
    NoSuchTableError,
    TableAlreadyExistsError,
)
from pyiceberg.io import load_file_io
from pyiceberg.io.fsspec import FsspecFileIO
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.serializers import FromInputFile
from pyiceberg.table import CommitTableRequest, CommitTableResponse, Table
from pyiceberg.table.metadata import new_table_metadata
from pyiceberg.table.sorting import UNSORTED_SORT_ORDER, SortOrder
from pyiceberg.typedef import EMPTY_DICT, Identifier, Properties

from faceberg import database as db
from faceberg.bridge import DatasetInfo, TableInfo
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
    from huggingface_hub import HfFileSystem

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
    - _load_database(): Load catalog from storage
    - _save_database(): Save catalog to staging
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

        # Initialize empty store if none provided
        self._db = None
        self._hf_token = hf_token

        # Temporary staging attributes (set within context manager)
        self._staging_dir = None
        self._staged_changes = None  # List of CommitOperation objects

    # =========================================================================
    # Catalog initialization
    # =========================================================================

    # TODO(kszucs): allow passing a config object which is an incomplete
    # db.Catalog instance without uri and revisions set, then initialization
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
            # Create empty catalog store
            self._db = db.Catalog(uri=self.uri, namespaces={})

    # =========================================================================
    # Internal helper methods (catalog persistence and utilities)
    # =========================================================================
    # Subclasses must implement these methods

    def _init_catalog(self) -> None:
        """Initialize catalog-specific storage.

        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement _init_catalog()")

    def _load_database(self) -> db.Catalog:
        """Load catalog from storage.

        Always sets self._db and returns it.
        Subclasses must implement this method.

        Returns:
            Catalog store object
        """
        raise NotImplementedError("Subclasses must implement _load_database()")

    def _save_database(self) -> None:
        """Save catalog to staging directory and record the change.

        Serializes self._db to staging_dir/faceberg.yml and appends the
        change to self._staged_changes.
        Must be called within _staging_changes() context.
        """
        if self._staging_dir is None:
            raise RuntimeError("_save_database() must be called within _staging_changes() context")
        if self._db is None:
            raise RuntimeError("no database loaded to save")

        catalog_file = self._staging_dir / "faceberg.yml"

        # Save catalog store to YAML
        self._db.to_yaml(catalog_file)

        # Record the change
        self._staged_changes.append(
            CommitOperationAdd(path_in_repo="faceberg.yml", path_or_fileobj=str(catalog_file))
        )

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
        - Loading the database before or within the staging context
        - Modifying self._db as needed
        - Persisting changes happens automatically on exit

        Usage:
            self._load_database()  # Load database first
            with self._staging_changes():
                # self._staging_dir is available
                # self._db contains loaded catalog
                # self._staged_changes tracks all modifications
                # Use helper methods: self._db.set_table('ns', 'table', db.Table(...))
                # Write metadata files to self._staging_dir
                # Append changes to self._staged_changes
        """
        # Create temporary staging directory
        temp_dir = tempfile.mkdtemp(prefix="faceberg_staging_")
        self._staging_dir = Path(temp_dir)

        # Initialize staged changes list
        self._staged_changes = []

        try:
            # Defer execution to caller
            yield

            # Save database to staging
            self._save_database()

            # Persist changes to storage
            self._persist_changes()
        finally:
            # Clean up both the staging directory and internal state
            self._staged_changes = None
            self._staging_dir = None
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _parse_identifier(self, identifier: Union[str, Identifier]) -> tuple[str, str]:
        """Parse and validate table identifier.

        Args:
            identifier: Table identifier (string or tuple)

        Returns:
            Tuple of (namespace, table_name)

        Raises:
            ValueError: If identifier format is invalid
        """
        # Convert to string
        if isinstance(identifier, str):
            table_id = identifier
        else:
            table_id = ".".join(identifier)

        # Parse and validate
        try:
            namespace, table_name = table_id.split(".", 1)
        except ValueError:
            raise ValueError(
                f"Invalid table identifier: {table_id}. "
                f"Expected format: 'namespace.table' (e.g., 'deepmind.code_contests')"
            )

        return namespace, table_name

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

        # Load database
        self._load_database()

        with self._staging_changes():
            # Check if namespace already exists
            if ns_str in self._db.namespaces:
                raise NamespaceAlreadyExistsError(f"Namespace {ns_str} already exists")

            # Add empty namespace to store
            self._db.namespaces[ns_str] = db.Namespace(tables={})

            # Create the directory in staging
            ns_dir = self._staging_dir / ns_str
            ns_dir.mkdir(parents=True, exist_ok=True)

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

        # Load database
        self._load_database()

        with self._staging_changes():
            # Check if namespace exists and has tables
            ns = self._db.namespaces.get(ns_str)
            if ns and len(ns.tables) > 0:
                raise NamespaceNotEmptyError(f"Namespace {ns_str} is not empty")

            # Remove namespace from store if it exists
            self._db.namespaces.pop(ns_str, None)

            # Remove the directory
            ns_dir = self._staging_dir / ns_str
            if ns_dir.exists():
                ns_dir.rmdir()

    def list_namespaces(self, namespace: Union[str, Identifier] = ()) -> List[Identifier]:
        """List namespaces.

        Args:
            namespace: Parent namespace (not used for flat namespace structure)

        Returns:
            List of namespace identifiers
        """
        # Load catalog and list namespaces
        catalog = self._load_database()
        return [tuple([ns_name]) for ns_name in catalog.namespaces.keys()]

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
        namespace, table_name = self._parse_identifier(identifier)

        # Load database
        self._load_database()

        with self._staging_changes():
            # Check if table already exists
            if self._db.has_table(namespace, table_name):
                raise TableAlreadyExistsError(f"Table {namespace}.{table_name} already exists")

            # Convert schema if needed
            schema = self._convert_schema_if_needed(schema)

            # Determine table directory path
            if location is None:
                table_path = self._staging_dir / namespace / table_name
            else:
                table_path = Path(location)

            # Ensure table directory exists
            table_path.mkdir(parents=True, exist_ok=True)

            # Create table URI for metadata
            table_uri = f"{self.uri.rstrip('/')}/{namespace}/{table_name}"

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
            self._staged_changes.append(
                CommitOperationAdd(
                    path_in_repo=str(rel_metadata_path), path_or_fileobj=str(metadata_file_path)
                )
            )

            # Write version hint
            version_hint_path = table_path / "metadata" / "version-hint.text"
            with open(version_hint_path, "w") as f:
                f.write(str(metadata.last_sequence_number))

            # Record version hint change
            rel_version_hint = version_hint_path.relative_to(self._staging_dir)
            self._staged_changes.append(
                CommitOperationAdd(
                    path_in_repo=str(rel_version_hint), path_or_fileobj=str(version_hint_path)
                )
            )

            # Register in catalog with full metadata URI
            rel_path = metadata_file_path.relative_to(self._staging_dir)
            metadata_uri = f"{self.uri.rstrip('/')}/{rel_path}"

            # Add table to store
            table_entry = db.Table(
                dataset="",  # Empty for manually created tables
                uri=metadata_uri,
                revision="",  # Empty for manually created tables
                config="default",
            )

            self._db.set_table(namespace, table_name, table_entry)

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
        namespace, table_name = self._parse_identifier(identifier)

        # Get metadata location from catalog store
        catalog = self._load_database()
        table_id = f"{namespace}.{table_name}"
        try:
            store_table = catalog.get_table(namespace, table_name)
        except KeyError:
            raise NoSuchTableError(f"Table {table_id} not found")
        metadata_uri = store_table.uri
        if not metadata_uri or metadata_uri == "":
            raise NoSuchTableError(
                f"Table {table_id} has not been synced yet. Run sync() to create the table."
            )

        # Load FileIO and metadata using the URI
        # PyIceberg's FileIO handles all protocol schemes (file://, hf://, etc.)
        io = load_file_io(properties=self.properties, location=metadata_uri)

        try:
            metadata_file = io.new_input(metadata_uri)
            metadata = FromInputFile.table_metadata(metadata_file)
        except FileNotFoundError as e:
            raise NoSuchTableError(
                f"Table {table_id} metadata file not found: {metadata_uri}"
            ) from e

        return Table(
            identifier=(
                self.identifier_to_tuple(identifier) if isinstance(identifier, str) else identifier
            ),
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
        namespace, table_name = self._parse_identifier(identifier)

        # Load database
        self._load_database()

        with self._staging_changes():
            # Check if table already exists
            if self._db.has_table(namespace, table_name):
                raise TableAlreadyExistsError(f"Table {namespace}.{table_name} already exists")

            # Ensure metadata_location is a full URI
            if "://" not in metadata_location:
                # Convert path to URI
                metadata_path = Path(metadata_location)
                if metadata_path.is_absolute():
                    metadata_uri = f"file://{metadata_location}"
                else:
                    # Relative path - should not happen, but handle it
                    metadata_uri = f"{self.uri.rstrip('/')}/{metadata_location}"
            else:
                # Already a URI
                metadata_uri = metadata_location

            # Register in store
            table_entry = db.Table(
                dataset="",  # Empty for registered tables
                uri=metadata_uri,
                revision="",  # Empty for registered tables
                config="default",
            )

            self._db.set_table(namespace, table_name, table_entry)
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

        catalog = self._load_database()
        try:
            return [
                tuple([ns_str, table_name])
                for table_name in catalog.namespaces[ns_str].tables.keys()
            ]
        except KeyError:
            return []

    def drop_table(self, identifier: Union[str, Identifier]) -> None:
        """Drop a table.

        Args:
            identifier: Table identifier

        Raises:
            NoSuchTableError: If table doesn't exist
        """
        namespace, table_name = self._parse_identifier(identifier)

        # Load database
        self._load_database()

        with self._staging_changes():
            # Check if table exists and remove it
            if not self._db.delete_table(namespace, table_name):
                raise NoSuchTableError(f"Table {namespace}.{table_name} not found")

            # Record deletion of table directory
            table_dir = f"{namespace}/{table_name}/"
            self._staged_changes.append(CommitOperationDelete(path_in_repo=table_dir))

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
        from_namespace, from_table_name = self._parse_identifier(from_identifier)
        to_namespace, to_table_name = self._parse_identifier(to_identifier)

        # Load database
        self._load_database()

        with self._staging_changes():
            # Check if source table exists
            try:
                from_table = self._db.get_table(from_namespace, from_table_name)
            except KeyError:
                raise NoSuchTableError(f"Table {from_namespace}.{from_table_name} not found")

            # Check if destination table already exists
            if self._db.has_table(to_namespace, to_table_name):
                raise TableAlreadyExistsError(
                    f"Table {to_namespace}.{to_table_name} already exists"
                )

            # Get source table directory
            source_table_dir = self._load_table_locally(from_namespace, from_table_name)
            new_table_dir = self._staging_dir / to_namespace / to_table_name

            if source_table_dir.exists():
                # Copy from source to new location in staging
                shutil.copytree(source_table_dir, new_table_dir)

                # Record all files in the new table directory
                for file_path in new_table_dir.rglob("*"):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(self._staging_dir)
                        self._staged_changes.append(
                            CommitOperationAdd(
                                path_in_repo=str(rel_path), path_or_fileobj=str(file_path)
                            )
                        )

                # Update store with full metadata URI
                # Find the metadata file in the copied directory
                metadata_files = list(new_table_dir.glob("metadata/*.metadata.json"))
                if metadata_files:
                    # Use the latest metadata file
                    metadata_file = sorted(metadata_files)[-1]
                    rel_path = metadata_file.relative_to(self._staging_dir)
                    metadata_uri = f"{self.uri.rstrip('/')}/{rel_path}"

                    # Add new table to store
                    new_table_entry = db.Table(
                        dataset=from_table.dataset,  # Preserve dataset info
                        uri=metadata_uri,
                        revision=from_table.revision,  # Preserve revision
                        config=from_table.config,  # Preserve config
                    )

                    self._db.set_table(to_namespace, to_table_name, new_table_entry)

            # Record deletion of old table directory
            old_table_dir = f"{from_namespace}/{from_table_name}/"
            self._staged_changes.append(CommitOperationDelete(path_in_repo=old_table_dir))

            # Remove old table from store
            self._db.delete_table(from_namespace, from_table_name)

        return self.load_table(to_identifier)

    def table_exists(self, identifier: Union[str, Identifier]) -> bool:
        """Check if table exists.

        Args:
            identifier: Table identifier

        Returns:
            True if table exists
        """
        try:
            namespace, table_name = self._parse_identifier(identifier)
        except ValueError:
            return False
        catalog = self._load_database()
        return catalog.has_table(namespace, table_name)

    def purge_table(self, identifier: Union[str, Identifier]) -> None:
        """Drop table and delete all files.

        Args:
            identifier: Table identifier
        """
        # For now, just drop the table (don't delete files)
        self.drop_table(identifier)

    def commit_table(self, request: CommitTableRequest) -> CommitTableResponse:
        """Commit table updates.

        Args:
            request: Commit request with requirements and updates

        Returns:
            Commit response
        """
        # Simplified implementation - just update metadata
        # In a full implementation, would validate requirements and apply updates
        raise NotImplementedError("commit_table not yet implemented")

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
        """Sync Iceberg tables with HuggingFace datasets in store.

        Discovers datasets and either creates new tables or updates existing ones
        with new snapshots if the dataset revision has changed.

        Args:
            table_name: Specific table to sync (None for all), format: "namespace.table"

        Returns:
            List of synced Table objects (created or updated)

        Raises:
            ValueError: If database is not set at initialization, or if table is invalid
        """
        # Load the database once at the start
        db = self._load_database()

        # Validate store is provided
        if db is None:
            raise ValueError(
                "No database available. Initialize the catalog first by calling catalog.init()"
            )

        # Determine which tables to process
        if table_name:
            # Parse table name (format: namespace.table)
            parts = table_name.split(".")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid table name: {table_name}. Expected format: namespace.table"
                )

            target_namespace, target_table = parts

            # Find the specific table in store
            try:
                store_table = db.get_table(target_namespace, target_table)
            except KeyError:
                raise ValueError(f"Table {table_name} not found in store")

            tables_to_process = [(target_namespace, target_table, store_table)]
        else:
            # Process all tables
            tables_to_process = [
                (ns_name, table_name, table)
                for ns_name, namespace in db.namespaces.items()
                for table_name, table in namespace.tables.items()
            ]

        tables = []

        # Process each table with its own staging context
        for namespace, table_name, store_table in tables_to_process:
            # Discover dataset (only for the specific config needed)
            dataset_info = DatasetInfo.discover(
                repo_id=store_table.dataset,
                configs=[store_table.config],  # Only discover the specific config
                token=self._hf_token,
            )

            # Convert to TableInfo using explicit namespace and table name
            table_info = dataset_info.to_table_info(
                namespace=namespace,
                table_name=table_name,
                config=store_table.config,
                token=self._hf_token,
            )

            # Sync table in its own staging context
            with self._staging_changes():
                # Sync table (create new or update existing)
                metadata_uri = self._sync_dataset(table_info)
                # Update store with metadata URI
                store_table.uri = metadata_uri
                store_table.revision = table_info.source_revision or ""

            # Load the table after persistence
            table = self.load_table(table_info.identifier)
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
        namespace, table_name = self._parse_identifier(identifier)

        # Load database to check if table exists
        db = self._load_database()

        # Check if table already exists with metadata
        if db.has_table(namespace, table_name):
            store_table = db.get_table(namespace, table_name)
            if store_table.uri and store_table.uri != "":
                raise TableAlreadyExistsError(
                    f"Table {namespace}.{table_name} already exists in catalog"
                )

        # Discover dataset
        dataset_info = DatasetInfo.discover(
            repo_id=dataset,
            configs=[config],
            token=self._hf_token,
        )

        # Convert to TableInfo
        table_info = dataset_info.to_table_info(
            namespace=namespace,
            table_name=table_name,
            config=config,
            token=self._hf_token,
        )

        # Create the table with full metadata in staging context
        with self._staging_changes():
            self._add_dataset(table_info)

        # Load and return table after persistence
        return self.load_table(identifier)

    def _sync_dataset(self, table_info: TableInfo) -> str:
        """Sync a table by creating it or checking if it needs updates.

        Must be called within a _staging_changes() context.

        Args:
            table_info: TableInfo containing all metadata needed for table creation/update

        Returns:
            Metadata URI (either existing or newly created)
        """
        namespace, table_name = self._parse_identifier(table_info.identifier)

        # Check if table entry exists in database
        if self._db.has_table(namespace, table_name):
            # Get table entry to check if it has been synced (has metadata uri)
            store_table = self._db.get_table(namespace, table_name)

            # If table hasn't been synced yet (empty uri), create it
            if not store_table.uri or store_table.uri == "":
                return self._add_dataset(table_info)

            # Table has been synced - check if revision has changed
            current_revision = store_table.revision
            new_revision = table_info.source_revision

            # If revisions match, no sync needed - return existing metadata URI
            if current_revision == new_revision and new_revision is not None and current_revision:
                return store_table.uri

            # Revision changed or not set - update the table snapshot
            return self._update_dataset(table_info)
        else:
            # Table doesn't exist - create it
            return self._add_dataset(table_info)

    def _add_dataset(self, table_info: TableInfo) -> str:
        """Create Iceberg table from TableInfo using metadata-only mode.

        This method creates an Iceberg table by:
        1. Creating the namespace on-demand if it doesn't exist
        2. Creating the table metadata directory
        3. Using IcebergMetadataWriter to write Iceberg metadata files
        4. Registering the table in the catalog

        Must be called within a _staging_changes() context.

        Args:
            table_info: TableInfo containing all metadata needed for table creation

        Returns:
            Metadata URI of the created table

        Raises:
            TableAlreadyExistsError: If table already exists with metadata
        """
        # Parse identifier
        namespace, table = self._parse_identifier(table_info.identifier)

        # Create namespace directory on-demand
        ns_dir = self._staging_dir / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)

        # Check if table already exists with metadata (has been synced)
        try:
            existing_table = self._db.get_table(namespace, table)
            # Only raise error if table has already been synced (has non-empty uri)
            if existing_table.uri and existing_table.uri != "":
                raise TableAlreadyExistsError(f"Table {table_info.identifier} already exists")
            # Table exists but hasn't been synced - OK to proceed with creation
        except KeyError:
            pass  # Table doesn't exist in database, continue

        # Create local metadata directory
        metadata_path = self._staging_dir / namespace / table
        metadata_path.mkdir(parents=True, exist_ok=True)

        # Create table URI for metadata
        table_uri = f"{self.uri.rstrip('/')}/{namespace}/{table}"

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
                self._staged_changes.append(
                    CommitOperationAdd(path_in_repo=str(rel_path), path_or_fileobj=str(file_path))
                )

        # Register table in store with full metadata URI
        rel_path = metadata_file_path.relative_to(self._staging_dir)
        metadata_uri = f"{self.uri.rstrip('/')}/{rel_path}"

        # Create and set table entry
        self._db.set_table(
            namespace,
            table,
            db.Table(
                dataset=table_info.source_repo,
                uri=metadata_uri,
                revision=table_info.source_revision or "",  # Empty if revision not available
                config=table_info.source_config,
            ),
        )

        return metadata_uri

    def _update_dataset(self, table_info: TableInfo) -> str:
        """Update existing table with a new snapshot for the updated dataset revision.

        Must be called within a _staging_changes() context.

        Args:
            table_info: TableInfo with updated revision and file information

        Returns:
            Metadata URI of the updated table
        """
        # Load existing table to get current metadata
        table = self.load_table(table_info.identifier)

        # Parse identifier and create paths
        namespace, table_name = self._parse_identifier(table_info.identifier)

        # Create local metadata directory
        metadata_path = self._staging_dir / namespace / table_name
        metadata_path.mkdir(parents=True, exist_ok=True)

        # Create table URI for metadata
        table_uri = f"{self.uri.rstrip('/')}/{namespace}/{table_name}"

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
                self._staged_changes.append(
                    CommitOperationAdd(path_in_repo=str(rel_path), path_or_fileobj=str(file_path))
                )

        # Update store with full metadata URI
        rel_path = metadata_file_path.relative_to(self._staging_dir)
        metadata_uri = f"{self.uri.rstrip('/')}/{rel_path}"

        # Update table entry with new snapshot
        self._db.set_table(
            namespace,
            table_name,
            db.Table(
                dataset=table_info.source_repo,
                uri=metadata_uri,
                revision=table_info.source_revision or "",  # Empty if revision not available
                config=table_info.source_config,
            ),
        )

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

    def _load_database(self) -> db.Catalog:
        """Load catalog from catalog directory.

        Always sets self._db and returns it.
        If faceberg.yml doesn't exist, returns the existing store (for new catalogs).

        Returns:
            Catalog store object

        Raises:
            FileNotFoundError: If faceberg.yml doesn't exist and store is not initialized
        """
        catalog_file = self.catalog_dir / "faceberg.yml"
        if not catalog_file.exists():
            # For new catalogs, use the existing store
            if self._db is None:
                raise FileNotFoundError(
                    f"Catalog not found at {catalog_file}. "
                    f"Initialize the catalog first by calling catalog.init()"
                )
            return self._db

        self._db = db.Catalog.from_yaml(catalog_file)
        return self._db

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
        self._loaded_revision = None  # Track revision loaded from hub

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
            self._staged_changes.extend([
                CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=spaces_dir / "README.md"),
                CommitOperationAdd(path_in_repo="Dockerfile", path_or_fileobj=spaces_dir / "Dockerfile"),
            ])

        # Create the repository
        self._hf_api.create_repo(
            repo_id=self._hf_repo,
            repo_type=self._hf_repo_type,
            space_sdk="docker",
            exist_ok=False,
        )

    def _load_database(self) -> db.Catalog:
        """Load catalog from HuggingFace Hub.

        Downloads faceberg.yml from hub using HfApi and tracks the revision.
        Always sets self._db and returns it.
        If faceberg.yml doesn't exist, returns the existing store (for new catalogs).

        Returns:
            Catalog store object

        Raises:
            FileNotFoundError: If faceberg.yml doesn't exist and store is not initialized
        """
        # Download faceberg.yml from hub
        try:
            local_path = self._hf_api.hf_hub_download(
                repo_id=self._hf_repo,
                filename="faceberg.yml",
                repo_type=self._hf_repo_type,
            )
        except Exception as e:
            # Check if it's a "file not found" error
            if "not found" in str(e).lower() or "404" in str(e):
                # For new catalogs, use the existing store
                if self._db is None:
                    raise FileNotFoundError(
                        f"Catalog not found in repository {self._hf_repo}. "
                        f"Initialize the catalog first by calling catalog.init()"
                    ) from e
                return self._db
            # Re-raise other exceptions
            raise

        # Load the catalog
        self._db = db.Catalog.from_yaml(local_path)

        # Extract revision from cached path structure (contains commit hash in path)
        # Path format: .../snapshots/{commit_hash}/faceberg.yml
        path_parts = Path(local_path).parts
        if "snapshots" in path_parts:
            snapshot_idx = path_parts.index("snapshots")
            self._loaded_revision = path_parts[snapshot_idx + 1]
        else:
            self._loaded_revision = None

        return self._db

    def _persist_changes(self) -> None:
        """Persist staged changes to HuggingFace Hub.

        Uses staged changes to create atomic commit with all operations.
        Tracks parent revision for proper concurrent update handling.
        Files are automatically cached by HuggingFace Hub's download mechanism.
        """
        # Create commit with all staged operations, using loaded revision as parent
        commit_info = self._hf_api.create_commit(
            repo_id=self._hf_repo,
            repo_type=self._hf_repo_type,
            operations=self._staged_changes,
            commit_message="Sync catalog metadata",
            parent_commit=self._loaded_revision,  # Use tracked revision as parent
        )

        # Update loaded revision to the new commit
        self._loaded_revision = commit_info.commit_url.split("/")[-1]

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
            revision=self._loaded_revision,
        )
        # The metadata file is in the table directory
        return Path(metadata_file).parent.parent


def catalog(uri: str, *, hf_token: Optional[str] = None, **properties: str) -> Union[LocalCatalog, RemoteCatalog]:
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
