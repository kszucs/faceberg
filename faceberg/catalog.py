"""Faceberg catalog implementation with HuggingFace Hub support."""

import json
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download
from pyiceberg.catalog import Catalog, PropertiesUpdateSummary
from pyiceberg.io.fsspec import FsspecFileIO
from pyiceberg.exceptions import (
    NamespaceAlreadyExistsError,
    NamespaceNotEmptyError,
    NoSuchTableError,
    TableAlreadyExistsError,
)
from pyiceberg.io import load_file_io
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.serializers import FromInputFile
from pyiceberg.table import CommitTableRequest, CommitTableResponse, Table
from pyiceberg.table.metadata import new_table_metadata
from pyiceberg.table.sorting import UNSORTED_SORT_ORDER, SortOrder
from pyiceberg.typedef import EMPTY_DICT, Identifier, Properties

from faceberg.bridge import DatasetInfo, TableInfo
from faceberg.config import CatalogConfig
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
    - _load_catalog(): Load catalog from storage
    - _save_catalog(): Save catalog to staging
    - _persist_changes(): Persist staged changes to final storage
    """

    def __init__(
        self,
        uri: str,
        config: Optional[CatalogConfig] = None,
        **properties: str,
    ):
        """Initialize base catalog.

        Args:
            uri: Full catalog URI (e.g., "file:///path/to/catalog" or "hf://datasets/org/repo")
            config: Optional catalog configuration (only needed for sync operations)
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

        # Remove 'name' from properties if present (we use uri as name)
        properties_copy = {k: v for k, v in properties.items() if k != 'name'}

        # Merge default properties with user properties (user properties take precedence)
        merged_properties = {**default_properties, **properties_copy}

        super().__init__(name=uri, **merged_properties)
        self.uri = uri
        self.config = config

        # Temporary staging attributes (set within context manager)
        self._staging_dir = None
        self._catalog = None
        self._staged_changes = None  # List of CommitOperation objects

    # =========================================================================
    # Internal helper methods (catalog persistence and utilities)
    # =========================================================================
    # Subclasses must implement these methods

    def _load_catalog(self) -> Dict[str, str]:
        """Load catalog from storage.

        Always sets self._catalog and returns it.
        Subclasses must implement this method.

        Returns:
            Dictionary mapping table identifier to metadata path
        """
        raise NotImplementedError("Subclasses must implement _load_catalog()")

    def _save_catalog(self) -> None:
        """Save catalog to staging directory and record the change.

        Serializes self._catalog to staging_dir/catalog.json and appends the
        change to self._staged_changes.
        Must be called within _staging() context.
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement _save_catalog()")

    def _persist_changes(self) -> None:
        """Persist staged changes to final storage.

        Must be called within _staging() context.
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
    def _staging(self):
        """Context manager for staging catalog operations.

        Creates a temporary staging directory, loads catalog, and provides
        staging context for modifications. On exit, saves catalog, persists changes, and cleans up.

        Usage:
            with self._staging():
                # self._staging_dir is available
                # self._catalog contains loaded catalog
                # self._staged_changes tracks all modifications
                self._catalog['new.table'] = 'path/to/metadata'
                # Write metadata files to self._staging_dir
                # Append changes to self._staged_changes
        """
        # Create temporary staging directory
        temp_dir = tempfile.mkdtemp(prefix="faceberg_staging_")
        self._staging_dir = Path(temp_dir)

        # Initialize staged changes list
        self._staged_changes = []

        # Load catalog from storage (sets self._catalog)
        self._load_catalog()

        try:
            yield
        finally:
            # Save catalog to staging (serializes self._catalog to staging_dir/catalog.json)
            # This also records the catalog.json change in self._staged_changes
            self._save_catalog()

            # Persist changes to storage
            self._persist_changes()

            # Clean up
            self._catalog = None
            self._staged_changes = None
            self._staging_dir = None
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _identifier_to_str(self, identifier: Union[str, Identifier]) -> str:
        """Convert identifier to string format.

        Args:
            identifier: Table identifier (string or tuple)

        Returns:
            String in format "namespace.table_name"
        """
        if isinstance(identifier, str):
            return identifier
        return ".".join(identifier)

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
        ns_str = self._identifier_to_str(namespace)

        with self._staging():
            # Check if any tables exist in this namespace
            for table_id in self._catalog.keys():
                if table_id.startswith(ns_str + "."):
                    raise NamespaceAlreadyExistsError(f"Namespace {ns_str} already exists")

            # For JSON catalog, namespaces are implicit from table names
            # Just create the directory in staging
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
        ns_str = self._identifier_to_str(namespace)

        with self._staging():
            # Check if namespace has tables
            for table_id in self._catalog.keys():
                if table_id.startswith(ns_str + "."):
                    raise NamespaceNotEmptyError(f"Namespace {ns_str} is not empty")

            # For JSON catalog, if no tables, namespace doesn't really exist
            # But we can try to remove the directory
            ns_dir = self._staging_dir / ns_str
            if ns_dir.exists():
                ns_dir.rmdir()

    def list_namespaces(
        self, namespace: Union[str, Identifier] = ()
    ) -> List[Identifier]:
        """List namespaces.

        Args:
            namespace: Parent namespace (not used for flat namespace structure)

        Returns:
            List of namespace identifiers
        """
        # Extract unique namespaces from table identifiers
        tables = self._load_catalog()
        namespaces: Set[str] = set()
        for table_id in tables.keys():
            parts = table_id.split(".")
            if len(parts) >= 2:
                # Take all parts except the last (table name)
                ns = ".".join(parts[:-1])
                namespaces.add(ns)

        return [tuple([ns]) if "." not in ns else tuple(ns.split(".")) for ns in sorted(namespaces)]

    def load_namespace_properties(
        self, namespace: Union[str, Identifier]
    ) -> Properties:
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
        table_id = self._identifier_to_str(identifier)

        with self._staging():
            # Check if table already exists
            if table_id in self._catalog:
                raise TableAlreadyExistsError(f"Table {table_id} already exists")

            # Convert schema if needed
            schema = self._convert_schema_if_needed(schema)

            # Parse identifier
            namespace, table_name = table_id.split(".", 1)

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
                    path_in_repo=str(rel_metadata_path),
                    path_or_fileobj=str(metadata_file_path)
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
                    path_in_repo=str(rel_version_hint),
                    path_or_fileobj=str(version_hint_path)
                )
            )

            # Register in catalog with full metadata URI
            rel_path = metadata_file_path.relative_to(self._staging_dir)
            metadata_uri = f"{self.uri.rstrip('/')}/{rel_path}"
            self._catalog[table_id] = metadata_uri

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
        table_id = self._identifier_to_str(identifier)

        # Get metadata location from catalog (should be a full URI)
        tables = self._load_catalog()
        metadata_uri = tables.get(table_id)
        if not metadata_uri:
            raise NoSuchTableError(f"Table {table_id} not found")

        # Load FileIO and metadata using the URI
        # PyIceberg's FileIO handles all protocol schemes (file://, hf://, etc.)
        io = load_file_io(properties=self.properties, location=metadata_uri)

        try:
            metadata_file = io.new_input(metadata_uri)
            metadata = FromInputFile.table_metadata(metadata_file)
        except FileNotFoundError as e:
            raise NoSuchTableError(f"Table {table_id} metadata file not found: {metadata_uri}") from e

        return Table(
            identifier=(
                self.identifier_to_tuple(identifier)
                if isinstance(identifier, str)
                else identifier
            ),
            metadata=metadata,
            metadata_location=metadata_uri,
            io=io,
            catalog=self,
        )

    def register_table(
        self, identifier: Union[str, Identifier], metadata_location: str
    ) -> Table:
        """Register existing table.

        Args:
            identifier: Table identifier
            metadata_location: Path to table metadata file or directory

        Returns:
            Registered table

        Raises:
            TableAlreadyExistsError: If table already exists
        """
        table_id = self._identifier_to_str(identifier)

        with self._staging():
            if table_id in self._catalog:
                raise TableAlreadyExistsError(f"Table {table_id} already exists")

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

            # Register in catalog with full URI
            self._catalog[table_id] = metadata_uri
            # Note: No files are added to staged_changes - only catalog.json will be updated

        return self.load_table(identifier)

    def list_tables(self, namespace: Union[str, Identifier]) -> List[Identifier]:
        """List tables in namespace.

        Args:
            namespace: Namespace identifier

        Returns:
            List of table identifiers in namespace
        """
        ns_str = self._identifier_to_str(namespace)

        catalog_tables = self._load_catalog()
        tables = []
        for table_id in catalog_tables.keys():
            if table_id.startswith(ns_str + "."):
                # Extract namespace and table name
                parts = table_id.split(".")
                tables.append(tuple(parts))

        return sorted(tables)

    def drop_table(self, identifier: Union[str, Identifier]) -> None:
        """Drop a table.

        Args:
            identifier: Table identifier

        Raises:
            NoSuchTableError: If table doesn't exist
        """
        table_id = self._identifier_to_str(identifier)

        with self._staging():
            if table_id not in self._catalog:
                raise NoSuchTableError(f"Table {table_id} not found")

            # Record deletion of table directory
            namespace, table_name = table_id.rsplit(".", 1)
            table_dir = f"{namespace}/{table_name}/"
            self._staged_changes.append(
                CommitOperationDelete(path_in_repo=table_dir)
            )

            # Remove from catalog
            del self._catalog[table_id]

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
        from_id = self._identifier_to_str(from_identifier)
        to_id = self._identifier_to_str(to_identifier)

        with self._staging():
            if from_id not in self._catalog:
                raise NoSuchTableError(f"Table {from_id} not found")

            if to_id in self._catalog:
                raise TableAlreadyExistsError(f"Table {to_id} already exists")

            # Parse the namespace and table name from identifiers
            from_parts = from_id.split(".")
            to_parts = to_id.split(".")

            from_namespace = from_parts[0]
            from_table = ".".join(from_parts[1:])
            to_namespace = to_parts[0]
            to_table = ".".join(to_parts[1:])

            # Get source table directory
            source_table_dir = self._load_table_locally(from_namespace, from_table)
            new_table_dir = self._staging_dir / to_namespace / to_table

            if source_table_dir.exists():
                # Copy from source to new location in staging
                shutil.copytree(source_table_dir, new_table_dir)

                # Record all files in the new table directory
                for file_path in new_table_dir.rglob("*"):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(self._staging_dir)
                        self._staged_changes.append(
                            CommitOperationAdd(
                                path_in_repo=str(rel_path),
                                path_or_fileobj=str(file_path)
                            )
                        )

                # Update catalog with full metadata URI
                # Find the metadata file in the copied directory
                metadata_files = list(new_table_dir.glob("metadata/*.metadata.json"))
                if metadata_files:
                    # Use the latest metadata file
                    metadata_file = sorted(metadata_files)[-1]
                    rel_path = metadata_file.relative_to(self._staging_dir)
                    metadata_uri = f"{self.uri.rstrip('/')}/{rel_path}"
                    self._catalog[to_id] = metadata_uri

            # Record deletion of old table directory
            old_table_dir = f"{from_namespace}/{from_table}/"
            self._staged_changes.append(
                CommitOperationDelete(path_in_repo=old_table_dir)
            )

            # Remove old table from catalog
            del self._catalog[from_id]

        return self.load_table(to_identifier)

    def table_exists(self, identifier: Union[str, Identifier]) -> bool:
        """Check if table exists.

        Args:
            identifier: Table identifier

        Returns:
            True if table exists
        """
        table_id = self._identifier_to_str(identifier)
        tables = self._load_catalog()
        return table_id in tables

    def purge_table(self, identifier: Union[str, Identifier]) -> None:
        """Drop table and delete all files.

        Args:
            identifier: Table identifier
        """
        # For now, just drop the table (don't delete files)
        self.drop_table(identifier)

    def commit_table(
        self, request: CommitTableRequest
    ) -> CommitTableResponse:
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

    def sync(
        self,
        config: Optional[CatalogConfig] = None,
        token: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> List[Table]:
        """Sync Iceberg tables with HuggingFace datasets in config.

        Discovers datasets and either creates new tables or updates existing ones
        with new snapshots if the dataset revision has changed.

        Args:
            config: Catalog configuration defining which datasets to sync (uses self.config if not provided)
            token: HuggingFace API token (optional, uses HF_TOKEN env var if not provided)
            table_name: Specific table to sync (None for all), format: "namespace.table_name"

        Returns:
            List of synced Table objects (created or updated)

        Raises:
            ValueError: If config is not provided and not set at initialization, or if table_name is invalid
        """
        # Use provided config or fall back to instance config
        sync_config = config or self.config

        # Validate config is provided
        if sync_config is None:
            raise ValueError(
                "No config provided. Pass a CatalogConfig to sync() or provide one during catalog initialization."
            )

        created_tables = []

        # Determine which tables to process
        if table_name:
            # Parse table name (format: namespace.table_name)
            parts = table_name.split(".")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid table name: {table_name}. Expected format: namespace.table_name"
                )

            target_namespace, target_table = parts

            # Find the specific table in config
            table_config = None
            namespace_name = None
            for ns in sync_config.namespaces:
                if ns.name == target_namespace:
                    for tbl in ns.tables:
                        if tbl.name == target_table:
                            table_config = tbl
                            namespace_name = ns.name
                            break
                    break

            if table_config is None:
                raise ValueError(f"Table {table_name} not found in config")

            tables_to_process = [(namespace_name, table_config)]
        else:
            # Process all tables
            tables_to_process = [
                (ns.name, tbl)
                for ns in sync_config.namespaces
                for tbl in ns.tables
            ]

        # Create tables (each method manages its own staging)
        for namespace, table_config in tables_to_process:
            # Discover dataset (only for the specific config needed)
            dataset_info = DatasetInfo.discover(
                repo_id=table_config.dataset,
                configs=[table_config.config],  # Only discover the specific config
                token=token,
            )

            # Convert to TableInfo using explicit namespace and table name
            table_info = dataset_info.to_table_info(
                namespace=namespace,
                table_name=table_config.name,
                config=table_config.config,
                token=token,
            )

            # Sync table (create new or update existing)
            table = self._sync_table(table_info)
            if table is not None:
                created_tables.append(table)

        return created_tables

    def _sync_table(self, table_info: TableInfo) -> Optional[Table]:
        """Sync a table by creating it or checking if it needs updates.

        Args:
            table_info: TableInfo containing all metadata needed for table creation/update

        Returns:
            Table object (created), or None if no sync needed
        """
        # Check if table already exists
        if self.table_exists(table_info.identifier):
            # Table exists - check if we need to update it
            table = self.load_table(table_info.identifier)

            # Get the current revision from table properties
            current_revision = table.properties.get("huggingface.dataset.revision")
            new_revision = table_info.source_revision

            # If revisions match, no sync needed
            if current_revision == new_revision and new_revision is not None:
                return None

            # Revision changed - update the table snapshot
            return self._update_table(table_info)

        # Table doesn't exist - create it
        return self._create_table(table_info)

    def _create_table(self, table_info: TableInfo) -> Table:
        """Create Iceberg table from TableInfo using metadata-only mode.

        This method creates an Iceberg table by:
        1. Creating the namespace on-demand if it doesn't exist
        2. Creating the table metadata directory
        3. Using IcebergMetadataWriter to write Iceberg metadata files
        4. Registering the table in the catalog

        Args:
            table_info: TableInfo containing all metadata needed for table creation

        Returns:
            Created Table object

        Raises:
            TableAlreadyExistsError: If table already exists
        """
        with self._staging():
            # Create namespace directory on-demand
            namespace = table_info.namespace
            ns_dir = self._staging_dir / namespace
            ns_dir.mkdir(parents=True, exist_ok=True)

            # Check if table already exists
            if table_info.identifier in self._catalog:
                raise TableAlreadyExistsError(f"Table {table_info.identifier} already exists")

            # Parse identifier and create paths
            table_id = self._identifier_to_str(table_info.identifier)
            namespace, table_name = table_id.split(".", 1)

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
                        CommitOperationAdd(
                            path_in_repo=str(rel_path),
                            path_or_fileobj=str(file_path)
                        )
                    )

            # Register table in catalog with full metadata URI
            rel_path = metadata_file_path.relative_to(self._staging_dir)
            metadata_uri = f"{self.uri.rstrip('/')}/{rel_path}"
            self._catalog[table_info.identifier] = metadata_uri

        # Load and return table after staging context exits and persists
        return self.load_table(table_info.identifier)

    def _update_table(self, table_info: TableInfo) -> Table:
        """Update existing table with a new snapshot for the updated dataset revision.

        Args:
            table_info: TableInfo with updated revision and file information

        Returns:
            Updated Table object
        """
        # Load existing table to get current metadata
        table = self.load_table(table_info.identifier)

        with self._staging():
            # Parse identifier and create paths
            table_id = self._identifier_to_str(table_info.identifier)
            namespace, table_name = table_id.split(".", 1)

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
                        CommitOperationAdd(
                            path_in_repo=str(rel_path),
                            path_or_fileobj=str(file_path)
                        )
                    )

            # Update catalog with full metadata URI
            rel_path = metadata_file_path.relative_to(self._staging_dir)
            metadata_uri = f"{self.uri.rstrip('/')}/{rel_path}"
            self._catalog[table_info.identifier] = metadata_uri

        # Reload and return table after staging context exits and persists
        return self.load_table(table_info.identifier)


class LocalCatalog(BaseCatalog):
    """Local Iceberg catalog with file system storage.

    Stores catalog metadata in a local directory.
    """

    def __init__(
        self,
        location: str | Path,
        config: Optional[CatalogConfig] = None,
        **properties: str,
    ):
        """Initialize local catalog.

        Args:
            location: Local directory path for catalog storage
            config: Optional catalog configuration (only needed for sync operations)
            **properties: Additional catalog properties
        """
        # Convert path to absolute file:// URI
        self.catalog_dir = Path(location).resolve()
        path_str = self.catalog_dir.as_posix()
        catalog_uri = f"file:///{path_str.lstrip('/')}"

        super().__init__(uri=catalog_uri, config=config, **properties)

        # Ensure catalog directory exists
        self.catalog_dir.mkdir(parents=True, exist_ok=True)

    def _load_catalog(self) -> Dict[str, str]:
        """Load catalog from catalog directory.

        Always sets self._catalog and returns it.

        Returns:
            Dictionary mapping table identifier to metadata path
        """
        catalog_file = self.catalog_dir / "catalog.json"
        if catalog_file.exists():
            with open(catalog_file) as f:
                data = json.load(f)

            # Assert type matches
            assert data["type"] == "local", f"Expected catalog type 'local', got '{data['type']}'"
            self._catalog = data.get("tables", {})
        else:
            self._catalog = {}

        return self._catalog

    def _save_catalog(self) -> None:
        """Save catalog to staging directory and record the change.

        Serializes self._catalog to staging_dir/catalog.json and appends to staged_changes.
        Must be called within _staging() context.
        """
        if self._staging_dir is None:
            raise RuntimeError("_save_catalog() must be called within _staging() context")

        catalog_file = self._staging_dir / "catalog.json"

        # Save in new format with metadata
        data = {
            "type": "local",
            "uri": self.uri,
            "tables": self._catalog,
        }

        with open(catalog_file, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)

        # Record the change
        self._staged_changes.append(
            CommitOperationAdd(
                path_in_repo="catalog.json",
                path_or_fileobj=str(catalog_file)
            )
        )

    def _persist_changes(self) -> None:
        """Persist staged changes to catalog directory.

        Applies staged changes (CommitOperations) to catalog_dir.
        - CommitOperationAdd: moves file from staging to catalog_dir
        - CommitOperationDelete: removes directory from catalog_dir

        Must be called within _staging() context.
        """
        if self._staging_dir is None:
            raise RuntimeError("_persist_changes() must be called within _staging() context")

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
        hf_repo_id: str,
        hf_token: Optional[str] = None,
        config: Optional[CatalogConfig] = None,
        **properties: str,
    ):
        """Initialize remote catalog.

        Args:
            hf_repo_id: HuggingFace repository ID (dataset) where catalog is stored
            hf_token: HuggingFace authentication token (optional)
            config: Optional catalog configuration (only needed for sync operations)
            **properties: Additional catalog properties
        """
        # Construct HuggingFace Hub URI
        catalog_uri = f"hf://datasets/{hf_repo_id}"

        super().__init__(uri=catalog_uri, config=config, **properties)

        # Hub-specific attributes
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token
        self._loaded_revision = None  # Track revision loaded from hub

    def _load_catalog(self) -> Dict[str, str]:
        """Load catalog from HuggingFace Hub.

        Downloads catalog.json from hub using hf_hub_download and tracks the revision.
        Falls back to local cache if hub is unavailable.
        Always sets self._catalog and returns it.

        Returns:
            Dictionary mapping table identifier to metadata path
        """
        # Download catalog.json from hub
        local_path = hf_hub_download(
            repo_id=self.hf_repo_id,
            filename="catalog.json",
            repo_type="dataset",
            token=self.hf_token,
        )

        # Load the catalog
        with open(local_path) as f:
            data = json.load(f)

        # Assert type matches
        assert data["type"] == "remote", f"Expected catalog type 'remote', got '{data['type']}'"
        self._catalog = data.get("tables", {})

        # Extract revision from cached path structure (contains commit hash in path)
        # Path format: .../snapshots/{commit_hash}/catalog.json
        path_parts = Path(local_path).parts
        if "snapshots" in path_parts:
            snapshot_idx = path_parts.index("snapshots")
            self._loaded_revision = path_parts[snapshot_idx + 1]
        else:
            self._loaded_revision = None

        return self._catalog

    def _save_catalog(self) -> None:
        """Save catalog to staging directory and record the change.

        Serializes self._catalog to staging_dir/catalog.json with remote catalog format
        and appends to staged_changes.
        Must be called within _staging() context.
        """
        if self._staging_dir is None:
            raise RuntimeError("_save_catalog() must be called within _staging() context")

        catalog_file = self._staging_dir / "catalog.json"

        # Save in new format with metadata
        data = {
            "type": "remote",
            "uri": self.uri,
            "tables": self._catalog,
        }

        with open(catalog_file, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)

        # Record the change
        self._staged_changes.append(
            CommitOperationAdd(
                path_in_repo="catalog.json",
                path_or_fileobj=str(catalog_file)
            )
        )

    def _persist_changes(self) -> None:
        """Persist staged changes to HuggingFace Hub.

        Uses staged changes to create atomic commit with all operations.
        Tracks parent revision for proper concurrent update handling.
        Files are automatically cached by HuggingFace Hub's download mechanism.
        """
        # Create commit with all staged operations, using loaded revision as parent
        api = HfApi(token=self.hf_token)
        commit_info = api.create_commit(
            repo_id=self.hf_repo_id,
            repo_type="dataset",
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
        # Download the catalog.json to get the snapshot directory where the table is cached
        catalog_path = hf_hub_download(
            repo_id=self.hf_repo_id,
            filename="catalog.json",
            repo_type="dataset",
            token=self.hf_token,
            revision=self._loaded_revision,
        )
        # The catalog is in the snapshot directory, table dirs are siblings
        catalog_dir = Path(catalog_path).parent
        return catalog_dir / namespace / table_name


# Alias for main API
FacebergCatalog = LocalCatalog


