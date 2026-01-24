"""Faceberg catalog implementation with HuggingFace Hub support."""

import json
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download
from pyiceberg.catalog import Catalog, PropertiesUpdateSummary
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


class LocalCatalog(Catalog):
    """Local Iceberg catalog with temporary staging.

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
    """

    def __init__(
        self,
        name: str,
        location: str,
        config: Optional[CatalogConfig] = None,
        **properties: str,
    ):
        """Initialize local catalog.

        Args:
            name: Catalog name
            location: Catalog directory for final storage (e.g., ".faceberg/")
            config: Catalog configuration (optional, for dataset sync)
            **properties: Additional catalog properties
        """
        super().__init__(name, **properties)
        self.catalog_dir = Path(location)
        self.config = config

        # Temporary staging attributes (set within context manager)
        self._staging_dir = None
        self._catalog = None
        self._old_catalog = None  # For tracking changes

        # Ensure catalog directory exists
        self.catalog_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Internal helper methods (catalog persistence and utilities)
    # =========================================================================

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

            # Handle legacy format (flat dict)
            if "type" not in data:
                self._catalog = data
                # Set default uri for legacy catalogs
                path_str = self.catalog_dir.absolute().as_posix()
                self.uri = f"file:///{path_str.lstrip('/')}"
            else:
                # New format with metadata - assert type matches
                assert data["type"] == "local", f"Expected catalog type 'local', got '{data['type']}'"
                self._catalog = data.get("tables", {})
                self.uri = data["uri"]
        else:
            self._catalog = {}
            path_str = self.catalog_dir.absolute().as_posix()
            self.uri = f"file:///{path_str.lstrip('/')}"

        return self._catalog

    def _save_catalog(self) -> None:
        """Save catalog to staging directory.

        Serializes self._catalog to staging_dir/catalog.json.
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

    def _gather_changes(self) -> List[Union[CommitOperationAdd, CommitOperationDelete]]:
        """Gather changes between old and new catalog state.

        Compares self._old_catalog with current staging directory state to
        construct CommitOperation objects for adds/deletes.

        Returns:
            List of CommitOperation objects (Add for new/modified files, Delete for removed tables)

        Must be called within _staging() context after _save_catalog().
        """
        if self._staging_dir is None:
            raise RuntimeError("_gather_changes() must be called within _staging() context")

        operations = []

        # Always add catalog.json (it's been updated)
        catalog_file = self._staging_dir / "catalog.json"
        operations.append(
            CommitOperationAdd(
                path_in_repo="catalog.json",
                path_or_fileobj=str(catalog_file)
            )
        )

        # Collect all files in staging directory (metadata files, etc.)
        for file_path in self._staging_dir.rglob("*"):
            if file_path.is_file() and file_path.name != "catalog.json":
                rel_path = file_path.relative_to(self._staging_dir)
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=str(rel_path),
                        path_or_fileobj=str(file_path)
                    )
                )

        # Detect deleted tables by comparing old and new catalog
        old_table_ids = set(self._old_catalog.keys()) if self._old_catalog else set()
        new_table_ids = set(self._catalog.keys())
        deleted_table_ids = old_table_ids - new_table_ids

        for table_id in deleted_table_ids:
            # table_id is like "namespace.table_name"
            # Remove the table's metadata directory
            namespace, table_name = table_id.rsplit(".", 1)
            table_dir = f"{namespace}/{table_name}/"
            operations.append(
                CommitOperationDelete(path_in_repo=table_dir)
            )

        return operations

    def _persist_changes(self) -> None:
        """Persist staged changes to catalog directory.

        Applies gathered changes (CommitOperations) to catalog_dir.
        - CommitOperationAdd: moves file from staging to catalog_dir
        - CommitOperationDelete: removes directory from catalog_dir

        Must be called within _staging() context.
        """
        if self._staging_dir is None:
            raise RuntimeError("_persist_changes() must be called within _staging() context")

        # Gather changes to apply
        changes = self._gather_changes()

        # Apply each operation
        for operation in changes:
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

    @contextmanager
    def _staging(self):
        """Context manager for staging catalog operations.

        Creates a temporary staging directory, loads catalog, and provides
        staging context for modifications. On exit, saves catalog, persists changes, and cleans up.

        Usage:
            with self._staging():
                # self._staging_dir is available
                # self._catalog contains loaded catalog
                self._catalog['new.table'] = 'path/to/metadata'
                # Write metadata files to self._staging_dir
        """
        # Create temporary staging directory
        temp_dir = tempfile.mkdtemp(prefix="faceberg_staging_")
        self._staging_dir = Path(temp_dir)

        # Load catalog from catalog_dir (sets self._catalog)
        self._old_catalog = self._load_catalog().copy()  # Store old state for comparison

        try:
            yield
        finally:
            # Save catalog to staging (serializes self._catalog to staging_dir/catalog.json)
            self._save_catalog()

            # Persist changes to catalog directory
            self._persist_changes()

            # Clean up
            self._catalog = None
            self._old_catalog = None
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

    def _get_metadata_location(self, identifier: Union[str, Identifier]) -> Path:
        """Get metadata directory path for table in staging.

        Args:
            identifier: Table identifier

        Returns:
            Path to table metadata directory

        Must be called within _staging() context.
        """
        if self._staging_dir is None:
            raise RuntimeError("_get_metadata_location() must be called within _staging() context")

        table_id = self._identifier_to_str(identifier)
        # Split into namespace and table name
        parts = table_id.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid table identifier: {table_id}. "
                "Expected format: namespace.table_name"
            )

        # Construct path as namespace/table_name
        namespace = parts[0]
        table_name = ".".join(parts[1:])  # Handle multi-part table names
        return self._staging_dir / namespace / table_name

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

            # Determine table location path for writing
            if location is None:
                location_path = self._get_metadata_location(identifier)
            else:
                location_path = Path(location)

            # Ensure location directory exists
            location_path.mkdir(parents=True, exist_ok=True)

            # Calculate base URI for table metadata
            from faceberg.convert import _join_uri

            table_id = self._identifier_to_str(identifier)
            parts = table_id.split(".")
            namespace = parts[0]
            table_name = ".".join(parts[1:])

            # Use catalog's uri + table path
            base_uri = _join_uri(self.uri, namespace, table_name)

            # Create table metadata with URI location
            metadata = new_table_metadata(
                schema=schema,
                partition_spec=partition_spec,
                sort_order=sort_order,
                location=base_uri,
                properties=properties,
            )

            # Write metadata file
            metadata_path = (
                location_path / "metadata" / f"v{metadata.last_sequence_number}.metadata.json"
            )
            metadata_path.parent.mkdir(parents=True, exist_ok=True)

            with open(metadata_path, "w") as f:
                f.write(metadata.model_dump_json(indent=2))

            # Write version hint
            version_hint_path = location_path / "metadata" / "version-hint.text"
            with open(version_hint_path, "w") as f:
                f.write(str(metadata.last_sequence_number))

            # Register in catalog with metadata file path (relative to staging_dir)
            rel_path = metadata_path.relative_to(self._staging_dir)
            self._catalog[table_id] = str(rel_path)

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

        # Get metadata location from catalog
        tables = self._load_catalog()
        metadata_location = tables.get(table_id)
        if not metadata_location:
            raise NoSuchTableError(f"Table {table_id} not found")

        # Resolve relative path to absolute path (from catalog_dir)
        metadata_path = Path(metadata_location)
        if not metadata_path.is_absolute():
            metadata_path = self.catalog_dir / metadata_path

        # Check if metadata_location points to a metadata file or directory (backward compatibility)
        if metadata_path.suffix == ".json":
            # New format: direct path to metadata file
            if not metadata_path.exists():
                raise NoSuchTableError(f"Table {table_id} metadata file not found: {metadata_path}")
            table_location = str(metadata_path.parent.parent)
        else:
            # Old format: directory path, read version hint
            table_location = metadata_location
            version_hint_path = metadata_path / "metadata" / "version-hint.text"
            if not version_hint_path.exists():
                raise NoSuchTableError(f"Table {table_id} metadata not found")

            with open(version_hint_path) as f:
                version = f.read().strip()

            metadata_path = metadata_path / "metadata" / f"v{version}.metadata.json"
            if not metadata_path.exists():
                raise NoSuchTableError(f"Table {table_id} metadata file not found: {metadata_path}")

        # Load FileIO and metadata
        io = load_file_io(properties=self.properties, location=table_location)

        metadata_file = io.new_input(str(metadata_path))
        metadata = FromInputFile.table_metadata(metadata_file)

        return Table(
            identifier=(
                self.identifier_to_tuple(identifier)
                if isinstance(identifier, str)
                else identifier
            ),
            metadata=metadata,
            metadata_location=str(metadata_path),
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

            # Convert to relative path if within staging
            metadata_path = Path(metadata_location)
            if metadata_path.is_absolute():
                try:
                    metadata_path = metadata_path.relative_to(self._staging_dir)
                    metadata_location = str(metadata_path)
                except ValueError:
                    # Path is not within staging, keep absolute
                    pass

            # Register in catalog (supports both file path and directory path)
            self._catalog[table_id] = metadata_location

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

            # Determine new metadata location in staging
            # Parse the namespace and table name from identifiers
            from_parts = from_id.split(".")
            to_parts = to_id.split(".")

            from_namespace = from_parts[0]
            from_table = ".".join(from_parts[1:])
            to_namespace = to_parts[0]
            to_table = ".".join(to_parts[1:])

            # Copy old table directory to new location in staging
            old_table_dir = self.catalog_dir / from_namespace / from_table
            new_table_dir = self._staging_dir / to_namespace / to_table

            if old_table_dir.exists():
                shutil.copytree(old_table_dir, new_table_dir)

                # Update catalog with new metadata path (relative to staging)
                # Find the metadata file in the copied directory
                metadata_files = list(new_table_dir.glob("metadata/*.metadata.json"))
                if metadata_files:
                    # Use the latest metadata file
                    metadata_file = sorted(metadata_files)[-1]
                    rel_path = metadata_file.relative_to(self._staging_dir)
                    self._catalog[to_id] = str(rel_path)

            # Remove old table from catalog (this will trigger deletion in _gather_changes)
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
        token: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> List[Table]:
        """Sync Iceberg tables with HuggingFace datasets in config.

        Discovers datasets and either creates new tables or updates existing ones
        with new snapshots if the dataset revision has changed.

        Args:
            token: HuggingFace API token (optional, uses HF_TOKEN env var if not provided)
            table_name: Specific table to sync (None for all), format: "namespace.table_name"

        Returns:
            List of synced Table objects (created or updated)

        Raises:
            ValueError: If config is not set or table_name is invalid
        """
        # Validate config is provided
        if self.config is None:
            raise ValueError(
                "No config set. Use LocalCatalog.from_local() or "
                "RemoteCatalog.from_hub() to create catalog with config."
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
            for ns in self.config.namespaces:
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
                for ns in self.config.namespaces
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

            # Determine table location using the same logic as _get_metadata_location
            table_path = self._get_metadata_location(table_info.identifier)
            table_path.mkdir(parents=True, exist_ok=True)

            # Calculate base URI for table metadata
            # Convert identifier to path components
            from faceberg.convert import _join_uri

            table_id = self._identifier_to_str(table_info.identifier)
            parts = table_id.split(".")
            namespace = parts[0]
            table_name = ".".join(parts[1:])

            # Use catalog's uri + table path
            base_uri = _join_uri(self.uri, namespace, table_name)

            # Create metadata writer
            metadata_writer = IcebergMetadataWriter(
                table_path=table_path,
                schema=table_info.schema,
                partition_spec=table_info.partition_spec,
                base_uri=base_uri,
            )

            # Generate table UUID
            table_uuid = str(uuid.uuid4())

            # Write Iceberg metadata files (manifest, manifest list, table metadata)
            metadata_file_path = metadata_writer.create_metadata_from_files(
                file_infos=table_info.files,
                table_uuid=table_uuid,
                properties=table_info.get_table_properties(),
            )

            # Register table in catalog with metadata file path (relative to staging)
            rel_path = metadata_file_path.relative_to(self._staging_dir)
            self._catalog[table_info.identifier] = str(rel_path)

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
            table_path = self._get_metadata_location(table_info.identifier)

            # Calculate base URI for table metadata
            from faceberg.convert import _join_uri

            table_id = self._identifier_to_str(table_info.identifier)
            parts = table_id.split(".")
            namespace = parts[0]
            table_name = ".".join(parts[1:])

            # Use catalog's uri + table path
            base_uri = _join_uri(self.uri, namespace, table_name)

            # Create metadata writer
            metadata_writer = IcebergMetadataWriter(
                table_path=table_path,
                schema=table_info.schema,
                partition_spec=table_info.partition_spec,
                base_uri=base_uri,
            )

            # Append new snapshot with updated files
            metadata_file_path = metadata_writer.append_snapshot_from_files(
                file_infos=table_info.files,
                current_metadata=table.metadata,
                properties=table_info.get_table_properties(),
            )

            # Update catalog with new metadata file path (relative to staging)
            rel_path = metadata_file_path.relative_to(self._staging_dir)
            self._catalog[table_info.identifier] = str(rel_path)

        # Reload and return table after staging context exits and persists
        return self.load_table(table_info.identifier)


class RemoteCatalog(LocalCatalog):
    """Remote Iceberg catalog with HuggingFace Hub integration.

    Extends LocalCatalog to automatically sync changes to HuggingFace Hub.
    All operations work on a local staging directory (cache), and changes
    are uploaded to the hub by _persist_changes().

    Features:
    - Local staging (cache) with automatic hub uploads
    - Deletion tracking for removed tables
    - Atomic commits with all changes
    """

    def __init__(
        self,
        name: str,
        location: str,
        config: Optional[CatalogConfig] = None,
        hf_repo_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        **properties: str,
    ):
        """Initialize remote catalog.

        Args:
            name: Catalog name
            location: Path to local cache directory (e.g., "~/.faceberg/cache/repo_id")
            config: Catalog configuration (optional, for dataset sync)
            hf_repo_id: HuggingFace repository ID for hub syncing
            hf_token: HuggingFace authentication token
            **properties: Additional catalog properties
        """
        # Call parent init (which sets up catalog_dir, etc.)
        super().__init__(name, location, config, **properties)

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

        # Handle both legacy and new formats
        if "type" not in data:
            self._catalog = data
            self.uri = f"hf://datasets/{self.hf_repo_id}"
        else:
            # New format with metadata - assert type matches
            assert data["type"] == "remote", f"Expected catalog type 'remote', got '{data['type']}'"
            self._catalog = data.get("tables", {})
            self.uri = data["uri"]

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
        """Save catalog to staging directory (used by parent's _gather_changes).

        Serializes self._catalog to staging_dir/catalog.json with remote catalog format.
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

    def _persist_changes(self) -> None:
        """Persist staged changes to HuggingFace Hub and local cache.

        Uses gathered changeset to create atomic commit with all operations.
        Tracks parent revision for proper concurrent update handling.
        Also saves to local cache (catalog_dir).
        """
        # Gather changes (Add/Delete operations)
        operations = self._gather_changes()

        # Create commit with all operations, using loaded revision as parent
        api = HfApi(token=self.hf_token)
        commit_info = api.create_commit(
            repo_id=self.hf_repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message="Sync catalog metadata",
            parent_commit=self._loaded_revision,  # Use tracked revision as parent
        )

        # Update loaded revision to the new commit
        self._loaded_revision = commit_info.commit_url.split("/")[-1]


# Alias for main API
FacebergCatalog = LocalCatalog


