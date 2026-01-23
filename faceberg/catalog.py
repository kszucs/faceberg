"""JSON-backed Iceberg catalog implementation."""

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

from pyiceberg.catalog import Catalog, PropertiesUpdateSummary
from pyiceberg.exceptions import (
    NamespaceAlreadyExistsError,
    NamespaceNotEmptyError,
    NoSuchNamespaceError,
    NoSuchTableError,
    TableAlreadyExistsError,
)
from pyiceberg.io import load_file_io
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import CommitTableRequest, CommitTableResponse, Table
from pyiceberg.table.sorting import UNSORTED_SORT_ORDER, SortOrder
from pyiceberg.typedef import EMPTY_DICT, Identifier, Properties

from faceberg.bridge import TableInfo

if TYPE_CHECKING:
    from faceberg.config import CatalogConfig


class JsonCatalog(Catalog):
    """Simple JSON-backed Iceberg catalog.

    Stores table metadata locations in a flat JSON dictionary:
    {
        "namespace.table_name": "path/to/metadata"
    }

    This is the simplest possible catalog implementation suitable for
    local development and single-user scenarios.
    """

    def __init__(
        self,
        name: str,
        warehouse: str,
        **properties: str,
    ):
        """Initialize JSON catalog.

        Args:
            name: Catalog name
            warehouse: Path to catalog directory (e.g., ".faceberg/")
            **properties: Additional catalog properties
        """
        super().__init__(name, **properties)
        self.warehouse = Path(warehouse)
        self.catalog_file = self.warehouse / "catalog.json"

        # Ensure warehouse directory exists
        self.warehouse.mkdir(parents=True, exist_ok=True)

        # Load existing catalog or create new one
        self._tables = self._load_catalog()

        # Save initial empty catalog if it doesn't exist
        if not self.catalog_file.exists():
            self._save_catalog()

    def _load_catalog(self) -> Dict[str, str]:
        """Load catalog from JSON file.

        Returns:
            Dictionary mapping table identifier to metadata path
        """
        if self.catalog_file.exists():
            with open(self.catalog_file) as f:
                return json.load(f)
        return {}

    def _save_catalog(self) -> None:
        """Save catalog to JSON file."""
        with open(self.catalog_file, "w") as f:
            json.dump(self._tables, f, indent=2, sort_keys=True)

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
        """Get metadata directory path for table.

        Args:
            identifier: Table identifier

        Returns:
            Path to table metadata directory
        """
        table_id = self._identifier_to_str(identifier)
        # Split into namespace and table name
        parts = table_id.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid table identifier: {table_id}. Expected format: namespace.table_name")

        # Construct path as namespace/table_name
        namespace = parts[0]
        table_name = ".".join(parts[1:])  # Handle multi-part table names
        return self.warehouse / namespace / table_name

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

        # Check if any tables exist in this namespace
        for table_id in self._tables.keys():
            if table_id.startswith(ns_str + "."):
                raise NamespaceAlreadyExistsError(f"Namespace {ns_str} already exists")

        # For JSON catalog, namespaces are implicit from table names
        # Just create the directory at the warehouse root
        ns_dir = self.warehouse / ns_str
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

        # Check if namespace has tables
        for table_id in self._tables.keys():
            if table_id.startswith(ns_str + "."):
                raise NamespaceNotEmptyError(f"Namespace {ns_str} is not empty")

        # For JSON catalog, if no tables, namespace doesn't really exist
        # But we can try to remove the directory
        ns_dir = self.warehouse / ns_str
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
        namespaces: Set[str] = set()
        for table_id in self._tables.keys():
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

        # Check if table already exists
        if table_id in self._tables:
            raise TableAlreadyExistsError(f"Table {table_id} already exists")

        # Convert schema if needed
        schema = self._convert_schema_if_needed(schema)

        # Determine table location
        if location is None:
            location = str(self._get_metadata_location(identifier))

        # Ensure location directory exists
        location_path = Path(location)
        location_path.mkdir(parents=True, exist_ok=True)

        # Create table metadata
        from pyiceberg.table.metadata import new_table_metadata

        metadata = new_table_metadata(
            schema=schema,
            partition_spec=partition_spec,
            sort_order=sort_order,
            location=location,
            properties=properties,
        )

        # Write metadata file
        metadata_path = location_path / "metadata" / f"v{metadata.last_sequence_number}.metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, "w") as f:
            f.write(metadata.model_dump_json(indent=2))

        # Write version hint
        version_hint_path = location_path / "metadata" / "version-hint.text"
        with open(version_hint_path, "w") as f:
            f.write(str(metadata.last_sequence_number))

        # Register in catalog
        self._tables[table_id] = str(location)
        self._save_catalog()

        # Load and return table
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
        metadata_location = self._tables.get(table_id)
        if not metadata_location:
            raise NoSuchTableError(f"Table {table_id} not found")

        # Load table from metadata location
        location_path = Path(metadata_location)

        # Read version hint to find current metadata file
        version_hint_path = location_path / "metadata" / "version-hint.text"
        if not version_hint_path.exists():
            raise NoSuchTableError(f"Table {table_id} metadata not found")

        with open(version_hint_path) as f:
            version = f.read().strip()

        # Load metadata file
        metadata_path = location_path / "metadata" / f"v{version}.metadata.json"
        if not metadata_path.exists():
            raise NoSuchTableError(f"Table {table_id} metadata file not found: {metadata_path}")

        # Load FileIO and metadata
        io = load_file_io(properties=self.properties, location=metadata_location)

        from pyiceberg.serializers import FromInputFile

        metadata_file = io.new_input(str(metadata_path))
        metadata = FromInputFile.table_metadata(metadata_file)

        return Table(
            identifier=self.identifier_to_tuple(identifier) if isinstance(identifier, str) else identifier,
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
            metadata_location: Path to table metadata

        Returns:
            Registered table

        Raises:
            TableAlreadyExistsError: If table already exists
        """
        table_id = self._identifier_to_str(identifier)

        if table_id in self._tables:
            raise TableAlreadyExistsError(f"Table {table_id} already exists")

        # Register in catalog
        self._tables[table_id] = metadata_location
        self._save_catalog()

        return self.load_table(identifier)

    def list_tables(self, namespace: Union[str, Identifier]) -> List[Identifier]:
        """List tables in namespace.

        Args:
            namespace: Namespace identifier

        Returns:
            List of table identifiers in namespace
        """
        ns_str = self._identifier_to_str(namespace)

        tables = []
        for table_id in self._tables.keys():
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

        if table_id not in self._tables:
            raise NoSuchTableError(f"Table {table_id} not found")

        # Remove from catalog
        del self._tables[table_id]
        self._save_catalog()

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

        if from_id not in self._tables:
            raise NoSuchTableError(f"Table {from_id} not found")

        if to_id in self._tables:
            raise TableAlreadyExistsError(f"Table {to_id} already exists")

        # Move table in catalog
        self._tables[to_id] = self._tables[from_id]
        del self._tables[from_id]
        self._save_catalog()

        return self.load_table(to_identifier)

    def table_exists(self, identifier: Union[str, Identifier]) -> bool:
        """Check if table exists.

        Args:
            identifier: Table identifier

        Returns:
            True if table exists
        """
        table_id = self._identifier_to_str(identifier)
        return table_id in self._tables

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


class FacebergCatalog(JsonCatalog):
    """Faceberg-specific catalog with dataset discovery and table creation.

    Extends JsonCatalog with methods for:
    - Initializing catalog from CatalogConfig
    - Discovering HuggingFace datasets
    - Creating Iceberg tables from datasets
    """

    def __init__(self, name: str, warehouse: str, config: Optional["CatalogConfig"] = None, **properties: str):
        """Initialize Faceberg catalog.

        Args:
            name: Catalog name
            warehouse: Path to catalog directory
            config: Catalog configuration (optional)
            **properties: Additional catalog properties
        """
        super().__init__(name, warehouse, **properties)
        self.config = config

    @classmethod
    def from_config(cls, config: "CatalogConfig") -> "FacebergCatalog":
        """Create catalog from catalog configuration.

        Args:
            config: Catalog configuration

        Returns:
            FacebergCatalog instance
        """
        return cls(
            name=config.name,
            warehouse=str(config.location),
            config=config,
        )

    def initialize(self) -> None:
        """Initialize catalog with all namespaces from config."""
        if self.config is None:
            raise ValueError("No config set. Use FacebergCatalog.from_config() to create catalog with config.")

        for namespace_config in self.config.namespaces:
            try:
                self.create_namespace(namespace_config.name)
            except NamespaceAlreadyExistsError:
                pass  # Namespace already exists, that's fine

    def create_tables(
        self,
        token: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> List[Table]:
        """Create Iceberg tables for HuggingFace datasets in config.

        Uses the tables defined in the FacebergConfig namespaces to create Iceberg tables.
        This method discovers datasets, converts them to TableInfo objects,
        and then creates the Iceberg metadata in metadata-only mode.

        Args:
            token: HuggingFace API token (optional, uses HF_TOKEN env var if not provided)
            table_name: Specific table to create (None for all), format: "namespace.table_name"

        Returns:
            List of created Table objects

        Raises:
            ValueError: If config is not set or if table_name is invalid
        """
        from faceberg.bridge import DatasetInfo

        if self.config is None:
            raise ValueError("No config set. Use FacebergCatalog.from_config() to create catalog with config.")

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

        # Create tables
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

            # Create table
            table = self._create_table_from_table_info(table_info)
            created_tables.append(table)

        return created_tables

    def _create_table_from_table_info(self, table_info: TableInfo) -> Table:
        """Create Iceberg table from TableInfo using metadata-only mode.

        This method creates an Iceberg table by:
        1. Creating the table metadata directory
        2. Using IcebergMetadataWriter to write Iceberg metadata files
        3. Registering the table in the catalog

        Args:
            table_info: TableInfo containing all metadata needed for table creation

        Returns:
            Created Table object

        Raises:
            TableAlreadyExistsError: If table already exists
        """
        from faceberg.convert import IcebergMetadataWriter

        # Check if table already exists
        if self.table_exists(table_info.identifier):
            raise TableAlreadyExistsError(f"Table {table_info.identifier} already exists")

        # Determine table location using the same logic as _get_metadata_location
        table_path = self._get_metadata_location(table_info.identifier)
        table_path.mkdir(parents=True, exist_ok=True)

        # Create metadata writer
        metadata_writer = IcebergMetadataWriter(
            table_path=table_path,
            schema=table_info.schema,
            partition_spec=table_info.partition_spec,
        )

        # Generate table UUID
        table_uuid = str(uuid.uuid4())

        # Write Iceberg metadata files (manifest, manifest list, table metadata)
        metadata_location = metadata_writer.create_metadata_from_files(
            file_infos=table_info.files,
            table_uuid=table_uuid,
            properties=table_info.get_table_properties(),
        )

        # Register table in catalog
        self._tables[table_info.identifier] = str(table_path)
        self._save_catalog()

        # Load and return table
        return self.load_table(table_info.identifier)
