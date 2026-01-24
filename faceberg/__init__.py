"""Faceberg - Bridge between HuggingFace datasets and Apache Iceberg tables."""

from faceberg.bridge import (
    DatasetInfo,
    FileInfo,
    TableInfo,
    build_iceberg_schema_from_features,
    build_split_partition_spec,
)
from faceberg.catalog import FacebergCatalog, HfFileIO, LocalCatalog, RemoteCatalog
from faceberg.config import CatalogConfig, NamespaceConfig, TableConfig
from faceberg.convert import IcebergMetadataWriter

__all__ = [
    # Catalog
    "FacebergCatalog",
    "LocalCatalog",
    "RemoteCatalog",
    # Config
    "CatalogConfig",
    "NamespaceConfig",
    "TableConfig",
    # Bridge (discovery + schema conversion + TableInfo output)
    "DatasetInfo",
    "FileInfo",
    "TableInfo",
    "build_iceberg_schema_from_features",
    "build_split_partition_spec",
    # Convert (Iceberg metadata writer)
    "IcebergMetadataWriter",
    # IO (FileIO implementations)
    "HfFileIO",
]
