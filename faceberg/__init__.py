"""Faceberg - Bridge between HuggingFace datasets and Apache Iceberg tables."""

from faceberg.bridge import (
    DatasetInfo,
    FileInfo,
    TableInfo,
    build_iceberg_schema_from_features,
    build_split_partition_spec,
)
from faceberg.catalog import FacebergCatalog, JsonCatalog
from faceberg.config import CatalogConfig, DatasetConfig, FacebergConfig
from faceberg.convert import IcebergMetadataWriter

__all__ = [
    # Catalog
    "FacebergCatalog",
    "JsonCatalog",
    # Config
    "CatalogConfig",
    "DatasetConfig",
    "FacebergConfig",
    # Bridge (discovery + schema conversion + TableInfo output)
    "DatasetInfo",
    "FileInfo",
    "TableInfo",
    "build_iceberg_schema_from_features",
    "build_split_partition_spec",
    # Convert (Iceberg metadata writer)
    "IcebergMetadataWriter",
]
