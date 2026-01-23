"""Faceberg - Bridge between HuggingFace datasets and Apache Iceberg tables."""

from faceberg.bridge import (
    DatasetInfo,
    build_iceberg_schema_from_features,
    infer_schema_from_dataset,
)
from faceberg.catalog import FacebergCatalog, JsonCatalog
from faceberg.config import CatalogConfig, DatasetConfig, FacebergConfig
from faceberg.convert import FileInfo, IcebergMetadataWriter, TableInfo, build_split_partition_spec

__all__ = [
    # Catalog
    "FacebergCatalog",
    "JsonCatalog",
    # Config
    "CatalogConfig",
    "DatasetConfig",
    "FacebergConfig",
    # Bridge (discovery + schema conversion)
    "DatasetInfo",
    "build_iceberg_schema_from_features",
    "infer_schema_from_dataset",
    # Convert (TableInfo to Iceberg metadata)
    "FileInfo",
    "TableInfo",
    "IcebergMetadataWriter",
    "build_split_partition_spec",
]
