#!/usr/bin/env python3
"""Demonstration of the new TableInfo abstraction.

This script shows how to:
1. Discover a HuggingFace dataset
2. Convert it to TableInfo objects (bridge between HF and Iceberg)
3. Create Iceberg metadata files in metadata-only mode
"""

import tempfile
from pathlib import Path

from faceberg.bridge import DatasetInfo
from faceberg.convert import IcebergMetadataWriter


def main():
    print("=" * 70)
    print("TableInfo Abstraction Demo")
    print("=" * 70)
    print()

    # Step 1: Discover a HuggingFace dataset
    print("Step 1: Discovering HuggingFace dataset...")
    dataset_info = DatasetInfo.discover(
        repo_id="stanfordnlp/imdb",
        configs=["plain_text"],
    )
    print(f"  ✓ Discovered dataset: {dataset_info.repo_id}")
    print(f"  ✓ Configs: {dataset_info.configs}")
    print(f"  ✓ Splits: {list(dataset_info.splits['plain_text'])}")
    print()

    # Step 2: Convert DatasetInfo to TableInfo objects
    print("Step 2: Converting to TableInfo objects...")
    table_infos = dataset_info.to_table_infos(
        namespace="default",
        table_name_prefix="imdb",
    )
    print(f"  ✓ Created {len(table_infos)} TableInfo object(s)")
    print()

    for table_info in table_infos:
        print(f"  TableInfo: {table_info.identifier}")
        print(f"    - Source: {table_info.source_repo}/{table_info.source_config}")
        print(f"    - Schema: {len(table_info.schema.fields)} fields")
        print(f"    - Partitioned by: {table_info.partition_spec.fields[0].name if table_info.partition_spec.fields else 'none'}")
        print(f"    - Files: {len(table_info.files)} files")
        print(f"    - Total rows: {table_info.total_rows:,}")
        print(f"    - Total size: {table_info.total_size:,} bytes")
        print()

        # Show schema fields
        print(f"    Schema fields:")
        for field in table_info.schema.fields[:5]:  # First 5 fields
            print(f"      - {field.name}: {field.field_type}")
        if len(table_info.schema.fields) > 5:
            print(f"      ... and {len(table_info.schema.fields) - 5} more")
        print()

        # Show file distribution by split
        split_counts = {}
        for file_info in table_info.files:
            split_counts[file_info.split] = split_counts.get(file_info.split, 0) + 1
        print(f"    Files by split:")
        for split, count in split_counts.items():
            print(f"      - {split}: {count} files")
        print()

    # Step 3: Create Iceberg metadata in metadata-only mode
    print("Step 3: Creating Iceberg metadata (metadata-only mode)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        for table_info in table_infos:
            # Create table directory
            table_path = Path(tmpdir) / table_info.table_name
            table_path.mkdir(parents=True, exist_ok=True)

            # Create metadata writer
            writer = IcebergMetadataWriter(
                table_path=table_path,
                schema=table_info.schema,
                partition_spec=table_info.partition_spec,
            )

            # Write metadata files
            import uuid
            metadata_location = writer.create_metadata_from_files(
                file_infos=table_info.files,
                table_uuid=str(uuid.uuid4()),
                properties=table_info.get_table_properties(),
            )

            print(f"  ✓ Created metadata for {table_info.identifier}")
            print(f"    - Metadata location: {metadata_location}")
            print(f"    - Manifest: {list((table_path / 'metadata').glob('*.avro'))[0].name}")
            print()

            # Show what files were created
            metadata_files = list((table_path / "metadata").iterdir())
            print(f"    Created {len(metadata_files)} metadata files:")
            for file in sorted(metadata_files):
                size = file.stat().st_size
                print(f"      - {file.name} ({size:,} bytes)")
            print()

    print("=" * 70)
    print("Demo complete!")
    print()
    print("Key takeaways:")
    print("  • DatasetInfo.discover() finds HuggingFace dataset structure")
    print("  • DatasetInfo.to_table_infos() converts to Iceberg-ready format")
    print("  • TableInfo contains all metadata needed for Iceberg table")
    print("  • IcebergMetadataWriter creates metadata files without copying data")
    print("  • Metadata-only mode references original HF files via hf:// URIs")
    print("=" * 70)


if __name__ == "__main__":
    main()
