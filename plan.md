# HuggingFace-Iceberg Bridge: Implementation Plan

## Overview

Design a config-driven CLI application (`faceberg`) that bridges HuggingFace datasets with Apache Iceberg:
- **Config file**: `faceberg.yml` lists HF datasets to expose as Iceberg tables
- **Metadata storage**: JSON-based catalog (simple dictionary, git commits provide atomicity)
- **Data storage**: Existing Parquet files in original HF dataset repositories (referenced via `hf://` URIs)
- **Local testing**: Everything testable locally without HF infrastructure
- **CLI commands**: Create metadata, sync with dataset updates, manage tables
- **Dataset awareness**: Handle multiple configs (different schemas) and splits (partitions)
- **Full read/write**: Read existing data, write new data to original dataset repos

## Architecture

```
Local Development:
├── faceberg.yml                  # Config file listing HF datasets
├── .faceberg/                    # Local catalog storage
│   ├── catalog.json              # Simple JSON dict: table → metadata path
│   └── metadata/
│       └── default/              # Namespace
│           ├── dataset1_config1/ # Table for dataset1, config "config1"
│           ├── dataset1_config2/ # Table for dataset1, config "config2"
│           ├── dataset2_default/ # Table for dataset2, default config
│           └── dataset3_default/
│
HuggingFace Hub (Remote):
├── kszucs/dataset1 (Data Repository)
│   ├── config1/                  # Dataset configuration 1
│   │   └── train/
│   │       └── *.parquet
│   └── config2/                  # Dataset configuration 2
│       └── train/
│           └── *.parquet
├── kszucs/dataset2
│   └── data/
│       ├── train/*.parquet       # Split: train
│       ├── test/*.parquet        # Split: test
│       └── validation/*.parquet  # Split: validation
└── kszucs/dataset3
    └── data/*.parquet

When pushed to HF:
├── kszucs/faceberg-catalog (Catalog Repository)
│   ├── faceberg.yml              # Config pushed with catalog
│   ├── catalog.json
│   └── metadata/...
```

## Config File Format

**faceberg.yml**:
```yaml
catalog:
  name: my_catalog
  location: .faceberg/  # Local path or hf://datasets/kszucs/faceberg-catalog/

datasets:
  - name: dataset1
    repo: kszucs/dataset1
    configs:  # Optional, defaults to all configs in dataset
      - config1
      - config2

  - name: dataset2
    repo: kszucs/dataset2
    # If configs not specified, uses default config

  - name: dataset3
    repo: kszucs/dataset3
```

## CLI Commands

### `faceberg init`
Initialize a new Faceberg catalog from config file:
```bash
faceberg init faceberg.yml
```
- Creates `.faceberg/` directory
- Initializes `catalog.json`
- Creates namespace structure

### `faceberg sync`
Sync Iceberg metadata with current HF dataset state:
```bash
faceberg sync [table_name]  # Sync specific table or all if not specified
```
- Discovers new Parquet files in dataset repos
- Updates Iceberg metadata to reference new files
- Creates new snapshot with updated file list
- Handles dataset revisions (git commits in dataset repo)

### `faceberg create`
Create Iceberg tables for datasets in config:
```bash
faceberg create [table_name]  # Create specific table or all if not specified
```
- Inspects dataset schema from Parquet files
- Creates Iceberg table with matching schema
- Registers existing Parquet files using `add_files()`
- Sets up split partitioning (if dataset has splits)
- Configures `write.data.path` to point to dataset repo

### `faceberg list`
List all tables in catalog:
```bash
faceberg list
```

### `faceberg info`
Show information about a table:
```bash
faceberg info default.dataset1_config1
```
- Schema
- Partitioning
- Current snapshot ID
- Number of files
- Data location

### `faceberg push`
Push local catalog to HuggingFace:
```bash
faceberg push
```
- Uploads `.faceberg/` contents to HF catalog repository
- Commits with descriptive message

### `faceberg pull`
Pull catalog from HuggingFace to local:
```bash
faceberg pull
```

## Data Flow

**Initial Setup (Local)**:
1. User creates `faceberg.yml` with dataset references
2. Run `faceberg init` to initialize catalog
3. Run `faceberg create` to create Iceberg tables
4. Tables reference existing Parquet files via `hf://` URIs
5. Ready for local querying with PyIceberg

**Syncing Updates**:
1. Dataset owner pushes new data to `kszucs/dataset1`
2. Run `faceberg sync dataset1_config1` to refresh metadata
3. Faceberg discovers new Parquet files
4. Creates new Iceberg snapshot with updated file list
5. Commits metadata changes locally
6. Run `faceberg push` to share updated catalog

**Writing New Data**:
1. Client uses PyIceberg to append data to table
2. New Parquet file written to `hf://datasets/kszucs/dataset1/config1/train/`
3. Iceberg manifest updated locally in `.faceberg/metadata/`
4. Run `faceberg push` to share metadata updates

## Technical Approach

### 1. Simple JSON Catalog

**catalog.json** (just a dictionary):
```json
{
  "default.dataset1_config1": ".faceberg/metadata/default/dataset1_config1",
  "default.dataset1_config2": ".faceberg/metadata/default/dataset1_config2",
  "default.dataset2_default": ".faceberg/metadata/default/dataset2_default",
  "default.dataset3_default": ".faceberg/metadata/default/dataset3_default"
}
```

**Design principles**:
- Simple flat dictionary: table identifier → metadata path
- No complex indexing or relationships
- Git commits provide atomicity (don't worry about it for now)
- Human-readable JSON for debugging

**Version tracking** (per table):
- Use Iceberg's standard `version-hint.text` file
- Points to current metadata version (e.g., "v5")
- Standard Iceberg metadata JSON files (`v1.metadata.json`, `v2.metadata.json`, etc.)

### 2. Custom JsonCatalog Implementation

Implement a minimal PyIceberg catalog:

```python
class JsonCatalog(Catalog):
    def __init__(self, name: str, catalog_dir: str):
        self.name = name
        self.catalog_dir = Path(catalog_dir)
        self.catalog_file = self.catalog_dir / "catalog.json"
        self._tables = self._load_catalog()

    def _load_catalog(self) -> Dict[str, str]:
        """Load simple JSON dict: table_id → metadata_path"""
        if self.catalog_file.exists():
            return json.loads(self.catalog_file.read_text())
        return {}

    def _save_catalog(self):
        """Save catalog dict to JSON"""
        self.catalog_file.write_text(json.dumps(self._tables, indent=2))

    def list_tables(self, namespace: str) -> List[Identifier]:
        """Return list of tables in namespace"""
        return [
            (ns, table)
            for full_id in self._tables.keys()
            if (ns := full_id.split('.')[0]) == namespace
            for table in [full_id.split('.')[1]]
        ]

    def load_table(self, identifier: Identifier) -> Table:
        """Load table from metadata path"""
        table_id = self._identifier_to_str(identifier)
        metadata_path = self._tables.get(table_id)
        if not metadata_path:
            raise NoSuchTableError(f"Table {table_id} not found")
        return self._load_table_from_metadata(metadata_path)

    def create_table(self, identifier: Identifier, schema: Schema, ...) -> Table:
        """Create new table and register in catalog"""
        table_id = self._identifier_to_str(identifier)
        metadata_path = self._get_metadata_path(identifier)

        # Create Iceberg table metadata
        table = self._create_table_metadata(
            metadata_path, schema, location, properties
        )

        # Register in catalog
        self._tables[table_id] = str(metadata_path)
        self._save_catalog()

        return table
```

**Keep it simple**:
- Just implement the minimal methods needed
- No fancy locking or concurrency (git commits provide atomicity)
- No caching (reload catalog.json each time if needed)
- Focus on correctness first, optimize later

### 3. Dataset Configurations and Splits

**HuggingFace Dataset Structure**:
- **Configs**: Different schemas/versions of the same dataset (e.g., "small", "large", "processed")
- **Splits**: Subsets of data within a config (e.g., "train", "test", "validation")

**Mapping to Iceberg**:

#### Multiple Configs → Multiple Tables
```yaml
datasets:
  - name: dataset1
    repo: kszucs/dataset1
    configs:
      - small   # Creates table: default.dataset1_small
      - large   # Creates table: default.dataset1_large
```

**Rationale**: Different schemas need different tables

#### Multiple Splits → Single Table with Split Partitioning
```python
# Create table with split as a virtual partition column
schema = Schema(
    NestedField(1, "id", LongType()),
    NestedField(2, "text", StringType()),
    NestedField(3, "split", StringType()),  # Virtual column for split
)

# Query by split
df = table.scan().filter("split = 'train'").to_pandas()
```

**Implementation approaches**:

**Option A: Physical Partitioning (Recommended)**
- Add `split` column to Parquet schema if not present
- Use Iceberg's partitioning on `split` column
- Efficient pruning: only reads files for requested split

**Option B: File-level Metadata**
- Store split info in Iceberg manifest entry metadata
- Filter files during scan planning
- No schema changes needed

**Decision**: Option A (physical partitioning) for better query performance

### 4. FileIO Configuration

Use PyIceberg's built-in `FsspecFileIO` with HuggingFace support:

```python
from pyiceberg.io.fsspec import FsspecFileIO
import os

# Set authentication
os.environ["HF_TOKEN"] = "hf_..."

# FsspecFileIO automatically uses HfFileSystem for hf:// URIs
file_io = FsspecFileIO()
```

**Always use hf:// protocol**:
```
hf://datasets/kszucs/dataset1/config1/train/data-00000.parquet
```

**Benefits of hf:// with XET storage**:
- Transparent deduplication
- Efficient delta transfers
- CDN acceleration
- Automatic caching

### 4. Table Creation and Registration

#### Registering Existing Parquet Files

Use PyIceberg's `add_files()` API to register existing data without copying:

```python
# Create table with metadata in catalog repo
table = catalog.create_table(
    identifier="default.dataset1",
    schema=inferred_schema,
    location="hf://datasets/kszucs/iceberg-catalog/metadata/default/dataset1",
    properties={
        "write.data.path": "hf://datasets/kszucs/dataset1/data",
        "write.parquet.compression-codec": "zstd",
    }
)

# Register existing Parquet files
existing_files = [
    "hf://datasets/kszucs/dataset1/data/train-00000.parquet",
    "hf://datasets/kszucs/dataset1/data/train-00001.parquet",
    "hf://datasets/kszucs/dataset1/data/test-00000.parquet",
]
table.add_files(existing_files)
```

**Key considerations**:
- Parquet files must have compatible schemas
- PyIceberg will create name mappings automatically
- Field IDs will be assigned based on schema order

#### Writing New Data

Use standard Iceberg write operations:

```python
import pyarrow as pa

# Create new data
new_data = pa.table({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
})

# Append to table
# - New Parquet file written to: hf://datasets/kszucs/dataset1/data/
# - New manifest written to: hf://datasets/kszucs/iceberg-catalog/metadata/
# - New metadata version created: v{N+1}.metadata.json
table.append(new_data)
```

### 6. Custom LocationProvider

**Problem**: Iceberg's default file naming may not match HF dataset conventions.

**Iceberg default**:
```
{table_location}/data/{partition}/00000-{uuid}.parquet
```

**HF dataset convention** (example):
```
{dataset_repo}/{config}/{split}/data-{shard:05d}-of-{total:05d}.parquet
```

**Solution**: Implement custom `LocationProvider`:

```python
from pyiceberg.io import LocationProvider

class HFDatasetLocationProvider(LocationProvider):
    def __init__(self, base_location: str, config: str, split: str):
        self.base_location = base_location
        self.config = config
        self.split = split
        self._counter = 0

    def new_data_location(self, filename: str) -> str:
        """Generate HF-compatible file path"""
        # Get total shards from existing files or use default
        total_shards = self._get_total_shards()

        # Generate path matching HF convention
        shard_id = self._counter
        filename = f"data-{shard_id:05d}-of-{total_shards:05d}.parquet"

        self._counter += 1

        return f"{self.base_location}/{self.config}/{self.split}/{filename}"
```

**Usage**:
```python
table = catalog.create_table(
    ...,
    properties={
        "write.data.path": f"hf://datasets/{repo}/{config}",
        "write.location-provider.impl": "faceberg.location.HFDatasetLocationProvider",
    }
)
```

**Note**: This may need refinement based on actual HF dataset file naming patterns. Start simple and iterate.

### 7. Core Feature Support

#### Time Travel / Versioning

Iceberg naturally supports time travel through snapshot metadata:

```python
# Read current data
df = table.scan().to_arrow()

# Read historical snapshot
snapshot_id = table.history()[0].snapshot_id
df_historical = table.scan(snapshot_id=snapshot_id).to_arrow()

# Read as of timestamp
df_timestamp = table.scan(as_of_timestamp=datetime(2024, 1, 15)).to_arrow()
```

**Metadata versioning**:
- Each write creates a new `vN.metadata.json` file
- `version-hint.text` points to current version
- Old versions retained for time travel
- Snapshot expiration policy controls retention

#### Schema Evolution

Supported through Iceberg's schema evolution APIs:

```python
with table.update_schema() as update:
    update.add_column("new_field", StringType())

# Or rename, drop columns
with table.update_schema() as update:
    update.rename_column("old_name", "new_name")
```

**Implementation notes**:
- Schema changes create new metadata version
- Name mapping ensures backward compatibility
- Field IDs maintained across schema versions

#### Partitioning

Define partition spec during table creation:

```python
from pyiceberg.partitioning import PartitionSpec, PartitionField
from pyiceberg.transforms import DayTransform

partition_spec = PartitionSpec(
    PartitionField(
        source_id=1,  # field_id of 'timestamp' column
        field_id=1000,
        transform=DayTransform(),
        name="day"
    )
)

table = catalog.create_table(
    identifier="default.events",
    schema=schema,
    partition_spec=partition_spec,
    ...
)
```

**Benefits**:
- Efficient query pruning
- Organized data layout
- Better performance for time-range queries

### 8. Authentication and Access Control

**Setup**:
```python
import os
os.environ["HF_TOKEN"] = "hf_..."  # User access token
```

**Token types**:
- **Read tokens**: For read-only clients
- **Write tokens**: For clients that need to append data
- **Fine-grained tokens**: Restrict access to specific repositories

**Repository permissions**:
- Catalog repository: Read for all users, Write for maintainers
- Data repositories: Read for all users, Write for data producers

**Best practices**:
- Store tokens in secure vault (not in code)
- Use separate tokens for different environments
- Rotate tokens periodically
- Monitor token usage

## Implementation Phases

### Phase 1: Project Setup and CLI Foundation
**Goal**: Set up project structure and basic CLI

**Tasks**:
1. Create Python project structure
2. Set up dependencies (click, pyiceberg, huggingface_hub, pyyaml)
3. Implement basic CLI with Click
4. Implement config file parsing (faceberg.yml)
5. Add basic logging and error handling

**Deliverables**:
- Working CLI skeleton: `faceberg --help`
- Config file parser
- Unit tests for config parsing

**Files**:
- `faceberg/cli.py` - CLI entry point
- `faceberg/config.py` - Config file parsing
- `faceberg.yml` - Example config file
- `setup.py` or `pyproject.toml` - Package configuration

### Phase 2: Simple JSON Catalog
**Goal**: Implement minimal JSON-backed catalog

**Tasks**:
1. Implement `JsonCatalog` class extending PyIceberg's `Catalog`
2. Implement catalog.json read/write
3. Implement basic table registration (create, load, list)
4. Test locally without HF (use file:// URIs first)
5. Unit tests for catalog operations

**Deliverables**:
- Working `JsonCatalog` implementation
- Can create and load tables locally
- Test suite for catalog

**Files**:
- `faceberg/catalog.py` - JsonCatalog implementation

### Phase 3: Dataset Discovery and Registration
**Goal**: Create Iceberg tables from HF datasets

**Tasks**:
1. Implement `faceberg init` command
2. Implement `faceberg create` command
3. Implement dataset discovery (list Parquet files in HF repo)
4. Implement schema inference from Parquet files
5. Implement `add_files()` to register existing Parquet files
6. Handle dataset configs (one table per config)
7. Test with real HF datasets

**Deliverables**:
- `faceberg init` creates catalog from config
- `faceberg create` creates tables for all datasets
- Tables reference existing Parquet files via hf:// URIs

**Files**:
- `faceberg/discovery.py` - Dataset discovery logic
- `faceberg/schema.py` - Schema inference
- `faceberg/commands/init.py` - Init command
- `faceberg/commands/create.py` - Create command

### Phase 4: Split Handling
**Goal**: Support dataset splits as partitions

**Tasks**:
1. Detect splits in HF datasets
2. Add split column to schema (if needed)
3. Configure partitioning by split
4. Test queries filtering by split
5. Verify partition pruning works

**Deliverables**:
- Tables with splits are queryable: `WHERE split = 'train'`
- Efficient query pruning by split
- Documentation on split handling

**Files**:
- Updates to `faceberg/discovery.py` and `faceberg/schema.py`

### Phase 5: Sync Command
**Goal**: Refresh metadata when datasets change

**Tasks**:
1. Implement `faceberg sync` command
2. Discover new Parquet files in dataset
3. Add new files to table with `add_files()`
4. Handle dataset revisions (git SHA tracking)
5. Test incremental sync

**Deliverables**:
- `faceberg sync` updates tables with new data
- Only adds new files (doesn't re-add existing)
- Tracks dataset revision for incremental syncs

**Files**:
- `faceberg/commands/sync.py` - Sync command

### Phase 6: Write Support
**Goal**: Enable writing new data to tables

**Tasks**:
1. Configure `write.data.path` for tables
2. Implement custom `LocationProvider` (if needed)
3. Test `table.append()` writes to correct location
4. Verify new files follow HF naming convention
5. Test end-to-end: write → sync → read

**Deliverables**:
- Can append data to tables
- New files written to dataset repos with correct paths
- Metadata updated correctly

**Files**:
- `faceberg/location.py` - Custom LocationProvider (if needed)

### Phase 7: CLI Utilities
**Goal**: Add helpful CLI commands

**Tasks**:
1. Implement `faceberg list` - list tables
2. Implement `faceberg info <table>` - show table info
3. Implement `faceberg push` - push catalog to HF
4. Implement `faceberg pull` - pull catalog from HF
5. Add progress bars and nice formatting

**Deliverables**:
- Complete CLI with all commands
- User-friendly output
- Documentation

**Files**:
- `faceberg/commands/list.py`
- `faceberg/commands/info.py`
- `faceberg/commands/push.py`
- `faceberg/commands/pull.py`

### Phase 8: Testing and Documentation
**Goal**: Comprehensive testing and docs

**Tasks**:
1. End-to-end integration tests
2. Test with multiple dataset types
3. Performance benchmarking
4. Write user guide
5. Write developer guide
6. Create example workflows

**Deliverables**:
- Test suite with >80% coverage
- User documentation
- Example notebooks
- README with quickstart

## Project Structure

```
faceberg/
├── faceberg/
│   ├── __init__.py
│   ├── cli.py                    # Main CLI entry point
│   ├── config.py                 # Config file parsing
│   ├── catalog.py                # JsonCatalog implementation
│   ├── discovery.py              # HF dataset discovery
│   ├── schema.py                 # Schema inference from Parquet
│   ├── location.py               # Custom LocationProvider
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── init.py               # faceberg init
│   │   ├── create.py             # faceberg create
│   │   ├── sync.py               # faceberg sync
│   │   ├── list.py               # faceberg list
│   │   ├── info.py               # faceberg info
│   │   ├── push.py               # faceberg push
│   │   └── pull.py               # faceberg pull
│   └── utils/
│       ├── __init__.py
│       ├── hf.py                 # HF helper functions
│       └── parquet.py            # Parquet utilities
├── tests/
│   ├── test_config.py
│   ├── test_catalog.py
│   ├── test_discovery.py
│   ├── test_cli.py
│   └── test_e2e.py               # End-to-end tests
├── examples/
│   ├── faceberg.yml              # Example config
│   └── query_tables.py           # Example usage
├── pyproject.toml
├── README.md
└── LICENSE
```

## Critical Files

### New Files to Create (Priority Order)

1. **faceberg/config.py** (Phase 1)
   - Parse faceberg.yml
   - Validate config structure
   - Config dataclasses

2. **faceberg/cli.py** (Phase 1)
   - CLI entry point with Click
   - Command routing
   - Global options (--verbose, --config)

3. **faceberg/catalog.py** (Phase 2)
   - `JsonCatalog` class extending Catalog
   - Simple JSON dict read/write
   - Table registration

4. **faceberg/discovery.py** (Phase 3)
   - Discover Parquet files in HF dataset
   - List configs and splits
   - Get dataset metadata

5. **faceberg/schema.py** (Phase 3)
   - Infer Iceberg schema from Parquet
   - Handle split columns
   - Field ID assignment

6. **faceberg/commands/init.py** (Phase 3)
   - Initialize catalog directory
   - Create catalog.json
   - Set up namespace structure

7. **faceberg/commands/create.py** (Phase 3)
   - Create tables for datasets in config
   - Call discovery and schema inference
   - Register existing Parquet files

8. **faceberg/commands/sync.py** (Phase 5)
   - Sync table with dataset updates
   - Detect new files
   - Update metadata

9. **faceberg/location.py** (Phase 6)
   - Custom LocationProvider for HF naming
   - Generate file paths matching HF conventions

10. **faceberg/commands/list.py**, **info.py**, **push.py**, **pull.py** (Phase 7)
    - Utility commands for catalog management

### Key Files to Reference

1. **experiment.py** (Current file)
   - Reference pattern for metadata/data separation
   - Example of `write.data.path` usage
   - Table creation and append operations

2. **PyIceberg source** (in site-packages)
   - `pyiceberg/catalog/base.py` - Catalog interface to implement
   - `pyiceberg/io/fsspec.py` - FileIO implementation
   - `pyiceberg/table/__init__.py` - Table class
   - `pyiceberg/table/metadata.py` - Metadata structures
   - `pyiceberg/schema.py` - Schema classes

3. **HuggingFace Hub source** (in site-packages)
   - `huggingface_hub/hf_file_system.py` - HfFileSystem implementation
   - `datasets` library - Dataset loading and inspection

## Key Design Decisions

### 1. Config-Driven vs Programmatic API
**Decision**: Config-driven with faceberg.yml
**Rationale**:
- Declarative and version-controllable
- Easy to understand and modify
- Natural fit for CLI tool
- Can add programmatic API later if needed

### 2. Simple JSON Catalog
**Decision**: Flat JSON dictionary for catalog
**Rationale**:
- Simplest possible implementation
- Human-readable and debuggable
- Git commits provide atomicity (don't worry about concurrency)
- No complex indexing needed

### 3. Dataset Configs → Tables
**Decision**: Each config is a separate table
**Rationale**:
- Configs have different schemas (can't be same table)
- Clear namespace: `dataset1_config1`, `dataset1_config2`
- Simple to implement and understand

### 4. Dataset Splits → Partitions
**Decision**: Splits as physical partitions within single table
**Rationale**:
- Efficient query pruning by split
- Natural fit with Iceberg partitioning
- Matches user expectation: "SELECT * FROM table WHERE split='train'"

### 5. Always use hf:// Protocol
**Decision**: Always use hf:// URIs for data files
**Rationale**:
- Profit from HF XET storage advantages
- Native fsspec integration
- Works with PyIceberg's FsspecFileIO
- Consistent and clean paths

### 6. Local-First Development
**Decision**: Everything testable locally before pushing to HF
**Rationale**:
- Faster development iteration
- No HF quota consumption during dev
- Can test offline
- Push to HF only when ready

### 7. Custom LocationProvider
**Decision**: Implement if needed for file naming compatibility
**Rationale**:
- HF datasets may have specific naming conventions
- Iceberg's default naming might not match
- Allows seamless integration with existing datasets
- Can be added incrementally (Phase 6)

## Potential Challenges and Mitigations

### Challenge 1: Schema Compatibility
**Issue**: Existing Parquet files may not have Iceberg field IDs
**Impact**: Need name mapping, potential schema mismatches
**Mitigation**:
- Use PyIceberg's automatic name mapping via `add_files()`
- Infer schema from first Parquet file
- Validate all files have compatible schemas
- Document schema requirements clearly

### Challenge 2: File Naming Conventions
**Issue**: Iceberg's default file naming may not match HF dataset conventions
**Impact**: Files written by Iceberg may have different naming than existing files
**Mitigation**:
- Implement custom LocationProvider if needed
- Study existing HF dataset file naming patterns
- Make it configurable per dataset
- Document the naming strategy

### Challenge 3: Split Detection and Handling
**Issue**: Splits may be organized in different ways (subdirectories, file prefixes, etc.)
**Impact**: May need heuristics to detect split structure
**Mitigation**:
- Use HuggingFace datasets library to get split info
- Support common patterns (subdirectories: train/, test/)
- Make split handling configurable in faceberg.yml
- Provide clear error messages if splits can't be detected

### Challenge 4: Large Number of Files
**Issue**: Datasets with thousands of Parquet files may be slow to sync
**Impact**: `faceberg create` and `faceberg sync` may take time
**Mitigation**:
- Show progress bar during file discovery
- Use batch operations where possible
- Consider parallel file listing
- Cache file listings between syncs

### Challenge 5: Dataset Revisions
**Issue**: HF datasets can have multiple commits/revisions
**Impact**: Need to track which revision metadata corresponds to
**Mitigation**:
- Store dataset revision (git SHA) in table properties
- Use specific revision when listing files
- `faceberg sync` can update to latest revision
- Support pinning to specific revision in config

### Challenge 6: HF Token Authentication
**Issue**: Need HF token for private datasets
**Impact**: Must configure authentication correctly
**Mitigation**:
- Check for `HF_TOKEN` environment variable
- Provide clear error message if token missing
- Document authentication setup in README
- Support `huggingface-cli login` workflow

### Challenge 7: Local vs Remote Paths
**Issue**: Testing locally but deploying to HF
**Impact**: Need to handle both file:// and hf:// URIs
**Mitigation**:
- Always use hf:// in metadata (even during local testing)
- PyIceberg's FsspecFileIO handles both transparently
- Test with actual HF datasets early
- Document local testing setup

## Success Criteria

### Phase 1-2: Foundation
- ✅ CLI installed and working: `faceberg --help`
- ✅ Can parse faceberg.yml config file
- ✅ JsonCatalog can create and load tables locally

### Phase 3-4: Core Functionality
- ✅ `faceberg create` creates tables for all datasets in config
- ✅ Tables reference existing Parquet files via hf:// URIs
- ✅ Can query tables through PyIceberg
- ✅ Dataset configs create separate tables
- ✅ Dataset splits are queryable: `WHERE split = 'train'`

### Phase 5: Sync
- ✅ `faceberg sync` detects new Parquet files in datasets
- ✅ Adds new files to tables without re-adding existing ones
- ✅ Tracks dataset revisions

### Phase 6: Write Support
- ✅ Can append data with `table.append()`
- ✅ New files written to dataset repos at correct paths
- ✅ File naming matches HF conventions (if custom LocationProvider implemented)

### Phase 7-8: Complete
- ✅ All CLI commands work: init, create, sync, list, info, push, pull
- ✅ End-to-end workflow: create → query → append → sync → query
- ✅ Works with actual HF datasets (kszucs/dataset1, dataset2, dataset3)
- ✅ Comprehensive documentation and examples
- ✅ Test suite with >80% coverage

### User Experience Goals
- Simple setup: `pip install faceberg && faceberg init config.yml`
- Clear error messages with actionable suggestions
- Progress bars for long operations
- Works entirely locally for development
- Easy to push to HF when ready

## Verification Plan

### Unit Tests
- Catalog operations (create, list, load, drop)
- Metadata serialization/deserialization
- Version tracking logic
- Concurrency control (conflict detection, retry)

### Integration Tests
- End-to-end: Create catalog, register table, read data
- Write flow: Append data, verify new files and metadata
- Time travel: Query historical snapshots
- Schema evolution: Add columns, verify backward compatibility
- Partitioning: Create partitioned table, verify pruning

### Multi-User Tests
- Concurrent reads (10 clients)
- Concurrent writes (3 clients, measure conflict rate)
- Mixed read/write workload
- Conflict resolution correctness

### Performance Tests
- Benchmark read latency vs direct Parquet
- Benchmark write latency (small, medium, large batches)
- Measure catalog operation latency
- Profile bottlenecks (network, serialization, git)

### Compatibility Tests
- Test with DuckDB Iceberg extension
- Test with Apache Spark Iceberg integration
- Test with Trino/Presto Iceberg connector
- Verify standard Iceberg clients can read tables

## Open Questions for User

1. **File naming conventions**: Do your HF datasets follow a specific file naming pattern? (e.g., `data-00000-of-00010.parquet` vs `train-00000.parquet`)
   - This determines if we need a custom LocationProvider

2. **Split organization**: How are splits organized in your datasets?
   - Option A: Subdirectories (train/, test/, validation/)
   - Option B: File prefixes (train-*.parquet, test-*.parquet)
   - Option C: Metadata-only (HF datasets library knows splits)

3. **Dataset revisions**: Should tables track specific dataset revisions or always use latest?
   - Option A: Pin to specific revision (reproducibility)
   - Option B: Always sync to latest (freshness)
   - Option C: Configurable per dataset

4. **Namespace strategy**: Should all tables be in `default` namespace or organized differently?
   - Option A: Single `default` namespace (simplest)
   - Option B: Namespace per dataset (e.g., `dataset1.config1`, `dataset1.config2`)

5. **Table naming**: How should tables be named?
   - Option A: `{dataset_name}_{config_name}` (e.g., `dataset1_small`)
   - Option B: `{dataset_name}` only if single config, else `{dataset_name}_{config_name}`
   - Option C: Configurable in faceberg.yml

## Example Usage

### Initial Setup
```bash
# Install
pip install faceberg

# Create config file
cat > faceberg.yml <<EOF
catalog:
  name: my_catalog
  location: .faceberg/

datasets:
  - name: dataset1
    repo: kszucs/dataset1
  - name: dataset2
    repo: kszucs/dataset2
  - name: dataset3
    repo: kszucs/dataset3
EOF

# Set HF token
export HF_TOKEN=hf_...

# Initialize catalog
faceberg init faceberg.yml

# Create tables for all datasets
faceberg create

# List tables
faceberg list
```

### Querying Data
```python
from faceberg.catalog import JsonCatalog

# Load catalog
catalog = JsonCatalog("my_catalog", ".faceberg/")

# Load table
table = catalog.load_table("default.dataset1_default")

# Query data
df = table.scan().to_pandas()

# Query specific split
df_train = table.scan().filter("split = 'train'").to_pandas()

# Time travel
snapshot_id = table.history()[0].snapshot_id
df_historical = table.scan(snapshot_id=snapshot_id).to_pandas()
```

### Writing Data
```python
import pyarrow as pa

# Load table
table = catalog.load_table("default.dataset1_default")

# Append data
new_data = pa.table({
    "id": [1, 2, 3],
    "text": ["A", "B", "C"],
    "split": ["train", "train", "train"],
})
table.append(new_data)
```

### Syncing Updates
```bash
# Sync all tables
faceberg sync

# Sync specific table
faceberg sync default.dataset1_default
```

### Pushing to HuggingFace
```bash
# Push catalog to HF
faceberg push
# Creates/updates kszucs/faceberg-catalog with local .faceberg/ contents
```

## Next Steps

After plan approval, start with Phase 1:

1. **Create project structure**
   - Set up Python package with Click for CLI
   - Add dependencies to pyproject.toml
   - Create basic file structure

2. **Implement config parsing**
   - Define config schema with Pydantic or dataclasses
   - Parse faceberg.yml
   - Validate config structure

3. **Basic CLI skeleton**
   - `faceberg --help` works
   - `faceberg init --help` shows help
   - Placeholder commands print "Not implemented yet"

4. **Begin Phase 2: JsonCatalog**
   - Start implementing simple JSON dict catalog
   - Test locally with file:// URIs first
   - Add hf:// support once basics work
