# Config Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the config module to use a simpler nested dictionary design with Mapping protocol support for pyiceberg Identifier keys.

**Architecture:** Replace the current Catalog/Namespace/Table dataclass hierarchy with a single Config class that wraps a nested dictionary. The Config class implements the Mapping protocol and accepts pyiceberg Identifier keys (tuples of strings like `('ns', 'subns', 'table')`), mapping them to table configuration dictionaries.

**Tech Stack:** Python dataclasses, collections.abc.Mapping, pyiceberg.typedef.Identifier

---

## Task 1: Create New Config Class with Mapping Protocol

**Files:**
- Modify: [faceberg/config.py:1-193](faceberg/config.py#L1-L193)

**Step 1: Write failing tests for new Config class**

Add tests to verify the Config class implements Mapping protocol and handles Identifiers:

```python
def test_config_mapping_protocol():
    """Test Config implements mapping protocol."""
    from collections.abc import Mapping
    from pyiceberg.typedef import Identifier

    config = Config(uri=".faceberg")
    assert isinstance(config, Mapping)

    # Test __getitem__
    config[Identifier(("ns", "table"))] = {"dataset": "org/repo", "config": "default"}
    assert config[Identifier(("ns", "table"))] == {"dataset": "org/repo", "config": "default"}

    # Test __len__
    assert len(config) == 1

    # Test __iter__
    assert list(config) == [("ns", "table")]


def test_config_nested_identifier():
    """Test Config handles nested identifiers."""
    from pyiceberg.typedef import Identifier

    config = Config(uri=".faceberg")
    config[Identifier(("ns", "subns", "table"))] = {"dataset": "org/repo", "config": "default"}

    assert config[Identifier(("ns", "subns", "table"))] == {"dataset": "org/repo", "config": "default"}
    assert len(config) == 1


def test_config_contains():
    """Test Config __contains__ method."""
    from pyiceberg.typedef import Identifier

    config = Config(uri=".faceberg")
    config[Identifier(("ns", "table"))] = {"dataset": "org/repo", "config": "default"}

    assert Identifier(("ns", "table")) in config
    assert Identifier(("ns", "other")) not in config
```

**Step 2: Run tests to verify they fail**

Run: `pytest faceberg/tests/test_config.py::test_config_mapping_protocol -xvs`
Expected: FAIL with "Config does not implement Mapping"

**Step 3: Implement new Config class**

Replace the existing Catalog, Namespace, and Table classes with a single Config class:

```python
"""Config module for Faceberg catalog configuration management."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterator

import yaml
from pyiceberg.typedef import Identifier


class Config(Mapping):
    """Config store - nested dictionary with Mapping protocol for Identifier keys.

    Stores table configurations indexed by pyiceberg Identifiers (tuples of strings).
    Identifiers can be nested: ('ns', 'table') or ('ns', 'subns', 'table').

    Internal structure:
    {
        'uri': 'file:///path/to/catalog',
        'data': {
            ('ns', 'table'): {'dataset': 'org/repo', 'config': 'default'},
            ('ns', 'subns', 'table'): {'dataset': 'org/repo2', 'config': 'custom'}
        }
    }
    """

    def __init__(self, uri: str, data: Dict[tuple, Dict[str, str]] | None = None):
        """Initialize Config with URI and optional data.

        Args:
            uri: Catalog URI
            data: Optional mapping of Identifier tuples to table config dicts
        """
        self.uri = uri
        self._data: Dict[tuple, Dict[str, str]] = data or {}

    def __getitem__(self, key: Identifier | tuple | str) -> Dict[str, str]:
        """Get table configuration by Identifier.

        Args:
            key: Identifier (tuple of strings) or single string

        Returns:
            Table configuration dictionary

        Raises:
            KeyError: If table doesn't exist
        """
        # Normalize key to tuple
        if isinstance(key, str):
            key_tuple = tuple(key.split('.'))
        else:
            key_tuple = tuple(key)

        return self._data[key_tuple]

    def __setitem__(self, key: Identifier | tuple | str, value: Dict[str, str]) -> None:
        """Set table configuration by Identifier.

        Args:
            key: Identifier (tuple of strings) or single string
            value: Table configuration dictionary with 'dataset' and 'config' keys
        """
        # Normalize key to tuple
        if isinstance(key, str):
            key_tuple = tuple(key.split('.'))
        else:
            key_tuple = tuple(key)

        self._data[key_tuple] = value

    def __delitem__(self, key: Identifier | tuple | str) -> None:
        """Delete table configuration by Identifier.

        Args:
            key: Identifier (tuple of strings) or single string

        Raises:
            KeyError: If table doesn't exist
        """
        # Normalize key to tuple
        if isinstance(key, str):
            key_tuple = tuple(key.split('.'))
        else:
            key_tuple = tuple(key)

        del self._data[key_tuple]

    def __len__(self) -> int:
        """Return number of tables in config."""
        return len(self._data)

    def __iter__(self) -> Iterator[tuple]:
        """Iterate over Identifier tuples."""
        return iter(self._data)

    def __contains__(self, key: object) -> bool:
        """Check if Identifier exists in config."""
        # Normalize key to tuple
        if isinstance(key, str):
            key_tuple = tuple(key.split('.'))
        elif isinstance(key, (tuple, list)):
            key_tuple = tuple(key)
        else:
            return False

        return key_tuple in self._data

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from YAML file.

        Args:
            path: Path to faceberg.yml file

        Returns:
            Parsed Config

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError("Config file is empty")

        # Extract URI if present
        uri = data.pop("uri", None)
        if not uri:
            raise ValueError("Missing required 'uri' field in config")

        # Parse namespaces and tables into flat dictionary
        config_data = {}
        for namespace_name, namespace_tables in data.items():
            # Validate namespace name
            if not namespace_name:
                raise ValueError("Namespace name cannot be empty")

            # Check for reserved names
            if namespace_name == "catalog":
                raise ValueError("Cannot use 'catalog' as namespace name (reserved)")

            # Validate namespace name format (alphanumeric, underscore, hyphen)
            import re
            if not re.match(r"^[a-zA-Z0-9_-]+$", namespace_name):
                raise ValueError(
                    f"Invalid namespace name '{namespace_name}'. "
                    "Must contain only alphanumeric characters, underscores, or hyphens"
                )

            if not isinstance(namespace_tables, dict):
                raise ValueError(f"Namespace '{namespace_name}' must be a dict of tables")

            # Parse tables in this namespace
            for table_name, table_data in namespace_tables.items():
                if not isinstance(table_data, dict):
                    raise ValueError(
                        f"Table '{namespace_name}.{table_name}' must be a dict with 'dataset' field"
                    )

                if "dataset" not in table_data:
                    raise ValueError(f"Missing 'dataset' in {namespace_name}.{table_name}")

                # Store with tuple key
                identifier = (namespace_name, table_name)
                config_data[identifier] = {
                    "dataset": table_data["dataset"],
                    "config": table_data.get("config", "default"),
                }

        return cls(uri=uri, data=config_data)

    def to_yaml(self, path: str | Path) -> None:
        """Save config to YAML file.

        Args:
            path: Path to save faceberg.yml file
        """
        path = Path(path)

        data = {"uri": self.uri}

        # Group tables by namespace
        namespaces: Dict[str, Dict[str, Any]] = {}
        for identifier, table_data in self._data.items():
            # identifier is a tuple like ('ns', 'table') or ('ns', 'subns', 'table')
            if len(identifier) < 2:
                continue  # Skip invalid identifiers

            namespace = identifier[0]
            # For nested identifiers, join the rest with '.'
            table_name = '.'.join(identifier[1:])

            if namespace not in namespaces:
                namespaces[namespace] = {}

            namespaces[namespace][table_name] = table_data

        # Add each namespace as a top-level key
        data.update(namespaces)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
```

**Step 4: Run tests to verify they pass**

Run: `pytest faceberg/tests/test_config.py::test_config_mapping_protocol faceberg/tests/test_config.py::test_config_nested_identifier faceberg/tests/test_config.py::test_config_contains -xvs`
Expected: PASS

**Step 5: Commit**

```bash
git add faceberg/config.py faceberg/tests/test_config.py
git commit -m "feat: implement Config class with Mapping protocol

Replace Catalog/Namespace/Table with single Config class that:
- Implements collections.abc.Mapping protocol
- Uses nested dict with Identifier tuple keys
- Supports pyiceberg Identifier ('ns', 'table') syntax
- Maintains YAML serialization compatibility

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Update Existing Config Tests

**Files:**
- Modify: [faceberg/tests/test_config.py:1-421](faceberg/tests/test_config.py#L1-L421)

**Step 1: Update imports and remove old class tests**

Remove TestTable, TestNamespace classes and update imports:

```python
"""Tests for faceberg.config module."""

import pytest
import yaml

from faceberg.config import Config
```

**Step 2: Update TestCatalog to TestConfig and adapt tests**

Rename class and update all tests to use new Config API:

```python
class TestConfig:
    """Tests for Config class and parsing."""

    def test_config_creation(self):
        """Test Config creation."""
        from pyiceberg.typedef import Identifier

        config = Config(uri=".faceberg")
        config[Identifier(("ns1", "table1"))] = {"dataset": "org/repo", "config": "default"}

        assert config.uri == ".faceberg"
        assert len(config) == 1
        assert Identifier(("ns1", "table1")) in config

    def test_from_yaml_valid_config(self, tmp_path):
        """Test parsing valid YAML config."""
        yaml_content = """
uri: .faceberg

namespace1:
  table1:
    dataset: org/repo1
    config: config1
  table2:
    dataset: org/repo2
    config: config2

namespace2:
  table3:
    dataset: org/repo3
    # config defaults to 'default'
"""
        config_file = tmp_path / "test_config.yml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(config_file)

        assert len(config) == 3

        # Check tables using tuple keys
        assert ("namespace1", "table1") in config
        assert config[("namespace1", "table1")] == {"dataset": "org/repo1", "config": "config1"}
        assert config[("namespace1", "table2")] == {"dataset": "org/repo2", "config": "config2"}
        assert config[("namespace2", "table3")] == {"dataset": "org/repo3", "config": "default"}

    # ... (continue adapting other tests)
```

Update all test methods to:
- Use tuple keys instead of namespace/table methods
- Use `config[("ns", "table")]` instead of `config.get_table("ns", "table")`
- Use `("ns", "table") in config` instead of `config.has_table("ns", "table")`
- Use `del config[("ns", "table")]` instead of `config.delete_table("ns", "table")`

**Step 3: Run tests to verify they pass**

Run: `pytest faceberg/tests/test_config.py -xvs`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add faceberg/tests/test_config.py
git commit -m "test: update config tests for new Config class

Adapt all tests to use Mapping protocol API:
- Replace get_table/set_table with dict access
- Replace has_table with 'in' operator
- Replace delete_table with del operator
- Use tuple keys for identifiers

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Update __init__.py Exports

**Files:**
- Modify: [faceberg/__init__.py:1-34](faceberg/__init__.py#L1-L34)

**Step 1: Write test for updated exports**

```python
def test_config_export():
    """Test that Config is exported but not Table/Namespace."""
    import faceberg

    assert hasattr(faceberg, 'Config')
    assert not hasattr(faceberg, 'Table')
    assert not hasattr(faceberg, 'Namespace')
    assert not hasattr(faceberg, 'Catalog')
```

**Step 2: Run test to verify it fails**

Run: `pytest faceberg/tests/test_config.py::test_config_export -xvs`
Expected: FAIL

**Step 3: Update exports in __init__.py**

```python
from faceberg.config import Config
```

Remove lines 11, 20-22 (Catalog, Namespace, Table imports and exports).

**Step 4: Run test to verify it passes**

Run: `pytest faceberg/tests/test_config.py::test_config_export -xvs`
Expected: PASS

**Step 5: Commit**

```bash
git add faceberg/__init__.py
git commit -m "refactor: update exports to use Config instead of Catalog

Remove Catalog, Namespace, Table from public API.
Export new Config class.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Update catalog.py to Use New Config

**Files:**
- Modify: [faceberg/catalog.py:31-32](faceberg/catalog.py#L31-L32)
- Modify: [faceberg/catalog.py:289](faceberg/catalog.py#L289)
- Modify: [faceberg/catalog.py:305-314](faceberg/catalog.py#L305-L314)
- Modify: [faceberg/catalog.py:468-480](faceberg/catalog.py#L468-L480)
- Modify: [faceberg/catalog.py:502-516](faceberg/catalog.py#L502-L516)
- Modify: [faceberg/catalog.py:527-529](faceberg/catalog.py#L527-L529)
- Modify: [faceberg/catalog.py:596-663](faceberg/catalog.py#L596-L663)
- Modify: [faceberg/catalog.py:682-686](faceberg/catalog.py#L682-L686)
- Modify: [faceberg/catalog.py:744-760](faceberg/catalog.py#L744-L760)
- Modify: [faceberg/catalog.py:778-785](faceberg/catalog.py#L778-L785)
- Modify: [faceberg/catalog.py:800-811](faceberg/catalog.py#L800-L811)
- Modify: [faceberg/catalog.py:836-890](faceberg/catalog.py#L836-L890)
- Modify: [faceberg/catalog.py:907-908](faceberg/catalog.py#L907-L908)
- Modify: [faceberg/catalog.py:1068-1070](faceberg/catalog.py#L1068-L1070)
- Modify: [faceberg/catalog.py:1134-1141](faceberg/catalog.py#L1134-L1141)
- Modify: [faceberg/catalog.py:1179-1196](faceberg/catalog.py#L1179-L1196)
- Modify: [faceberg/catalog.py:1226](faceberg/catalog.py#L1226)
- Modify: [faceberg/catalog.py:1266-1278](faceberg/catalog.py#L1266-L1278)
- Modify: [faceberg/catalog.py:1332-1342](faceberg/catalog.py#L1332-L1342)
- Modify: [faceberg/catalog.py:1396-1419](faceberg/catalog.py#L1396-L1419)
- Modify: [faceberg/catalog.py:1565-1610](faceberg/catalog.py#L1565-L1610)

**Step 1: Update import and type hints**

Change line 31-32:
```python
from faceberg import config as cfg
```

To:
```python
from faceberg.config import Config
```

Update all type hints from `cfg.Catalog` to `Config`.

**Step 2: Update _load_config return type**

Change method signature at lines 305, 1396, 1565:
```python
def _load_config(self) -> Config:
```

**Step 3: Update config initialization**

Change line 289:
```python
self._config = Config(uri=self.uri)
```

**Step 4: Update all config access patterns**

Replace all occurrences of:
- `cfg.Catalog(uri=..., namespaces={})` → `Config(uri=...)`
- `self._config.namespaces[ns_str]` → Check if any identifier with first element `ns_str` exists
- `self._config.namespaces[ns_str].tables[table_name]` → `self._config[(ns_str, table_name)]`
- `cfg.Namespace(tables={})` → Remove (no longer needed)
- `cfg.Table(dataset=..., config=...)` → `{"dataset": ..., "config": ...}`
- `self._config.set_table(ns, table, cfg.Table(...))` → `self._config[(ns, table)] = {"dataset": ..., "config": ...}`
- `self._config.get_table(ns, table)` → `self._config[(ns, table)]`
- `self._config.has_table(ns, table)` → `(ns, table) in self._config`
- `self._config.delete_table(ns, table)` → `del self._config[(ns, table)]`

**Step 5: Update namespace operations**

For `create_namespace` (line 468-480):
```python
# Check if namespace already exists by looking for any table with this namespace
if any(identifier[0] == ns_str for identifier in self._config):
    raise NamespaceAlreadyExistsError(f"Namespace {ns_str} already exists")

# Create the directory in staging (no config change needed for empty namespace)
ns_dir = self._staging_dir / ns_str
ns_dir.mkdir(parents=True, exist_ok=True)

# Only save config if needed (we might just save later when tables are added)
```

For `drop_namespace` (line 502-516):
```python
# Check if namespace has tables
tables_in_ns = [id for id in self._config if id[0] == ns_str]
if tables_in_ns:
    raise NamespaceNotEmptyError(f"Namespace {ns_str} is not empty")

# No config change needed - empty namespace doesn't exist in config

# Remove the directory
ns_dir = self._staging_dir / ns_str
if ns_dir.exists():
    ns_dir.rmdir()
```

For `list_namespaces` (line 527-529):
```python
# Get unique namespaces from all identifiers
config = self._load_config()
namespaces = {identifier[0] for identifier in config}
return [tuple([ns]) for ns in sorted(namespaces)]
```

For `list_tables` (line 778-785):
```python
# Convert to string
if isinstance(namespace, str):
    ns_str = namespace
else:
    ns_str = ".".join(namespace)

config = self._load_config()
tables = [identifier for identifier in config if identifier[0] == ns_str]
return [tuple([identifier[0], '.'.join(identifier[1:])]) for identifier in tables]
```

**Step 6: Update sync_datasets**

Line 1068-1070:
```python
try:
    config_table = config[(target_namespace, target_table)]
except KeyError:
    raise ValueError(f"Table {table_name} not found in config")
```

Line 1075-1079:
```python
# Process all tables
tables_to_process = [
    (identifier[0], '.'.join(identifier[1:]), table_data)
    for identifier, table_data in config.items()
]
```

**Step 7: Run existing catalog tests**

Run: `pytest faceberg/tests/test_catalog.py -xvs`
Expected: Tests should pass with new Config implementation

**Step 8: Commit**

```bash
git add faceberg/catalog.py
git commit -m "refactor: update catalog to use new Config class

Replace all config access patterns:
- Use dict operations instead of methods
- Use tuple keys for table identifiers
- Use dict literals instead of Table dataclass
- Simplify namespace operations

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Update Remaining Files

**Files:**
- Modify: All remaining files that import from faceberg.config

**Step 1: Search for remaining imports**

Run: `grep -r "from faceberg.config import" --include="*.py" faceberg/`
Run: `grep -r "from faceberg import.*Catalog\|Namespace\|Table" --include="*.py" faceberg/`

**Step 2: Update each file found**

For each file:
- Replace `from faceberg.config import Catalog, ...` with `from faceberg.config import Config`
- Replace any Catalog/Namespace/Table usage with Config dict operations
- Update any dataclass instantiations to dict literals

**Step 3: Run full test suite**

Run: `pytest faceberg/ -xvs`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add faceberg/
git commit -m "refactor: complete migration to Config class

Update all remaining files to use new Config API.
Remove all references to Catalog, Namespace, Table classes.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Final Verification and Cleanup

**Files:**
- Verify: All Python files in faceberg/

**Step 1: Run full test suite with coverage**

Run: `pytest faceberg/ -xvs --cov=faceberg --cov-report=term-missing`
Expected: All tests PASS with good coverage

**Step 2: Check for any remaining old API usage**

Run: `grep -r "Catalog\|Namespace\|Table" --include="*.py" faceberg/ | grep -v "# " | grep -v 'from pyiceberg'`
Expected: Only legitimate usage (like "FacebergCatalog" class name)

**Step 3: Run type checker**

Run: `mypy faceberg/ --ignore-missing-imports`
Expected: No type errors

**Step 4: Final commit if any fixes needed**

```bash
git add .
git commit -m "fix: address final type errors and test issues

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Notes

- The Config class is intentionally simple - just a wrapper around a dict
- Identifier tuples are normalized in __getitem__/__setitem__ to handle strings
- YAML format remains backward compatible
- The Mapping protocol makes Config behave like a built-in dict
- Nested identifiers like ('ns', 'subns', 'table') are supported and serialized as 'ns.subns.table' in YAML
