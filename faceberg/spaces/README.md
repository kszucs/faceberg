---
title: {space_display_name}
emoji: 🗃️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Faceberg REST Catalog: {space_display_name}

This Space serves an Apache Iceberg REST catalog powered by [Faceberg](https://github.com/kszucs/faceberg).

## Configuration

This Space is configured with the following environment variables:

- `CATALOG_URI`: `{catalog_uri}`
- `HF_TOKEN`: (set as secret if needed for private catalogs)

## API Endpoints

The catalog is accessible at:
```
https://{api_url}.hf.space/v1
```

### Available Endpoints

- `GET /v1/config` - Get catalog configuration
- `GET /v1/namespaces` - List all namespaces
- `GET /v1/namespaces/{{namespace}}` - Load namespace properties
- `HEAD /v1/namespaces/{{namespace}}` - Check if namespace exists
- `GET /v1/namespaces/{{namespace}}/tables` - List tables in namespace
- `GET /v1/namespaces/{{namespace}}/tables/{{table}}` - Load table metadata
- `HEAD /v1/namespaces/{{namespace}}/tables/{{table}}` - Check if table exists

## Usage Example

### Connect with PyIceberg

```python
from pyiceberg.catalog.rest import RestCatalog

# Connect to the catalog
catalog = RestCatalog(
    name="faceberg",
    uri="https://{api_url}.hf.space",
)

# List namespaces
namespaces = catalog.list_namespaces()
print(f"Namespaces: {{namespaces}}")

# List tables
for ns in namespaces:
    tables = catalog.list_tables(ns)
    print(f"Tables in {{ns}}: {{tables}}")
```

### Test with curl

```bash
# Get catalog configuration
curl https://{api_url}.hf.space/v1/config

# List namespaces
curl https://{api_url}.hf.space/v1/namespaces

# List tables in a namespace
curl https://{api_url}.hf.space/v1/namespaces/my_namespace/tables
```

## About Faceberg

Faceberg is a lightweight Python library for working with Apache Iceberg catalogs on HuggingFace Hub.

- **GitHub**: [https://github.com/kszucs/faceberg](https://github.com/kszucs/faceberg)
- **Apache Iceberg**: [https://iceberg.apache.org/](https://iceberg.apache.org/)

---

*Server listening on 0.0.0.0:7860*
