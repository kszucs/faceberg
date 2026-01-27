---
title: Faceberg REST Catalog
emoji: 🗃️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Faceberg REST Catalog

An Apache Iceberg REST catalog that syncs table metadata from Hugging Face datasets using [Faceberg](https://github.com/kszucs/faceberg).

This Space provides a read-only REST API for querying Iceberg tables stored as datasets on Hugging Face Hub. Connect using any Iceberg-compatible client (PyIceberg, Spark, etc.) to access your data.

## About

Faceberg enables storing Apache Iceberg table metadata directly on Hugging Face Hub as datasets, making your data lake tables easily shareable and version-controlled.

Learn more:
- [Faceberg on GitHub](https://github.com/kszucs/faceberg)
- [Apache Iceberg](https://iceberg.apache.org/)
