"""Schema inference from Parquet files."""

import pyarrow as pa
import pyarrow.parquet as pq
from pyiceberg.schema import Schema
from pyiceberg.types import (
    BinaryType,
    BooleanType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    ListType,
    LongType,
    MapType,
    NestedField,
    StringType,
    StructType,
    TimestampType,
    TimeType,
)


def arrow_to_iceberg_type(arrow_type: pa.DataType):
    """Convert PyArrow type to Iceberg type.

    Args:
        arrow_type: PyArrow data type

    Returns:
        Corresponding Iceberg type
    """
    if pa.types.is_boolean(arrow_type):
        return BooleanType()
    elif pa.types.is_int32(arrow_type):
        return IntegerType()
    elif pa.types.is_int64(arrow_type):
        return LongType()
    elif pa.types.is_float32(arrow_type):
        return FloatType()
    elif pa.types.is_float64(arrow_type):
        return DoubleType()
    elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return StringType()
    elif pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
        return BinaryType()
    elif pa.types.is_date(arrow_type):
        return DateType()
    elif pa.types.is_time(arrow_type):
        return TimeType()
    elif pa.types.is_timestamp(arrow_type):
        return TimestampType()
    elif pa.types.is_decimal(arrow_type):
        return DecimalType(arrow_type.precision, arrow_type.scale)
    elif pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        element_type = arrow_to_iceberg_type(arrow_type.value_type)
        return ListType(element_id=1, element_type=element_type, element_required=False)
    elif pa.types.is_struct(arrow_type):
        fields = []
        for i, field in enumerate(arrow_type):
            field_type = arrow_to_iceberg_type(field.type)
            fields.append(
                NestedField(
                    field_id=i + 1,
                    name=field.name,
                    field_type=field_type,
                    required=not field.nullable,
                )
            )
        return StructType(*fields)
    elif pa.types.is_map(arrow_type):
        key_type = arrow_to_iceberg_type(arrow_type.key_type)
        value_type = arrow_to_iceberg_type(arrow_type.item_type)
        return MapType(
            key_id=1,
            key_type=key_type,
            value_id=2,
            value_type=value_type,
            value_required=False,
        )
    else:
        # Default to string for unknown types
        return StringType()


def infer_schema_from_parquet(file_path: str) -> Schema:
    """Infer Iceberg schema from a Parquet file.

    Args:
        file_path: Path to Parquet file (can be local or hf:// URI)

    Returns:
        Iceberg Schema
    """
    # Read Parquet schema
    parquet_file = pq.ParquetFile(file_path)
    arrow_schema = parquet_file.schema_arrow

    # Convert to Iceberg schema
    fields = []
    for i, field in enumerate(arrow_schema):
        iceberg_type = arrow_to_iceberg_type(field.type)
        fields.append(
            NestedField(
                field_id=i + 1,
                name=field.name,
                field_type=iceberg_type,
                required=not field.nullable,
            )
        )

    return Schema(*fields)
