"""
TOML-only IO with strict typing.

Constraints:
- Reading: use tomllib
- Writing: implement a minimal TOML serializer for our supported types
- Deterministic output ordering
- No Any / no untyped dicts
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TypeGuard, cast

type TomlScalar = str | int | float | bool
type TomlValue = TomlScalar | list["TomlValue"] | dict[str, "TomlValue"]
type TomlRawValue = (
    str | int | float | bool | list["TomlRawValue"] | dict[str, "TomlRawValue"]
)
type TomlRawTable = dict[str, TomlRawValue]


def load_toml(path: Path) -> dict[str, TomlValue]:
    """Load a TOML file into a strictly-typed nested dictionary.

    Args:
        path: Path to TOML file.

    Returns:
        Parsed TOML as nested dict[str, TomlValue].

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If parsed content contains unsupported types.
    """
    data = cast(TomlRawTable, tomllib.loads(path.read_text(encoding="utf-8")))
    if not _is_str_key_dict(data):
        raise ValueError("TOML root must be a table with string keys.")
    return _validate_toml_dict(data)


def dump_toml(data: dict[str, TomlValue]) -> str:
    """Serialize a TOML dictionary deterministically.

    Args:
        data: Nested dict with TomlValue leaves.

    Returns:
        TOML string with stable ordering.

    Raises:
        ValueError: If data contains unsupported types.
    """
    return _dumps_table(data, prefix="")


def save_toml(path: Path, data: dict[str, TomlValue]) -> None:
    """Write TOML to disk with stable ordering."""
    path.write_text(dump_toml(data), encoding="utf-8")


def _is_str_key_dict(value: TomlRawValue) -> TypeGuard[TomlRawTable]:
    """Check whether a value is a dict with string keys.

    Args:
        value: Candidate value.

    Returns:
        True if the value is a dict and all keys are strings.
    """
    # Keep a narrow check to avoid leaking object-typed dicts.
    return isinstance(value, dict) and all(
        isinstance(key, str) for key in value
    )


def _validate_toml_dict(raw: TomlRawTable) -> dict[str, TomlValue]:
    """Validate TOML data without leaking `object` to callers."""
    validated: dict[str, TomlValue] = {}
    # Validate nested tables and scalars recursively.
    for key, value in raw.items():
        if _is_str_key_dict(value):
            nested = _validate_toml_dict(cast(TomlRawTable, value))
            validated[key] = nested
            continue

        validated[key] = _validate_toml_value(value)
    return validated


def _validate_toml_value(value: TomlRawValue) -> TomlValue:
    """Validate a TOML value against allowed types.

    Args:
        value: Parsed TOML value.

    Returns:
        Validated TomlValue.

    Raises:
        ValueError: If the value is an unsupported type.
    """
    # Scalars are allowed directly.
    if isinstance(value, str | int | float | bool):
        return value

    # Lists are allowed but cannot contain dicts.
    if isinstance(value, list):
        validated_list: list[TomlValue] = []
        for item in value:
            if isinstance(item, dict):
                raise ValueError("dict values in lists are not supported")
            validated_list.append(_validate_toml_value(item))
        return validated_list

    raise ValueError(f"unsupported TOML value type: {type(value)}")


def _dumps_table(table: dict[str, TomlValue], prefix: str) -> str:
    """Dump a TOML table and nested subtables with deterministic ordering."""
    scalar_items: dict[str, TomlValue] = {}
    table_items: dict[str, dict[str, TomlValue]] = {}

    # Partition tables from scalars for deterministic ordering.
    for key, value in table.items():
        if isinstance(value, dict):
            table_items[key] = value
        else:
            scalar_items[key] = value

    lines: list[str] = []
    if prefix:
        # Emit table header for nested tables.
        lines.append(f"[{prefix}]")

    for key in sorted(scalar_items):
        value = scalar_items[key]
        lines.append(f"{key} = {_format_value(value)}")

    for key in sorted(table_items):
        subtable = table_items[key]
        if lines:
            lines.append("")
        next_prefix = f"{prefix}.{key}" if prefix else key
        lines.append(_dumps_table(subtable, prefix=next_prefix).rstrip("\n"))

    return "\n".join(lines) + "\n"


def _format_value(value: TomlValue) -> str:
    """Format a TOML value into a string representation.

    Args:
        value: TomlValue to format.

    Returns:
        Serialized TOML literal.

    Raises:
        ValueError: If the value type is unsupported.
    """
    # Handle booleans first so ints don't swallow them.
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, int | float):
        return repr(value)

    if isinstance(value, str):
        # Escape backslashes and quotes to keep output valid.
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    if isinstance(value, list):
        # Format lists recursively.
        rendered = ", ".join(_format_value(item) for item in value)
        return f"[{rendered}]"

    raise ValueError("unsupported TOML value type")
