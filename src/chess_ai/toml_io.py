"""
TOML-only IO with strict typing.

Constraints:
- Reading: use tomllib
- Writing: implement a minimal TOML serializer for our supported types
- Deterministic output ordering
- No Any / no untyped dicts
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import tomllib

TomlScalar: TypeAlias = str | int | float | bool
TomlValue: TypeAlias = TomlScalar | list["TomlValue"] | dict[str, "TomlValue"]


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
    data = tomllib.loads(path.read_text(encoding="utf-8"))
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


def _validate_toml_dict(raw: dict[str, object]) -> dict[str, TomlValue]:
    """Validate TOML data without leaking `object` to callers."""
    validated: dict[str, TomlValue] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            nested = _validate_toml_dict(value)
            validated[key] = nested
            continue

        validated[key] = _validate_toml_value(value)
    return validated


def _validate_toml_value(value: object) -> TomlValue:
    if isinstance(value, (str, int, float, bool)):
        return value

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

    for key, value in table.items():
        if isinstance(value, dict):
            table_items[key] = value
        else:
            scalar_items[key] = value

    lines: list[str] = []
    if prefix:
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
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float)):
        return repr(value)

    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', "\\\"")
        return f"\"{escaped}\""

    if isinstance(value, list):
        rendered = ", ".join(_format_value(item) for item in value)
        return f"[{rendered}]"

    raise ValueError("unsupported TOML value type")
