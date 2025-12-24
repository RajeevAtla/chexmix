"""Tests for TOML IO validation and formatting."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from toml_io import (
    TomlValue,
    _validate_toml_dict,
    dump_toml,
    load_toml,
    save_toml,
)


def test_dump_toml_deterministic_order() -> None:
    """dump_toml produces deterministic key ordering."""
    # Use mixed-order keys to validate sorting.
    data = {
        "b": 1,
        "a": 2,
        "section": {
            "z": "hi",
            "y": [1, 2],
        },
    }
    expected = 'a = 2\nb = 1\n\n[section]\ny = [1, 2]\nz = "hi"\n'
    assert dump_toml(data) == expected


def test_save_and_load_toml_roundtrip(tmp_path: Path) -> None:
    """save_toml/load_toml roundtrip data correctly."""
    # Write and reload a small config.
    data = {
        "run": {
            "name": "demo",
            "seed": 42,
            "enabled": True,
        },
        "values": [1, 2, 3],
    }
    path = tmp_path / "config.toml"
    save_toml(path, data)
    loaded = load_toml(path)
    assert loaded == data


def test_validate_rejects_unsupported_types() -> None:
    """_validate_toml_dict rejects unsupported types."""
    # Object values should be rejected.
    with pytest.raises(ValueError):
        _validate_toml_dict({"bad": object()})

    # Dicts inside lists should be rejected.
    with pytest.raises(ValueError):
        _validate_toml_dict({"bad": [1, {"nested": 2}]})


def test_load_toml_rejects_non_string_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_toml rejects non-string keys from parser."""
    import toml_io as toml_io

    # Force tomllib to return non-string keys.
    path = tmp_path / "bad.toml"
    path.write_text("", encoding="utf-8")
    monkeypatch.setattr(toml_io.tomllib, "loads", lambda _: {1: "bad"})
    with pytest.raises(ValueError, match="string keys"):
        _ = toml_io.load_toml(path)


def test_dump_toml_rejects_invalid_value() -> None:
    """dump_toml rejects invalid TomlValue types."""
    with pytest.raises(ValueError, match="unsupported TOML value type"):
        bad = cast(dict[str, TomlValue], {"bad": object()})
        _ = dump_toml(bad)
