from __future__ import annotations

from pathlib import Path

import pytest

from chess_ai.toml_io import _validate_toml_dict, dump_toml, load_toml, save_toml


def test_dump_toml_deterministic_order() -> None:
    data = {
        "b": 1,
        "a": 2,
        "section": {
            "z": "hi",
            "y": [1, 2],
        },
    }
    expected = (
        "a = 2\n"
        "b = 1\n"
        "\n"
        "[section]\n"
        "y = [1, 2]\n"
        "z = \"hi\"\n"
    )
    assert dump_toml(data) == expected


def test_save_and_load_toml_roundtrip(tmp_path: Path) -> None:
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
    with pytest.raises(ValueError):
        _validate_toml_dict({"bad": object()})

    with pytest.raises(ValueError):
        _validate_toml_dict({"bad": [1, {"nested": 2}]})
