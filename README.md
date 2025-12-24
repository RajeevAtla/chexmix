# chexmix

[![pytest](https://github.com/RajeevAtla/chexmix/actions/workflows/pytest.yml/badge.svg)](https://github.com/RajeevAtla/chexmix/actions/workflows/pytest.yml)
[![ruff](https://github.com/RajeevAtla/chexmix/actions/workflows/ruff.yml/badge.svg)](https://github.com/RajeevAtla/chexmix/actions/workflows/ruff.yml)
[![ty](https://github.com/RajeevAtla/chexmix/actions/workflows/ty.yml/badge.svg)](https://github.com/RajeevAtla/chexmix/actions/workflows/ty.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

JAX/Flax-NNX + PGX + MCTS (mctx) self-play chess.

## Install (CPU)

```bash
uv sync --group dev
```

## Install (CUDA 13)

```bash
uv sync --group cuda --group dev
```

## Train

```bash
uv run python -m . train --config config/default.toml
```

## Tests

```bash
uv run python -m pytest
```

## Lint and type check

```bash
uv run ruff check .
uv run ruff format .
uv run ty check
```
