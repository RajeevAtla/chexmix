# chexmix

[![ci](https://github.com/RajeevAtla/chexmix/actions/workflows/ci.yml/badge.svg)](https://github.com/RajeevAtla/chexmix/actions/workflows/ci.yml)

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
PYTHONPATH=src uv run python -m cli train --config config/default.toml
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

