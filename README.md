# chexmix

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
uv run python -m chess_ai.cli train --config config/default.toml
```

## Tests

```bash
uv run pytest
```

## Lint and type check

```bash
uv run ruff check .
uv run ruff format .
uv run ty check
```
