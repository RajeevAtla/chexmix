"""
Command-line entrypoints.

Commands:
- train: run self-play + learning loop
- eval: evaluate checkpoints (arena matches)
"""

from __future__ import annotations

import argparse
import platform
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from chex_types import PRNGKey, Step
from env.pgx_chess import make_chess_env
from eval.arena import MatchResult
from eval.elo import expected_score, update_elo
from mcts.planner import MctsConfig
from model.chess_transformer import ChessTransformer
from model.nnx_blocks import TransformerConfig
from paths import RunPaths
from pgn.writer import PgnHeaders, format_pgn, write_pgn_file
from rng import RngStream
from selfplay.buffer import ReplayBuffer, ReplayConfig
from selfplay.rollout import SelfPlayConfig, generate_selfplay_trajectories
from selfplay.trajectory import Trajectory
from toml_io import TomlValue, load_toml, save_toml
from train.checkpointing import (
    CheckpointConfig,
    make_checkpoint_manager,
    restore_latest,
    save_checkpoint,
)
from train.learner import TrainConfig, train_step
from train.logging import Metrics, write_metrics_snapshot
from train.losses import LossConfig
from train.optimizer import OptimConfig, make_optimizer
from train.state import TrainState


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="chexmix")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run training loop")
    train_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to TOML config",
    )

    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to TOML config",
    )

    return parser


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Run-level configuration for CLI training/eval.

    Attributes:
        name: Run name used in artifact paths.
        seed: Base RNG seed.
        devices: Requested device count.
        log_every_steps: Metrics logging cadence.
        checkpoint_every_steps: Checkpoint cadence.
        pgn_every_games: PGN snapshot cadence in games.
        max_runtime_minutes: Hard wall-clock limit in minutes.
    """

    name: str
    seed: int
    devices: int
    log_every_steps: int
    checkpoint_every_steps: int
    pgn_every_games: int
    max_runtime_minutes: int


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Environment-level configuration for self-play."""

    max_moves: int


def _get_int(table: dict[str, TomlValue], key: str) -> int:
    """Fetch a required integer from a TOML table.

    Args:
        table: TOML table to read from.
        key: Key to fetch.

    Returns:
        Integer value.

    Raises:
        ValueError: If the key is missing or not an int.
    """
    value = table.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Missing int key: {key}")
    return value


def _get_float(table: dict[str, TomlValue], key: str) -> float:
    """Fetch a required float from a TOML table.

    Args:
        table: TOML table to read from.
        key: Key to fetch.

    Returns:
        Float value (ints coerced to float).

    Raises:
        ValueError: If the key is missing or not a float-like value.
    """
    value = table.get(key)
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, float):
        raise ValueError(f"Missing float key: {key}")
    return value


def _get_table(data: dict[str, TomlValue], key: str) -> dict[str, TomlValue]:
    """Fetch a required TOML table.

    Args:
        data: Root TOML dict.
        key: Table key to fetch.

    Returns:
        Nested table dict.

    Raises:
        ValueError: If the key is missing or not a table.
    """
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing TOML table: {key}")
    return value


def _get_table_optional(
    data: dict[str, TomlValue], key: str
) -> dict[str, TomlValue] | None:
    """Fetch an optional TOML table.

    Args:
        data: Root TOML dict.
        key: Table key to fetch.

    Returns:
        Table dict if present, otherwise None.

    Raises:
        ValueError: If the key exists but is not a table.
    """
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Missing TOML table: {key}")
    return value


def _get_str(table: dict[str, TomlValue], key: str) -> str:
    """Fetch a required string from a TOML table.

    Args:
        table: TOML table to read from.
        key: Key to fetch.

    Returns:
        String value.

    Raises:
        ValueError: If the key is missing or not a string.
    """
    value = table.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Missing string key: {key}")
    return value


def _append_event(paths: RunPaths, event: dict[str, TomlValue]) -> None:
    """Append a run event to events.toml with stable numbering.

    Args:
        paths: RunPaths for the current run.
        event: Event payload to append.
    """
    # Load existing events to keep numbering monotonic.
    data = load_toml(paths.events_toml) if paths.events_toml.exists() else {}
    idx = 0
    for key in data:
        if key.startswith("event_"):
            try:
                idx = max(idx, int(key.split("_", 1)[1]))
            except ValueError:
                continue
    data[f"event_{idx + 1:04d}"] = event
    save_toml(paths.events_toml, data)


def _git_sha() -> str:
    """Resolve the current git SHA for run metadata.

    Returns:
        Full SHA string if available, otherwise "unknown".
    """
    head_path = Path(".git") / "HEAD"
    if not head_path.exists():
        return "unknown"
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref_path = Path(".git") / head.split(" ", 1)[1]
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
        return "unknown"
    return head


def _run_id(run_name: str) -> str:
    """Construct a UTC run_id with timestamp and name.

    Args:
        run_name: Human-readable run name.

    Returns:
        Run identifier string.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{run_name}"


def _write_start_event(paths: RunPaths, run_id: str, run_name: str) -> None:
    """Write a run start event.

    Args:
        paths: RunPaths for the current run.
        run_id: Generated run identifier.
        run_name: Run name from config.
    """
    event: dict[str, TomlValue] = {
        "event": "start",
        "run_id": run_id,
        "run_name": run_name,
        "started_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    _append_event(paths, event)


def _write_stop_event(
    paths: RunPaths, run_id: str, run_name: str, reason: str
) -> None:
    """Write a run stop event.

    Args:
        paths: RunPaths for the current run.
        run_id: Generated run identifier.
        run_name: Run name from config.
        reason: Stop reason label.
    """
    event: dict[str, TomlValue] = {
        "event": "stop",
        "run_id": run_id,
        "run_name": run_name,
        "stopped_utc": datetime.now(UTC).isoformat(),
        "reason": reason,
    }
    _append_event(paths, event)


def _write_bootstrap_artifacts(paths: RunPaths, run_name: str) -> None:
    """Write initial metrics and a stub PGN for an empty run.

    Args:
        paths: RunPaths for the current run.
        run_name: Run name for PGN headers.
    """
    # Write a zeroed metrics snapshot for tooling expectations.
    metrics = Metrics(
        step=0,
        loss_total=0.0,
        loss_policy=0.0,
        loss_value=0.0,
        value_mean=0.0,
        entropy_mean=0.0,
    )
    write_metrics_snapshot(paths, metrics)

    # Emit a placeholder PGN so the games directory is non-empty.
    headers = PgnHeaders(
        event=run_name,
        site="local",
        date=date.today(),
        round="1",
        white="selfplay",
        black="selfplay",
        result="*",
    )
    pgn = format_pgn(headers, moves=[])
    write_pgn_file(paths.games_dir / "game_0000000001.pgn", pgn)


def _parse_run_config(config: dict[str, TomlValue]) -> RunConfig:
    """Parse the run configuration from TOML.

    Args:
        config: Loaded config TOML data.

    Returns:
        Parsed RunConfig.
    """
    run_table = _get_table(config, "run")
    return RunConfig(
        name=_get_str(run_table, "name"),
        seed=_get_int(run_table, "seed"),
        devices=_get_int(run_table, "devices"),
        log_every_steps=_get_int(run_table, "log_every_steps"),
        checkpoint_every_steps=_get_int(run_table, "checkpoint_every_steps"),
        pgn_every_games=_get_int(run_table, "pgn_every_games"),
        max_runtime_minutes=_get_int(run_table, "max_runtime_minutes"),
    )


def _parse_env_config(config: dict[str, TomlValue]) -> EnvConfig:
    """Parse the environment configuration from TOML.

    Args:
        config: Loaded config TOML data.

    Returns:
        Parsed EnvConfig.
    """
    env_table = _get_table(config, "env")
    return EnvConfig(
        max_moves=_get_int(env_table, "max_moves"),
    )


def _split_batch(
    batch: dict[str, jax.Array], device_count: int
) -> dict[str, jax.Array]:
    """Shard a batch dict along the leading dimension for pmap.

    Args:
        batch: Host batch with leading dimension.
        device_count: Number of devices to shard across.

    Returns:
        Batch with leading axis split across devices.
    """

    # Reshape each leaf to (devices, per_device, ...).
    def _split(x: jax.Array) -> jax.Array:
        return x.reshape((device_count, -1) + x.shape[1:])

    return jax.tree_util.tree_map(_split, batch)


def _combine_traj(traj: Trajectory) -> Trajectory:
    """Flatten (devices, games, ...) trajectory axes into a single batch.

    Args:
        traj: Per-device self-play trajectories.

    Returns:
        Trajectory with device and game axes combined.
    """

    # Flatten the first two axes consistently across all fields.
    def _merge(x: jax.Array) -> jax.Array:
        return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])

    return Trajectory(
        obs=_merge(traj.obs),
        policy_targets=_merge(traj.policy_targets),
        player_id=_merge(traj.player_id),
        valid=_merge(traj.valid),
        outcome=_merge(traj.outcome),
    )


def _write_pgn_snapshot(
    paths: RunPaths, run_name: str, game_index: int
) -> None:
    """Write a placeholder PGN snapshot for a given game index.

    Args:
        paths: RunPaths for the current run.
        run_name: Run name for PGN headers.
        game_index: Sequence number for naming.
    """
    # Use minimal headers until real move logging is wired in.
    headers = PgnHeaders(
        event=run_name,
        site="local",
        date=date.today(),
        round=str(game_index),
        white="selfplay",
        black="selfplay",
        result="*",
    )
    pgn = format_pgn(headers, moves=[])
    filename = f"game_{game_index:010d}.pgn"
    write_pgn_file(paths.games_dir / filename, pgn)


def _train(config: dict[str, TomlValue], paths: RunPaths, run_id: str) -> None:
    """Run the training loop with self-play and learning steps.

    Args:
        config: Loaded TOML configuration.
        paths: RunPaths for artifact output.
        run_id: Generated run identifier.
    """
    run_cfg = _parse_run_config(config)
    env_cfg = _parse_env_config(config)
    model_table = _get_table(config, "model")
    mcts_table = _get_table(config, "mcts")
    selfplay_table = _get_table(config, "selfplay")
    train_table = _get_table(config, "train")

    # Resolve the actual number of devices to use.
    device_count = min(run_cfg.devices, jax.local_device_count())
    if device_count < 1:
        raise ValueError("No JAX devices available.")
    devices = jax.devices()[:device_count]

    # Construct environment and model.
    env = make_chess_env()
    model_cfg = TransformerConfig(
        d_model=_get_int(model_table, "d_model"),
        n_heads=_get_int(model_table, "n_heads"),
        mlp_ratio=_get_int(model_table, "mlp_ratio"),
        n_layers=_get_int(model_table, "n_layers"),
    )
    model = ChessTransformer(model_cfg, rngs=nnx.Rngs(run_cfg.seed))
    params = nnx.state(model)

    # Build algorithm configs.
    selfplay_cfg = SelfPlayConfig(
        games_per_device=_get_int(selfplay_table, "games_per_device"),
        max_moves=env_cfg.max_moves,
    )
    mcts_cfg = MctsConfig(
        num_simulations=_get_int(mcts_table, "num_simulations"),
        max_depth=_get_int(mcts_table, "max_depth"),
        c_puct=_get_float(mcts_table, "c_puct"),
        gumbel_scale=_get_float(mcts_table, "gumbel_scale"),
    )
    replay_cfg = ReplayConfig(
        capacity=_get_int(selfplay_table, "replay_capacity"),
        min_to_sample=_get_int(selfplay_table, "min_replay_to_train"),
    )
    loss_cfg = LossConfig(
        value_loss_weight=_get_float(train_table, "value_loss_weight"),
        weight_decay=_get_float(train_table, "weight_decay"),
    )
    optim_cfg = OptimConfig(
        learning_rate=_get_float(train_table, "learning_rate"),
        warmup_steps=_get_int(train_table, "warmup_steps"),
        total_steps=_get_int(train_table, "total_steps"),
        grad_clip_norm=_get_float(train_table, "grad_clip_norm"),
        weight_decay=_get_float(train_table, "weight_decay"),
    )
    train_cfg = TrainConfig(
        batch_size_per_device=_get_int(train_table, "batch_size_per_device"),
    )
    # Initialize training state and optimizer.
    tx, _ = make_optimizer(optim_cfg)
    opt_state = tx.init(params)
    state = TrainState(
        step=Step(0),
        params=params,
        opt_state=opt_state,
        rng_key=jax.random.PRNGKey(run_cfg.seed),
    )

    # Host-side replay buffer and RNG stream.
    replay = ReplayBuffer(replay_cfg)
    rng_stream = RngStream(jax.random.PRNGKey(run_cfg.seed))
    # Checkpoint manager handles periodic persistence.
    manager = make_checkpoint_manager(
        paths.checkpoints,
        CheckpointConfig(
            every_steps=run_cfg.checkpoint_every_steps,
            max_to_keep=3,
        ),
    )
    # Resume from latest checkpoint if available.
    if manager.latest_step() is not None:
        state = restore_latest(manager, state)

    try:
        # Define pmapped functions for self-play and training.
        def _selfplay_fn(rng_key: PRNGKey, params: nnx.State):
            return generate_selfplay_trajectories(
                env=env,
                model=model,
                params=params,
                rng_key=rng_key,
                selfplay_cfg=selfplay_cfg,
                mcts_cfg=mcts_cfg,
            )

        def _train_fn(state_in: TrainState, batch):
            return train_step(
                model=model,
                tx=tx,
                state=state_in,
                batch=batch,
                loss_cfg=loss_cfg,
            )

        p_selfplay = jax.pmap(
            _selfplay_fn, axis_name="data", devices=devices, in_axes=(0, None)
        )
        p_train = jax.pmap(
            _train_fn,
            axis_name="data",
            devices=devices,
            in_axes=(None, 0),
            out_axes=(None, None),
        )

        # Bootstrap metrics/PGN to ensure run dirs have artifacts.
        _write_bootstrap_artifacts(paths, run_cfg.name)
        games_played = 0
        next_pgn_index = 1
        start_time = time.monotonic()
        total_steps = optim_cfg.total_steps

        for step in range(total_steps):
            # Enforce wall-clock timeout.
            elapsed_minutes = (time.monotonic() - start_time) / 60.0
            if elapsed_minutes >= run_cfg.max_runtime_minutes:
                _write_stop_event(paths, run_id, run_cfg.name, "time")
                return

            # Generate self-play trajectories until replay is primed.
            if not replay.can_sample():
                step_key = rng_stream.key_for_step(Step(step))
                device_keys = jnp.stack(
                    [
                        rng_stream.key_for_device(step_key, i)
                        for i in range(device_count)
                    ]
                )
                traj = p_selfplay(device_keys, state.params)
                replay.add(_combine_traj(traj))
                games_played += device_count * selfplay_cfg.games_per_device

            # Emit placeholder PGN snapshots at the configured cadence.
            if games_played >= run_cfg.pgn_every_games * next_pgn_index:
                next_pgn_index += 1
                _write_pgn_snapshot(paths, run_cfg.name, next_pgn_index)

            # Sample a batch and run a training step.
            batch_size = train_cfg.batch_size_per_device * device_count
            sample_key = rng_stream.key_for_step(Step(step + 10_000))
            batch = replay.sample_batch(sample_key, batch_size)
            shard_batch = _split_batch(batch, device_count)
            state, losses = p_train(state, shard_batch)

            # Log scalar metrics for the first replica.
            if (step + 1) % run_cfg.log_every_steps == 0:
                losses_host = jax.tree_util.tree_map(
                    lambda x: float(jax.device_get(x)), losses
                )
                metrics = Metrics(
                    step=step + 1,
                    loss_total=losses_host.total,
                    loss_policy=losses_host.policy,
                    loss_value=losses_host.value,
                    value_mean=0.0,
                    entropy_mean=0.0,
                )
                write_metrics_snapshot(paths, metrics)

            # Persist checkpoints on schedule.
            if (step + 1) % run_cfg.checkpoint_every_steps == 0:
                state_host = jax.device_get(state)
                save_checkpoint(manager, state_host)

        # Normal completion.
        _write_stop_event(paths, run_id, run_cfg.name, "complete")
    finally:
        manager.wait_until_finished()


def _eval(config: dict[str, TomlValue], paths: RunPaths, run_id: str) -> None:
    """Run evaluation and write a summary results TOML.

    Args:
        config: Loaded TOML configuration.
        paths: RunPaths for artifact output.
        run_id: Generated run identifier.
    """
    run_cfg = _parse_run_config(config)
    eval_table = _get_table_optional(config, "eval")
    checkpoints_dir = None
    if eval_table is not None:
        value = eval_table.get("checkpoints_dir")
        if isinstance(value, str):
            checkpoints_dir = Path(value)

    # Placeholder eval logic until arena play is wired in.
    if checkpoints_dir is None:
        result = MatchResult(wins=0, draws=0, losses=0)
        expected = expected_score(1000.0, 1000.0)
        rating_a, rating_b = update_elo(1000.0, 1000.0, expected)
    else:
        # Keep a separate branch to avoid unused variables and allow expansion.
        _ = checkpoints_dir
        result = MatchResult(wins=0, draws=0, losses=0)
        expected = expected_score(1000.0, 1000.0)
        rating_a, rating_b = update_elo(1000.0, 1000.0, expected)

    # Write evaluation summary for downstream tools.
    data: dict[str, TomlValue] = {
        "wins": result.wins,
        "draws": result.draws,
        "losses": result.losses,
        "score": result.score(),
        "expected_score": expected,
        "rating_a": rating_a,
        "rating_b": rating_b,
    }
    save_toml(paths.root / "eval_results.toml", data)
    _write_stop_event(paths, run_id, run_cfg.name, "eval_complete")


def main(argv: list[str] | None = None) -> int:
    """Main entrypoint.

    Returns:
        Process exit code (0 for success).
    """
    # Parse args and load config.
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_toml(args.config)
    run_cfg = _parse_run_config(config)
    run_id = _run_id(run_cfg.name)
    # Create run directories and persist config.
    paths = RunPaths.create(run_id)
    save_toml(paths.config_toml, config)
    _write_start_event(paths, run_id, run_cfg.name)
    if args.command == "train":
        _train(config, paths, run_id)
    else:
        _eval(config, paths, run_id)
    return 0
