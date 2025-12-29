"""CLI entrypoint and helper tests."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pgx
import pytest
from flax import nnx

from cli import (
    _append_event,
    _combine_traj,
    _eval,
    _get_float,
    _get_int,
    _get_str,
    _get_table,
    _get_table_optional,
    _git_sha,
    _parse_env_config,
    _parse_run_config,
    _run_id,
    _split_batch,
    _train,
    _write_bootstrap_artifacts,
    _write_pgn_snapshot,
    _write_start_event,
    _write_stop_event,
    build_parser,
    main,
)
from mcts.planner import MctsConfig
from paths import RunPaths
from selfplay.rollout import SelfPlayConfig
from selfplay.trajectory import Trajectory
from toml_io import TomlValue, load_toml, save_toml
from train.losses import LossConfig, Losses
from train.state import TrainState


class _DummyCheckpointManager:
    """Minimal checkpoint manager stub for CLI tests."""

    def latest_step(self) -> int | None:
        """Return None to indicate no checkpoint is present."""
        return None

    def wait_until_finished(self) -> None:
        """No-op wait for async checkpointing."""
        return None


class _ResumeCheckpointManager:
    """Checkpoint manager stub that signals a resume is available."""

    def latest_step(self) -> int | None:
        """Return a fake latest step to trigger restore."""
        return 0

    def wait_until_finished(self) -> None:
        """No-op wait for async checkpointing."""
        return None


def _minimal_train_config() -> dict[str, TomlValue]:
    """Return a minimal config dict for fast training tests.

    Returns:
        Minimal training configuration as TOML-compatible values.
    """
    # Keep values tiny to reduce runtime.
    return {
        "run": {
            "name": "demo",
            "seed": 42,
            "devices": 1,
            "log_every_steps": 1,
            "checkpoint_every_steps": 1,
            "pgn_every_games": 1,
            "max_runtime_minutes": 1,
        },
        "env": {"max_moves": 4},
        "model": {"d_model": 8, "n_heads": 2, "mlp_ratio": 1, "n_layers": 1},
        "mcts": {
            "num_simulations": 1,
            "max_depth": 1,
            "c_puct": 1.0,
            "gumbel_scale": 1.0,
        },
        "selfplay": {
            "games_per_device": 1,
            "replay_capacity": 8,
            "min_replay_to_train": 2,
        },
        "train": {
            "batch_size_per_device": 1,
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "total_steps": 1,
            "value_loss_weight": 1.0,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
        },
    }


def _fake_selfplay(
    *,
    env: pgx.Env,
    model: nnx.Module,
    params: nnx.State,
    rng_key: jax.Array,
    selfplay_cfg: SelfPlayConfig,
    mcts_cfg: MctsConfig,
) -> Trajectory:
    """Generate a deterministic dummy trajectory for training tests.

    Args:
        env: Unused environment placeholder.
        model: Unused model placeholder.
        params: Unused params placeholder.
        rng_key: Unused RNG placeholder.
        selfplay_cfg: Self-play config used for shapes.
        mcts_cfg: Unused MCTS config placeholder.

    Returns:
        Trajectory filled with zeros and uniform policy targets.
    """
    # Build predictable arrays for easy shape checks.
    del env, model, params, rng_key, mcts_cfg
    cfg = selfplay_cfg
    batch = cfg.games_per_device
    max_moves = cfg.max_moves
    obs = jnp.zeros((batch, max_moves, 8, 8, 119), dtype=jnp.float32)
    policy = jnp.full((batch, max_moves, 4672), 1.0 / 4672.0, dtype=jnp.float32)
    player_id = jnp.zeros((batch, max_moves), dtype=jnp.int32)
    valid = jnp.ones((batch, max_moves), dtype=jnp.bool_)
    outcome = jnp.zeros((batch, max_moves), dtype=jnp.float32)
    return Trajectory(
        obs=obs,
        policy_targets=policy,
        player_id=player_id,
        valid=valid,
        outcome=outcome,
    )


def _fake_train_step(
    *,
    model: nnx.Module,
    tx: optax.GradientTransformation,
    state: TrainState,
    batch: dict[str, jax.Array],
    loss_cfg: LossConfig,
) -> tuple[TrainState, Losses]:
    """Return a no-op TrainState and zero losses.

    Args:
        model: Unused model placeholder.
        tx: Unused optimizer placeholder.
        state: TrainState to return unchanged.
        batch: Unused batch placeholder.
        loss_cfg: Unused loss config placeholder.

    Returns:
        Same TrainState and zero Losses.
    """
    # Keep loss values deterministic.
    del model, tx, batch, loss_cfg
    zero = jnp.array(0.0, dtype=jnp.float32)
    return state, Losses(total=zero, policy=zero, value=zero, l2=zero)


def test_build_parser_train_args() -> None:
    """Parser recognizes train command and config path."""
    # Parse a sample command.
    parser = build_parser()
    args = parser.parse_args(["train", "--config", "config/default.toml"])
    assert args.command == "train"
    assert isinstance(args.config, Path)


def test_main_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main returns zero for eval command."""
    # Run eval in a temp working directory.
    monkeypatch.chdir(tmp_path)
    config_path = (
        Path(__file__).resolve().parents[1] / "config" / "default.toml"
    )
    assert main(["eval", "--config", str(config_path)]) == 0


def test_main_train_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train command writes expected run artifacts."""
    monkeypatch.chdir(tmp_path)
    # Stub out heavy training dependencies.
    monkeypatch.setattr("cli.generate_selfplay_trajectories", _fake_selfplay)
    monkeypatch.setattr("cli.train_step", _fake_train_step)
    monkeypatch.setattr(
        "cli.make_checkpoint_manager", lambda *_: _DummyCheckpointManager()
    )
    monkeypatch.setattr("cli.save_checkpoint", lambda *_: None)
    config_path = tmp_path / "config.toml"
    save_toml(config_path, _minimal_train_config())

    # CLI should complete successfully.
    assert main(["train", "--config", str(config_path)]) == 0

    runs_dir = tmp_path / "runs"
    run_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_root = run_dirs[0]
    assert (run_root / "config.toml").exists()
    assert (run_root / "events.toml").exists()
    assert (run_root / "metrics").exists()
    assert (run_root / "games" / "game_0000000001.pgn").exists()


def test_main_missing_run_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing run table raises ValueError."""
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.toml"
    save_toml(config_path, {"env": {"id": "chess"}})
    with pytest.raises(ValueError, match="Missing TOML table"):
        _ = main(["eval", "--config", str(config_path)])


def test_main_missing_run_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing run name raises ValueError."""
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.toml"
    save_toml(config_path, {"run": {"seed": 42}})
    with pytest.raises(ValueError, match="Missing string key"):
        _ = main(["eval", "--config", str(config_path)])


def test_get_helpers_and_parsers() -> None:
    """Helper getters enforce types and missing key handling."""
    table: dict[str, TomlValue] = {"count": 3, "name": "demo"}
    assert _get_int(table, "count") == 3
    assert _get_str(table, "name") == "demo"
    assert _get_float({"pi": 3.0}, "pi") == 3.0
    assert _get_float({"pi": 3}, "pi") == 3.0

    with pytest.raises(ValueError, match="Missing int key"):
        _ = _get_int(table, "missing")
    with pytest.raises(ValueError, match="Missing string key"):
        _ = _get_str(table, "missing")
    with pytest.raises(ValueError, match="Missing float key"):
        _ = _get_float(table, "missing")
    with pytest.raises(ValueError, match="Missing TOML table"):
        _ = _get_table({"run": {"name": "x", "seed": 1}}, "missing")
    assert (
        _get_table_optional({"run": {"name": "x", "seed": 1}}, "missing")
        is None
    )
    with pytest.raises(ValueError, match="Missing TOML table"):
        _ = _get_table_optional({"bad": 1}, "bad")


def test_append_event_and_git_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Event appending and git SHA resolution behave correctly."""
    monkeypatch.chdir(tmp_path)
    paths = RunPaths.create("run")
    # Seed events to validate index incrementing.
    save_toml(
        paths.events_toml, {"event_bad": {"event": "x"}, "event_0002": {}}
    )
    _append_event(paths, {"event": "start"})
    data = load_toml(paths.events_toml)
    assert "event_0003" in data

    # No .git directory yields unknown.
    assert _git_sha() == "unknown"
    git_dir = tmp_path / ".git" / "refs" / "heads"
    git_dir.mkdir(parents=True)
    (tmp_path / ".git" / "HEAD").write_text(
        "ref: refs/heads/main", encoding="utf-8"
    )
    # Missing ref target yields unknown.
    assert _git_sha() == "unknown"
    (git_dir / "main").write_text("deadbeef", encoding="utf-8")
    assert _git_sha() == "deadbeef"
    (tmp_path / ".git" / "HEAD").write_text("cafebabe", encoding="utf-8")
    assert _git_sha() == "cafebabe"


def test_events_and_pgn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Start/stop events and PGN snapshots are written."""
    monkeypatch.chdir(tmp_path)
    paths = RunPaths.create("run")
    run_id = _run_id("demo")
    _write_start_event(paths, run_id, "demo")
    _write_stop_event(paths, run_id, "demo", "done")
    _write_bootstrap_artifacts(paths, "demo")
    _write_pgn_snapshot(paths, "demo", 2)
    assert paths.events_toml.exists()
    assert (paths.games_dir / "game_0000000001.pgn").exists()
    assert (paths.games_dir / "game_0000000002.pgn").exists()


def test_split_batch_and_combine_traj() -> None:
    """Batch sharding and trajectory merging preserve shapes."""
    batch = {
        "obs": jnp.zeros((4, 8, 8, 119), dtype=jnp.float32),
        "policy_targets": jnp.zeros((4, 4672), dtype=jnp.float32),
    }
    split = _split_batch(batch, device_count=2)
    assert split["obs"].shape[0] == 2
    traj = Trajectory(
        obs=jnp.zeros((2, 3, 8, 8, 119), dtype=jnp.float32),
        policy_targets=jnp.zeros((2, 3, 4672), dtype=jnp.float32),
        player_id=jnp.zeros((2, 3), dtype=jnp.int32),
        valid=jnp.ones((2, 3), dtype=jnp.bool_),
        outcome=jnp.zeros((2, 3), dtype=jnp.float32),
    )
    merged = _combine_traj(traj)
    assert merged.obs.shape[0] == 6


def test_train_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Training loop runs end-to-end with stubs."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("cli.generate_selfplay_trajectories", _fake_selfplay)
    monkeypatch.setattr("cli.train_step", _fake_train_step)
    monkeypatch.setattr(
        "cli.make_checkpoint_manager", lambda *_: _DummyCheckpointManager()
    )
    monkeypatch.setattr("cli.save_checkpoint", lambda *_: None)
    config = _minimal_train_config()
    paths = RunPaths.create("run")
    _train(config, paths, "run")
    assert paths.events_toml.exists()
    assert paths.metrics_dir.exists()
    assert paths.games_dir.exists()


def test_train_resume_uses_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training resumes from checkpoint when manager reports latest step."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("cli.generate_selfplay_trajectories", _fake_selfplay)
    monkeypatch.setattr("cli.train_step", _fake_train_step)
    monkeypatch.setattr(
        "cli.make_checkpoint_manager", lambda *_: _ResumeCheckpointManager()
    )
    monkeypatch.setattr("cli.save_checkpoint", lambda *_: None)

    called = {"restore": False}

    def _restore_latest(manager, state):
        del manager
        called["restore"] = True
        return state

    monkeypatch.setattr("cli.restore_latest", _restore_latest)
    config = _minimal_train_config()
    paths = RunPaths.create("run")
    _train(config, paths, "run")
    assert called["restore"]


def test_train_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Training stops early when runtime limit is hit."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("cli.generate_selfplay_trajectories", _fake_selfplay)
    monkeypatch.setattr("cli.train_step", _fake_train_step)
    monkeypatch.setattr(
        "cli.make_checkpoint_manager", lambda *_: _DummyCheckpointManager()
    )
    monkeypatch.setattr("cli.save_checkpoint", lambda *_: None)
    config = _minimal_train_config()
    run_table = _get_table(config, "run")
    run_table["max_runtime_minutes"] = 0
    paths = RunPaths.create("run")
    _train(config, paths, "run")
    data = load_toml(paths.events_toml)
    event = data["event_0001"]
    assert isinstance(event, dict)
    assert event.get("reason") == "time"


def test_train_no_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training raises when no devices are available."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("cli.jax.local_device_count", lambda: 0)
    with pytest.raises(ValueError, match="No JAX devices available"):
        _train(_minimal_train_config(), RunPaths.create("run"), "run")


def test_eval_writes_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eval writes results TOML."""
    monkeypatch.chdir(tmp_path)
    config = _minimal_train_config()
    config["run"] = {
        "name": "eval",
        "seed": 1,
        "devices": 1,
        "log_every_steps": 1,
        "checkpoint_every_steps": 1,
        "pgn_every_games": 1,
        "max_runtime_minutes": 1,
    }
    paths = RunPaths.create("eval_run")
    _eval(config, paths, "eval_run")
    assert (paths.root / "eval_results.toml").exists()


def test_eval_with_checkpoints_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eval accepts explicit checkpoints_dir configuration."""
    monkeypatch.chdir(tmp_path)
    config = _minimal_train_config()
    config["eval"] = {"checkpoints_dir": "runs/demo/checkpoints"}
    config["run"] = {
        "name": "eval",
        "seed": 1,
        "devices": 1,
        "log_every_steps": 1,
        "checkpoint_every_steps": 1,
        "pgn_every_games": 1,
        "max_runtime_minutes": 1,
    }
    paths = RunPaths.create("eval_run")
    _eval(config, paths, "eval_run")
    assert (paths.root / "eval_results.toml").exists()


def test_parse_env_and_run_config() -> None:
    """Parsed run/env configs return expected values."""
    config = {
        "run": {
            "name": "demo",
            "seed": 1,
            "devices": 1,
            "log_every_steps": 1,
            "checkpoint_every_steps": 1,
            "pgn_every_games": 1,
            "max_runtime_minutes": 1,
        },
        "env": {"max_moves": 2},
    }
    run_cfg = _parse_run_config(config)
    env_cfg = _parse_env_config(config)
    assert run_cfg.name == "demo"
    assert env_cfg.max_moves == 2
