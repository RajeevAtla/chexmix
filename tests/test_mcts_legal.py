from __future__ import annotations

import jax
import jax.numpy as jnp
import pgx
from flax import nnx

from chess_ai.mcts.planner import MctsConfig, run_mcts
from chess_ai.model.chess_transformer import ChessTransformer
from chess_ai.model.nnx_blocks import TransformerConfig


def test_mcts_outputs_legal_actions_and_weights() -> None:
    env = pgx.make("chess")
    cfg = TransformerConfig(d_model=32, n_heads=4, mlp_ratio=2, n_layers=2)
    model = ChessTransformer(cfg, rngs=nnx.Rngs(0))
    params = nnx.state(model)

    batch = 2
    keys = jax.random.split(jax.random.PRNGKey(0), batch)
    state = jax.vmap(env.init)(keys)
    mcts_cfg = MctsConfig(
        num_simulations=2, max_depth=2, c_puct=1.5, gumbel_scale=1.0
    )
    out = run_mcts(
        env=env,
        model=model,
        params=params,
        rng_key=jax.random.PRNGKey(1),
        state=state,
        cfg=mcts_cfg,
    )
    assert out.action.shape == (batch,)
    assert out.action_weights.shape == (batch, 64 * 73)
    legal = state.legal_action_mask
    chosen_legal = jnp.take_along_axis(legal, out.action[:, None], axis=1)
    assert jnp.all(chosen_legal)
    weights_sum = jnp.sum(out.action_weights, axis=-1)
    assert jnp.allclose(weights_sum, 1.0, atol=1e-5)
