"""
mctx-based MCTS planner.

We run MCTS over the true PGX environment dynamics (perfect model):
- root_fn evaluates current state via neural net
- recurrent_fn steps PGX state and re-evaluates with neural net

Hard requirements:
- Return legal actions
- Return action_weights distribution
- Fully vectorized over batch
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mctx
import pgx
from flax import nnx

from chess_ai.env.pgx_chess import mask_illegal_logits
from chess_ai.types import Array, PRNGKey, PolicyValue


@dataclass(frozen=True, slots=True)
class MctsConfig:
    """MCTS hyperparameters."""

    num_simulations: int
    max_depth: int
    c_puct: float
    gumbel_scale: float


@dataclass(frozen=True, slots=True)
class MctsOutput:
    """Planner output for a batch."""

    action: Array  # (B,)
    action_weights: Array  # (B, 4672)


def _apply_model(model: nnx.Module, params: nnx.State, obs: Array) -> PolicyValue:
    model_with_params = nnx.merge(model, params)
    return model_with_params(obs)


def run_mcts(
    *,
    env: pgx.Env,
    model: nnx.Module,
    params: nnx.State,
    rng_key: PRNGKey,
    state: pgx.State,
    cfg: MctsConfig,
) -> MctsOutput:
    """Run MCTS and return an action + improved policy distribution.

    Args:
        env: PGX chess environment.
        model: NNX model (callable).
        params: NNX state/params pytree.
        rng_key: PRNG key for MCTS stochasticity.
        state: Batched PGX state.
        cfg: MctsConfig.

    Returns:
        MctsOutput with sampled/selected action and action_weights distribution.
    """

    def root_fn(
        root_state: pgx.State,
    ) -> mctx.RootFnOutput:
        pv = _apply_model(model, params, root_state.observation)
        logits = mask_illegal_logits(pv.policy_logits, root_state.legal_action_mask)
        return mctx.RootFnOutput(
            prior_logits=logits,
            value=pv.value,
            embedding=root_state,
        )

    def recurrent_fn(
        rng: PRNGKey,
        action: Array,
        embed: pgx.State,
    ) -> mctx.RecurrentFnOutput:
        next_state = env.step(embed, action)
        pv = _apply_model(model, params, next_state.observation)
        logits = mask_illegal_logits(pv.policy_logits, next_state.legal_action_mask)
        reward = next_state.rewards[jnp.arange(action.shape[0]), next_state.current_player]
        discount = jnp.where(next_state.terminated, 0.0, 1.0)
        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=pv.value,
            embedding=next_state,
        )

    root = root_fn(state)
    output = mctx.gumbel_muzero_policy(
        rng_key,
        root,
        recurrent_fn,
        num_simulations=cfg.num_simulations,
        max_depth=cfg.max_depth,
        c_puct=cfg.c_puct,
        gumbel_scale=cfg.gumbel_scale,
        qtransform=mctx.qtransform_by_parent_and_siblings,
    )
    legal = state.legal_action_mask
    action_weights = jnp.where(legal, output.action_weights, 0.0)
    weight_sum = jnp.sum(action_weights, axis=-1, keepdims=True)
    action_weights = jnp.where(
        weight_sum > 0, action_weights / weight_sum, action_weights
    )
    action = output.action
    legal_action = jnp.take_along_axis(legal, action[:, None], axis=1).squeeze(-1)
    fallback = jnp.argmax(action_weights, axis=-1)
    action = jnp.where(legal_action, action, fallback)
    return MctsOutput(action=action, action_weights=action_weights)
