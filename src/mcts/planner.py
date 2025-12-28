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
from typing import cast

import jax
import jax.numpy as jnp
import mctx
import pgx
from flax import nnx
from mctx._src import base as mctx_base

from chex_types import Array, PolicyValue, PRNGKey
from env.pgx_chess import mask_illegal_logits


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


def _apply_model(
    graphdef: nnx.GraphDef[nnx.Module], params: nnx.State, obs: Array
) -> PolicyValue:
    """Apply an NNX model graphdef with explicit parameters.

    Args:
        graphdef: Split NNX graph definition.
        params: Parameter/state pytree.
        obs: Observation batch.

    Returns:
        PolicyValue outputs for the batch.
    """
    # Merge graphdef and parameters to get a callable module.
    model_with_params = nnx.merge(graphdef, params)
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

    def _run_mcts_batched(
        batched_state: pgx.State, batched_key: PRNGKey
    ) -> MctsOutput:
        """Run MCTS on a batched state with a single RNG key."""
        # Split the module so params can be passed explicitly.
        graphdef, _ = nnx.split(model)

        def root_fn(root_state: pgx.State) -> mctx.RootFnOutput:
            """Compute root priors/value for MCTS."""
            pv = _apply_model(graphdef, params, root_state.observation)
            logits = mask_illegal_logits(
                pv.policy_logits, root_state.legal_action_mask
            )
            return mctx.RootFnOutput(
                prior_logits=logits,
                value=pv.value,
                embedding=root_state,
            )

        def recurrent_fn(
            params: mctx_base.Params,
            rng: PRNGKey,
            action: Array,
            embed: mctx_base.RecurrentState,
        ) -> tuple[mctx.RecurrentFnOutput, mctx_base.RecurrentState]:
            """Advance environment and compute priors/value for child states."""
            _ = rng
            state = cast(pgx.State, embed)
            params_state = cast(nnx.State, params)
            next_state = jax.vmap(env.step)(state, action)
            pv = _apply_model(graphdef, params_state, next_state.observation)
            logits = mask_illegal_logits(
                pv.policy_logits, next_state.legal_action_mask
            )
            reward = jnp.take_along_axis(
                next_state.rewards,
                next_state.current_player[:, None],
                axis=1,
            ).squeeze(-1)
            discount = jnp.where(next_state.terminated, 0.0, 1.0)
            output = mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=logits,
                value=pv.value,
            )
            return output, cast(mctx_base.RecurrentState, next_state)

        # Seed MCTS at the root and run a Gumbel MuZero policy.
        root = root_fn(batched_state)
        invalid_actions = jnp.logical_not(batched_state.legal_action_mask)
        output = mctx.gumbel_muzero_policy(
            params,
            batched_key,
            root,
            cast(mctx_base.RecurrentFn, recurrent_fn),
            num_simulations=cfg.num_simulations,
            invalid_actions=invalid_actions,
            max_depth=cfg.max_depth,
            gumbel_scale=cfg.gumbel_scale,
            qtransform=mctx.qtransform_by_parent_and_siblings,
        )
        # Normalize action weights after masking illegal actions.
        legal = batched_state.legal_action_mask
        action_weights = jnp.where(legal, output.action_weights, 0.0)
        weight_sum = jnp.sum(action_weights, axis=-1, keepdims=True)
        action_weights = jnp.where(
            weight_sum > 0, action_weights / weight_sum, action_weights
        )
        # Ensure the selected action is legal, falling back if needed.
        action = jnp.asarray(output.action)
        legal_action = jnp.take_along_axis(
            legal, action[:, None], axis=1
        ).squeeze(-1)
        fallback = jnp.argmax(action_weights, axis=-1)
        action = jnp.where(legal_action, action, fallback)
        return MctsOutput(action=action, action_weights=action_weights)

    if rng_key.ndim == 2:
        batch = state.legal_action_mask.shape[0]
        if rng_key.shape[0] != batch:
            raise ValueError("rng_key batch does not match state batch")

        def _run_single(
            single_state: pgx.State, single_key: PRNGKey
        ) -> tuple[Array, Array]:
            """Run MCTS for a single state with a per-game key."""
            batched_state = jax.tree_util.tree_map(
                lambda x: x[None, ...], single_state
            )
            output = _run_mcts_batched(batched_state, single_key)
            return output.action[0], output.action_weights[0]

        action, action_weights = jax.vmap(_run_single)(state, rng_key)
        return MctsOutput(action=action, action_weights=action_weights)

    return _run_mcts_batched(state, rng_key)
