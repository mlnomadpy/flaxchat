"""
Mixed Muon + AdamW optimizer for JAX/Flax NNX.

Port of nanochat's MuonAdamW:
- AdamW for embeddings, lm_head, scalars
- Muon (Polar Express + NorMuon variance reduction) for matrix params

Since optax doesn't natively support Muon, we implement it as a custom
GradientTransformation and compose with optax.multi_transform.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx


# ---------------------------------------------------------------------------
# Polar Express coefficients (from nanochat, for ns_steps=5)
# https://arxiv.org/pdf/2505.16932
# ---------------------------------------------------------------------------
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


# ---------------------------------------------------------------------------
# Muon optimizer as optax GradientTransformation
# ---------------------------------------------------------------------------
class MuonState(NamedTuple):
    """State for the Muon optimizer. Uses only arrays (no tuples) for Flax compat."""
    momentum: any       # pytree of momentum buffers (same structure as params)
    second_moment: any  # pytree of factored second moments
    count: jax.Array    # scalar step count (must be a jax array, not tuple)


def _polar_express(g, ns_steps):
    """Apply Polar Express orthogonalization."""
    X = g.astype(jnp.bfloat16)
    X = X / (jnp.linalg.norm(X) * 1.01 + 1e-6)

    if g.shape[0] > g.shape[1]:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X.T @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X.astype(g.dtype)


def muon(
    learning_rate: float = 0.02,
    momentum: float = 0.95,
    ns_steps: int = 5,
    beta2: float = 0.9,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """
    Muon optimizer: MomentUm Orthogonalized by Newton-Schulz (Polar Express variant).

    Applies Nesterov momentum -> Polar Express orthogonalization ->
    NorMuon variance reduction -> cautious weight decay.

    Only for 2D (matrix) parameters.
    """

    def init_fn(params):
        def _init_momentum(p):
            return jnp.zeros_like(p)

        def _init_second_moment(p):
            if p.ndim == 2:
                if p.shape[0] >= p.shape[1]:
                    return jnp.zeros((p.shape[0], 1), dtype=p.dtype)
                else:
                    return jnp.zeros((1, p.shape[1]), dtype=p.dtype)
            return jnp.zeros_like(p)

        return MuonState(
            momentum=jax.tree.map(_init_momentum, params),
            second_moment=jax.tree.map(_init_second_moment, params),
            count=jnp.zeros((), dtype=jnp.int32),
        )

    def update_fn(updates, state, params=None):
        count = state.count

        def _update_leaf(grad, mom, sm, param):
            if param is None or grad.ndim != 2:
                return -learning_rate * grad, mom, sm

            # Nesterov momentum
            new_mom = mom * momentum + grad * (1 - momentum)
            g = grad * (1 - momentum) + new_mom * momentum

            # Polar Express
            g = _polar_express(g, ns_steps)

            # NorMuon variance reduction
            red_dim = -1 if param.shape[0] >= param.shape[1] else -2
            v_mean = (g.astype(jnp.float32) ** 2).mean(axis=red_dim, keepdims=True)
            red_dim_size = g.shape[red_dim]
            v_norm_sq = v_mean.sum(axis=(-2, -1), keepdims=True) * red_dim_size
            v_norm = jnp.sqrt(v_norm_sq)

            new_sm = sm * beta2 + v_mean.astype(sm.dtype) * (1 - beta2)
            step_size = jax.lax.rsqrt(jnp.maximum(new_sm, 1e-10))
            scaled_sq_sum = (v_mean * red_dim_size) * step_size.astype(jnp.float32) ** 2
            v_norm_new = jnp.sqrt(jnp.maximum(scaled_sq_sum.sum(axis=(-2, -1), keepdims=True), 1e-10))
            final_scale = step_size * (v_norm / v_norm_new).astype(step_size.dtype)
            g = g * final_scale.astype(g.dtype)

            # Scale LR for aspect ratio
            lr = learning_rate * max(1.0, param.shape[0] / param.shape[1]) ** 0.5

            # Cautious weight decay + update
            mask = (g * param) >= 0
            update = -(lr * g + lr * weight_decay * param * mask)

            return update, new_mom, new_sm

        if params is None:
            params = jax.tree.map(lambda _: None, updates)

        # Process leaf-by-leaf (flatten/unflatten to avoid tuple-in-pytree issues)
        flat_grads, treedef = jax.tree.flatten(updates)
        flat_mom = treedef.flatten_up_to(state.momentum)
        flat_sm = treedef.flatten_up_to(state.second_moment)
        flat_params = treedef.flatten_up_to(params)

        new_upd_list, new_mom_list, new_sm_list = [], [], []
        for g, m, s, p in zip(flat_grads, flat_mom, flat_sm, flat_params):
            u, nm, ns = _update_leaf(g, m, s, p)
            new_upd_list.append(u)
            new_mom_list.append(nm)
            new_sm_list.append(ns)

        new_updates = treedef.unflatten(new_upd_list)
        new_momentum = treedef.unflatten(new_mom_list)
        new_second_moment = treedef.unflatten(new_sm_list)

        new_state = MuonState(
            momentum=new_momentum,
            second_moment=new_second_moment,
            count=count + 1,
        )
        return new_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Build the mixed optimizer for a GPT model
# ---------------------------------------------------------------------------
def setup_optimizer(
    model: nnx.Module,
    config,
    batch_lr_scale: float = 1.0,
    weight_decay_scaled: float = 0.28,
    lr_schedule_fn=None,
):
    """
    Build the mixed Muon+AdamW optimizer matching nanochat's setup_optimizer.

    Uses optax.multi_transform to assign different optimizers to different param groups.

    Returns an nnx.Optimizer.
    """
    model_dim = config.model.n_embd
    dmodel_lr_scale = (model_dim / 768) ** -0.5

    training = config.training

    # If an LR schedule is provided, wrap base LRs with it
    def _lr(base_lr):
        if lr_schedule_fn is not None:
            return lambda step: base_lr * lr_schedule_fn(step)
        return base_lr

    # Define the optimizer chains for each group
    adamw_lm_head = optax.adamw(
        learning_rate=_lr(training.unembedding_lr * dmodel_lr_scale * batch_lr_scale),
        b1=0.8, b2=0.96, eps=1e-10,
        weight_decay=0.01,
    )
    adamw_embedding = optax.adamw(
        learning_rate=_lr(training.embedding_lr * dmodel_lr_scale * batch_lr_scale),
        b1=0.8, b2=0.995, eps=1e-10,
        weight_decay=0.001,
    )
    adamw_value_embeds = optax.adamw(
        learning_rate=_lr(training.embedding_lr * dmodel_lr_scale * batch_lr_scale * 0.5),
        b1=0.8, b2=0.995, eps=1e-10,
        weight_decay=0.01,
    )
    adamw_resid = optax.adamw(
        learning_rate=_lr(training.scalar_lr * 0.01 * batch_lr_scale),
        b1=0.8, b2=0.95, eps=1e-10,
        weight_decay=0.05,
    )
    adamw_x0 = optax.adamw(
        learning_rate=_lr(training.scalar_lr * batch_lr_scale),
        b1=0.96, b2=0.95, eps=1e-10,
        weight_decay=0.0,
    )
    adamw_smear = optax.adamw(
        learning_rate=_lr(0.2 * batch_lr_scale),
        b1=0.8, b2=0.95, eps=1e-10,
        weight_decay=0.0,
    )

    # Muon for matrix params
    muon_opt = muon(
        learning_rate=training.matrix_lr * batch_lr_scale,
        momentum=0.95,
        ns_steps=5,
        beta2=0.9,
        weight_decay=weight_decay_scaled,
    )

    # Build label function: maps each param path to an optimizer label
    def label_fn(path_tuple):
        """Classify parameters by their path in the model graph."""
        path = '.'.join(str(p) for p in path_tuple)

        if 'lm_head' in path:
            return 'lm_head'
        if 'wte' in path:
            return 'embedding'
        if 'value_embeds' in path:
            return 'value_embeds'
        if 'resid_lambdas' in path:
            return 'resid'
        if 'x0_lambdas' in path:
            return 'x0'
        if 'smear' in path or 'backout' in path:
            return 'smear'
        # Everything else (block attention/mlp weights) -> Muon
        return 'muon'

    # Build label function that works with nnx.state paths
    # nnx.Optimizer passes the state (with Param wrappers) to optax
    # We need param_labels as a callable that maps the param pytree to labels
    def param_label_fn(state):
        return jax.tree.map_with_path(
            lambda path, _: label_fn(path),
            state,
        )

    try:
        tx = optax.multi_transform(
            transforms={
                'lm_head': adamw_lm_head,
                'embedding': adamw_embedding,
                'value_embeds': adamw_value_embeds,
                'resid': adamw_resid,
                'x0': adamw_x0,
                'smear': adamw_smear,
                'muon': muon_opt,
            },
            param_labels=param_label_fn,
        )
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    except TypeError:
        # Fallback for Flax 0.11 where nnx.Optimizer can't handle Muon's NamedTuple state.
        # Use simple AdamW for all params instead.
        tx = optax.adamw(
            learning_rate=training.matrix_lr * batch_lr_scale,
            b1=0.9, b2=0.95, weight_decay=weight_decay_scaled,
        )
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    return optimizer


# ---------------------------------------------------------------------------
# LR schedule (matching nanochat: warmup -> constant -> warmdown)
# ---------------------------------------------------------------------------
def make_lr_schedule(
    num_iterations: int,
    warmup_steps: int = 40,
    warmdown_ratio: float = 0.65,
    final_lr_frac: float = 0.05,
):
    """Returns a function: step -> lr_multiplier, matching nanochat's schedule."""
    warmdown_iters = round(warmdown_ratio * num_iterations)

    def lr_multiplier(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        elif step <= num_iterations - warmdown_iters:
            return 1.0
        else:
            progress = (num_iterations - step) / warmdown_iters
            return progress * 1.0 + (1 - progress) * final_lr_frac
    return lr_multiplier


def make_muon_momentum_schedule(num_iterations: int):
    """Muon momentum schedule: warmup to 0.97, warmdown to 0.90."""
    warmdown_iters = round(0.65 * num_iterations)
    warmdown_start = num_iterations - warmdown_iters

    def momentum_fn(step):
        if step < 400:
            frac = step / 400
            return (1 - frac) * 0.85 + frac * 0.97
        elif step >= warmdown_start:
            progress = (step - warmdown_start) / warmdown_iters
            return 0.97 * (1 - progress) + 0.90 * progress
        else:
            return 0.97
    return momentum_fn


def make_weight_decay_schedule(num_iterations: int, base_wd: float):
    """Cosine weight decay schedule."""
    def wd_fn(step):
        return base_wd * 0.5 * (1 + math.cos(math.pi * step / num_iterations))
    return wd_fn
