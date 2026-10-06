"""Training-only contrastive parameters; never part of the serving encoder."""
from argparse import Namespace
import math

from flax import nnx
import jax
import jax.numpy as jnp

from flaxchat.embedding_stage import contrastive_objective_identity


def configure_objective_state(model, identity):
    """Create a restore template, or reset parameters for an explicit new stage."""
    if identity is None:
        if hasattr(model, 'contrastive_raw_alpha'):
            delattr(model, 'contrastive_raw_alpha')
        return
    expected = contrastive_objective_identity(Namespace(
        contrastive_similarity='yat', yat_infonce_alpha_init=identity.get('alpha_init')))
    if identity != expected:
        raise ValueError('Unsupported contrastive objective identity')
    value = identity['alpha_init'] - identity['alpha_floor']
    raw = value + math.log(-math.expm1(-value))
    model.contrastive_raw_alpha = nnx.Param(jnp.asarray(raw, dtype=jnp.float32))


def objective_loss_arguments(model, identity):
    if identity is None:
        return {}
    return {'similarity': 'yat', 'yat_alpha':
            jax.nn.softplus(model.contrastive_raw_alpha[...]) + identity['alpha_floor']}
