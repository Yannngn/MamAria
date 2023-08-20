import jax.numpy as jnp
import numpy as np


def clip_for_log(X, a_min=None, a_max=None):
    """Clip the values in X between a_min and a_max"""
    eps = np.finfo(X.dtype).tiny
    return np.clip(X, eps, 1 - eps)


def clip(X, a_min=None, a_max=None):
    """Clip the values in X between a_min and a_max"""
    eps = np.finfo(X.dtype).tiny
    return np.clip(X, eps, 1 - eps)


def clip_jax(X, a_min=None, a_max=None):
    """Clip the values in X between a_min and a_max"""
    eps = jnp.finfo(X.dtype).eps
    return jnp.clip(X, eps, 1 - eps)
