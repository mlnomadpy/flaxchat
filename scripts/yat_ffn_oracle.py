"""Independent NumPy FP64 YAT FFN value/VJP; never used in model training.

Inputs should be the actual rounded operands when auditing low-precision runs.
The oracle describes the real-valued kernel, not intermediate BF16 rounding.
"""
import numpy as np


def yat_ffn_value_and_grad(x, kernel, alpha, cotangent):
    x, kernel, cotangent = (np.asarray(a, dtype=np.float64)
                            for a in (x, kernel, cotangent))
    alpha_array = np.asarray(alpha, dtype=np.float64)
    if (x.ndim != 2 or kernel.ndim != 2 or kernel.shape[0] != x.shape[1]
            or kernel.shape[1] % 2 or alpha_array.shape != ()):
        raise ValueError('Expected 2D inputs/kernel, even kernel output width and scalar alpha')
    width = kernel.shape[1] // 2
    if cotangent.shape != (x.shape[0], width):
        raise ValueError('Cotangent must match FFN output shape')
    if not all(np.isfinite(a).all() for a in (x, kernel, cotangent, alpha_array)):
        raise ValueError('Oracle inputs must be finite')
    alpha = float(alpha_array)
    prototypes, gates = np.split(kernel, 2, axis=1)
    dot, gate = x @ prototypes, x @ gates
    distance = np.zeros(dot.shape, np.float64)
    for start in range(0, x.shape[1], 16):
        delta = x[:, None, start:start+16] - prototypes[start:start+16].T[None, :, :]
        distance += np.sum(delta * delta, axis=-1)
    denominator = distance + .01
    similarity = (dot + 1)**2 / denominator
    output = alpha * similarity * gate
    dot_grad = cotangent * gate * (2 * alpha * (dot + 1) / denominator)
    distance_grad = -cotangent * gate * alpha * (dot + 1)**2 / denominator**2
    gate_grad = cotangent * alpha * similarity
    # Accumulate distance derivatives from differences too, so the oracle does
    # not introduce cancellation by subtracting weighted matrix products.
    dx_distance = np.zeros_like(x)
    dp_distance = np.zeros_like(prototypes)
    for start in range(0, x.shape[1], 16):
        delta = x[:, None, start:start+16] - prototypes[start:start+16].T[None, :, :]
        weighted = 2 * distance_grad[..., None] * delta
        dx_distance[:, start:start+16] = weighted.sum(axis=1)
        dp_distance[start:start+16] = -weighted.sum(axis=0).T
    return dict(loss=np.asarray(np.sum(output * cotangent)), output=output,
                dx=dot_grad @ prototypes.T + dx_distance + gate_grad @ gates.T,
                dk=np.concatenate([x.T @ dot_grad + dp_distance, x.T @ gate_grad], axis=1),
                dalpha=np.asarray(np.sum(cotangent * similarity * gate)))
