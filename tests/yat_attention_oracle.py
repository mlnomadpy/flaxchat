"""Independent NumPy/FP64 YAT attention and analytic gradient test oracle.

Only small test fixtures: direct pairwise differences deliberately use O(L²D)
space. This is not a training backend or a proposal for production precision.
"""
import numpy as np


def attention_and_gradients(q, k, v, segments, alpha, cotangent, *, radius=None):
    q, k, v, cotangent = [
        np.asarray(x, dtype=np.float64).transpose(0, 2, 1, 3)
        for x in (q, k, v, cotangent)
    ]
    segments = np.asarray(segments)
    valid = segments >= 0
    mask = (
        (segments[:, :, None] == segments[:, None, :])
        & valid[:, :, None] & valid[:, None, :]
    )
    if radius is not None:
        positions = np.arange(q.shape[2])
        mask &= abs(positions[:, None] - positions[None, :]) <= radius
    # A dummy key makes invalid query rows finite; their output/cotangent is zero.
    mask |= (~valid)[:, :, None] & (np.arange(q.shape[2]) == 0)[None, None, :]
    differences = q[:, :, :, None, :] - k[:, :, None, :, :]
    dot = q @ k.swapaxes(-1, -2)
    denominator = np.sum(differences * differences, axis=-1) + .01
    kernel = (dot + 1) ** 2 / denominator
    scores = np.where(mask[:, None], alpha * kernel, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    output = weights @ v
    output = np.where(valid[:, None, :, None], output, 0)
    g = np.where(valid[:, None, :, None], cotangent, 0)
    gp = g @ v.swapaxes(-1, -2)
    gs = weights * (gp - np.sum(weights * gp, axis=-1, keepdims=True))
    gz = gs * alpha * 2 * (dot + 1) / denominator
    gd = -gs * alpha * kernel / denominator
    coordinate_gradient = 2 * gd[..., None] * differences
    gq = gz @ k + coordinate_gradient.sum(axis=-2)
    gk = gz.swapaxes(-1, -2) @ q - coordinate_gradient.sum(axis=-3)
    gv = weights.swapaxes(-1, -2) @ g
    alpha_terms = gs * kernel
    gradients = tuple(x.transpose(0, 2, 1, 3) for x in (gq, gk, gv))
    return (
        output.transpose(0, 2, 1, 3),
        (*gradients, alpha_terms.sum()),
        np.abs(alpha_terms).sum(),
    )


def attention_and_gradients_tiled(q, k, v, segments, alpha, cotangent, *,
                                  radius=None, query_tile=32):
    """FP64 oracle with bounded coordinate scratch for saved device failures.

    Query tiling preserves a full eligible-key softmax for every row. Process
    heads separately; coordinate scratch is O(query_tile * length * width),
    rather than O(batch * heads * length**2 * width). This is test code only.
    """
    if query_tile < 1 or (radius is not None and radius < 0):
        raise ValueError("Positive query tile and nonnegative radius required")
    q, k, v, cotangent = [np.asarray(x, dtype=np.float64)
                         for x in (q, k, v, cotangent)]
    segments = np.asarray(segments)
    if (q.ndim != 4 or not q.shape[1] or q.shape != k.shape
            or q.shape != v.shape or q.shape != cotangent.shape
            or segments.shape != q.shape[:2]):
        raise ValueError("Matching nonempty Q/K/V/cotangent and segment shapes required")
    batch, length, heads, _ = q.shape
    output, gq, gk, gv = [np.zeros_like(q) for _ in range(4)]
    ga = absolute_alpha = 0.
    positions = np.arange(length)
    for b in range(batch):
        valid = segments[b] >= 0
        for h in range(heads):
            keys, values = k[b, :, h], v[b, :, h]
            for start in range(0, length, query_tile):
                stop = min(start + query_tile, length)
                queries = q[b, start:stop, h]
                query_valid = valid[start:stop]
                mask = ((segments[b, start:stop, None] == segments[b, None, :])
                        & query_valid[:, None] & valid[None, :])
                if radius is not None:
                    mask &= abs(positions[start:stop, None] - positions[None, :]) <= radius
                mask |= (~query_valid)[:, None] & (positions == 0)[None, :]
                delta = queries[:, None, :] - keys[None, :, :]
                dot = queries @ keys.T
                denominator = np.sum(delta * delta, axis=-1) + .01
                kernel = (dot + 1) ** 2 / denominator
                scores = np.where(mask, alpha * kernel, -np.inf)
                p = np.exp(scores - scores.max(-1, keepdims=True))
                p /= p.sum(-1, keepdims=True)
                output[b, start:stop, h] = np.where(query_valid[:, None], p @ values, 0)
                g = np.where(query_valid[:, None], cotangent[b, start:stop, h], 0)
                gp = g @ values.T
                gs = p * (gp - (p * gp).sum(-1, keepdims=True))
                gz = gs * alpha * 2 * (dot + 1) / denominator
                gd = -gs * alpha * kernel / denominator
                coordinate = 2 * gd[..., None] * delta
                gq[b, start:stop, h] = gz @ keys + coordinate.sum(axis=1)
                gk[b, :, h] += gz.T @ queries - coordinate.sum(axis=0)
                gv[b, :, h] += p.T @ g
                terms = gs * kernel
                ga += terms.sum()
                absolute_alpha += np.abs(terms).sum()
    return output, (gq, gk, gv, ga), absolute_alpha
