"""SAX S-Matrix utilities."""

import jax
import jax.scipy as jsp
from jaxtyping import Array
from pydantic import validate_call

import sax


@validate_call
def reciprocal(sdict: sax.SDict) -> sax.SDict:
    """Make an SDict reciprocal."""
    return {
        **{(p1, p2): v for (p1, p2), v in sdict.items()},
        **{(p2, p1): v for (p1, p2), v in sdict.items()},
    }


def block_diag(*arrs: Array) -> Array:
    """Create block diagonal matrix with arbitrary batch dimensions."""
    batch_shape = arrs[0].shape[:-2]

    N = 0
    for arr in arrs:
        if batch_shape != arr.shape[:-2]:
            msg = "Batch dimensions for given arrays don't match."
            raise ValueError(msg)
        m, n = arr.shape[-2:]
        if m != n:
            msg = "given arrays are not square."
            raise ValueError(msg)
        N += n

    arrs = tuple(arr.reshape(-1, arr.shape[-2], arr.shape[-1]) for arr in arrs)
    batch_block_diag = jax.vmap(jsp.linalg.block_diag, in_axes=0, out_axes=0)
    block_diag = batch_block_diag(*arrs)
    return block_diag.reshape(*batch_shape, N, N)
