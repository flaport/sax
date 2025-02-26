"""SAX S-Matrix utilities."""

from pydantic import validate_call

import sax


@validate_call
def reciprocal(sdict: sax.SDict) -> sax.SDict:
    """Make an SDict reciprocal."""
    return {
        **{(p1, p2): v for (p1, p2), v in sdict.items()},
        **{(p2, p1): v for (p1, p2), v in sdict.items()},
    }
