# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_caching.ipynb (unless otherwise specified).


from __future__ import annotations


__all__ = ['cache', 'cache_clear']

# Cell
#nbdev_comment from __future__ import annotations

import gc
from functools import _lru_cache_wrapper, lru_cache, partial, wraps
from typing import Callable, Optional

# Internal Cell

_cached_functions = []

# Cell

def cache(func: Optional[Callable] = None, /, *, maxsize: Optional[int] = None) -> Callable:
    """cache a function"""
    if func is None:
        return partial(cache, maxsize=maxsize)

    cached_func = lru_cache(maxsize=maxsize)(func)

    @wraps(func)
    def new_func(*args, **kwargs):
        return cached_func(*args, **kwargs)

    new_func.cache_clear = cached_func.cache_clear

    _cached_functions.append(new_func)

    return new_func

# Cell
def cache_clear(*, force: bool=False):
    """clear all function caches"""
    if not force:
        for func in _cached_functions:
            func.cache_clear()
    else:
        gc.collect()
        funcs = [a for a in gc.get_objects() if isinstance(a, _lru_cache_wrapper)]

        for func in funcs:
            func.cache_clear()