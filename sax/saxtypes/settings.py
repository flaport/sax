"""SAX Settings."""

from __future__ import annotations

__all__ = [
    "Settings",
    "SettingsValue",
]

from typing import (
    TypeAlias,
)

from .core import ComplexArrayLike

Settings: TypeAlias = dict[str, "SettingsValue"]
"""A (possibly nested) settings mapping.

Example:

.. code-block::

    mzi_settings: sax.Settings = {
        "wl": 1.5,  # global settings
        "lft": {"coupling": 0.5},  # settings for the left coupler
        "top": {"neff": 3.4},  # settings for the top waveguide
        "rgt": {"coupling": 0.3},  # settings for the right coupler
    }

"""

SettingsValue: TypeAlias = Settings | ComplexArrayLike | str | None
"""Anything that can be used as value in a settings dict."""
