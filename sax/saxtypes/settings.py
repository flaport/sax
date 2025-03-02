"""SAX Settings."""

from __future__ import annotations

__all__ = [
    "Settings",
    "SettingsValue",
]

from typing import Annotated, Any, TypeAlias

from .core import ComplexArrayLike, val, val_complex_array


def val_settings(settings: dict) -> Settings:
    if not isinstance(settings, dict):
        msg = "Settings should be a dictionary. Got: {settings!r}."
        raise TypeError(msg)
    return {k: val_settings_value(v) for k, v in settings.items()}


Settings: TypeAlias = Annotated[dict[str, "SettingsValue"], val(val_settings)]
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


def val_settings_value(value: Any) -> SettingsValue:
    """Validate a parameter dictionary"""
    if isinstance(value, str) or value is None:
        return value
    elif isinstance(value, dict):
        return {k: val_settings_value(v) for k, v in value.items()}
    else:
        return val_complex_array(value, strict=False, cast=False)


SettingsValue: TypeAlias = Annotated[
    Settings | ComplexArrayLike | str | None, val(val_settings_value)
]
"""Anything that can be used as value in a settings dict."""
