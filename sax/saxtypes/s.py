"""SAX Types and type coercions.

Numpy type reference: https://numpy.org/doc/stable/reference/arrays.scalars.html
"""

from __future__ import annotations

__all__ = [
    "GeneralPortMap",
    "InstanceName",
    "InstancePort",
    "Mode",
    "Model",
    "ModelFactory",
    "Port",
    "PortCombination",
    "PortMap",
    "PortMode",
    "PortModeMap",
    "SCoo",
    "SCooModel",
    "SCooModelFactory",
    "SDense",
    "SDenseModel",
    "SDenseModelFactory",
    "SDict",
    "SDictModel",
    "SDictModelFactory",
    "SType",
    "Settings",
    "SettingsValue",
]

from collections.abc import Callable
from typing import (
    Annotated,
    Any,
    TypeAlias,
)

from sax.saxtypes.core import ComplexArrayLike, IntArray1D, val


def _cast_string(obj: Any) -> str:
    if isinstance(obj, bytes):
        obj = obj.decode()
    return str(obj)


def _val_identifier(obj: Any, *, type_name: str) -> str:
    s = _cast_string(obj)
    if not s.isidentifier():
        msg = (
            f"A {type_name!r} string should be a valid python identifier. Got: {s!r}. "
            "note: python identifiers should only contain letters, numbers or "
            "underscores. The first character should not be a number."
        )
        raise TypeError(msg)
    return s


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


def val_instance_name(obj: Any) -> Port:
    return _val_identifier(obj, type_name="InstanceName")


InstanceName: TypeAlias = Annotated[str, val(val_instance_name)]


def val_port(obj: Any) -> Port:
    return _val_identifier(obj, type_name="Port")


Port: TypeAlias = Annotated[str, val(val_port)]
"""A port definition '{port}'."""


def val_mode(obj: Any) -> Mode:
    return _val_identifier(obj, type_name="Mode")


Mode: TypeAlias = Annotated[str, val(val_mode)]
"""A mode definition '{mode}'."""


def val_port_mode(obj: Any) -> PortMode:
    s = _cast_string(obj)
    parts = s.split("@")
    if len(parts) > 2:
        msg = f"a PortMode should have exactly one '@'-separator. Got: {obj!r}"
        raise TypeError(msg)
    port, mode = parts
    port = val_port(port)
    mode = val_mode(mode)
    return f"{port}@{mode}"


PortMode: TypeAlias = Annotated[str, val(val_port_mode)]
"""A port-mode definition '{port}@{mode}'."""


def val_instance_port(obj: Any) -> InstancePort:
    s = _cast_string(obj)
    parts = s.split(",")
    if len(parts) > 2:
        msg = f"an InstancePort should have exactly one ','-separator. Got: {obj!r}"
        raise TypeError(msg)
    inst, port = parts
    inst = val_instance_name(inst)
    port = val_port(port)
    return f"{inst},{port}"


InstancePort: TypeAlias = Annotated[str, val(val_instance_port)]
"""An instance port definition '{inst},{port}'."""


PortMap: TypeAlias = dict[Port, int]
"""A mapping from a port to an index."""


PortModeMap: TypeAlias = dict[Port, int]
"""A mapping from a port-mode to an index."""


GeneralPortMap: TypeAlias = PortMap | PortModeMap
"""Either a regular port map or a port-mode map."""


PortCombination: TypeAlias = tuple[InstancePort, InstancePort]
"""A combination of two port names."""


SDict: TypeAlias = dict[PortCombination, ComplexArrayLike]
"""A sparse dictionary-based S-matrix representation.

A mapping from a port combination to an S-parameter or an array of S-parameters.

Example:

.. code-block::

    sdict: sax.SDict = {
        ("in0", "out0"): 3.0,
    }

"""

SDense: TypeAlias = tuple[ComplexArrayLike, PortMap]
"""A dense S-matrix representation.

S-matrix (2D array) or multidimensional batched S-matrix (N+2)-D array with a port map.
If (N+2)-D array then the S-matrix dimensions are the last two.

Example:

.. code-block::

    Sd = jnp.arange(9, dtype=float).reshape(3, 3)
    port_map = {"in0": 0, "in1": 2, "out0": 1}
    sdense = Sd, port_map

"""

SCoo: TypeAlias = tuple[IntArray1D, IntArray1D, ComplexArrayLike, PortMap]
"""A sparse S-matrix in COO format (recommended for internal library use only).

An `SCoo` is a sparse matrix based representation of an S-matrix consisting of three
arrays and a port map. The three arrays represent the input port indices [`int`],
output port indices [`int`] and the S-matrix values [`ComplexFloat`] of the sparse
matrix. The port map maps a port name [`str`] to a port index [`int`].

Only these four arrays **together** and in this specific **order** are considered a
valid `SCoo` representation!

Example:

.. code-block::

    Si = jnp.arange(3, dtype=int)
    Sj = jnp.array([0, 1, 0], dtype=int)
    Sx = jnp.array([3.0, 4.0, 1.0])
    port_map = {"in0": 0, "in1": 2, "out0": 1}
    scoo: sax.SCoo = (Si, Sj, Sx, port_map)

Note:
    This representation is only recommended for internal library use. Please don't
    write user-facing code using this representation.

"""

SType: TypeAlias = SDict | SCoo | SDense
"""Any S-Matrix type [SDict, SDense, SCOO]."""


SDictModel: TypeAlias = Callable[..., SDict]
"""A keyword-only function producing an SDict."""

SDenseModel: TypeAlias = Callable[..., SDense]
"""A keyword-only function producing an SDense."""


SCooModel: TypeAlias = Callable[..., SCoo]
"""A keyword-only function producing an Scoo."""


Model: TypeAlias = SDictModel | SDenseModel | SCooModel
"""A keyword-only function producing an SType."""

SDictModelFactory: TypeAlias = Callable[..., SDictModel]
"""A keyword-only function producing an SDictModel."""


SDenseModelFactory: TypeAlias = Callable[..., SDenseModel]
"""A keyword-only function producing an SDenseModel."""

SCooModelFactory: TypeAlias = Callable[..., SCooModel]
"""A keyword-only function producing an ScooModel."""


ModelFactory: TypeAlias = SDictModelFactory | SDenseModelFactory | SCooModelFactory
"""A keyword-only function producing a Model."""

if __name__ == "__main__":
    import sax

    x = sax.into[SDict]({"a": 3})
