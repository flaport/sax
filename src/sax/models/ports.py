"""Port naming strategies."""

from __future__ import annotations

import re
from typing import Literal, TypeAlias

import sax

PortNamingStrategy: TypeAlias = Literal["optical", "inout"]

PORT_NAMING_STRATEGY: PortNamingStrategy = "inout"


def set_port_naming_strategy(strategy: PortNamingStrategy) -> PortNamingStrategy:
    """Set the port naming strategy for the default sax models."""
    global PORT_NAMING_STRATEGY  # noqa: PLW0603
    _strategy = str(strategy).lower()
    if _strategy not in ["optical", "inout"]:
        msg = f"Invalid port naming strategy: {strategy}"
        raise ValueError(msg)
    PORT_NAMING_STRATEGY = strategy
    return strategy


def get_port_naming_strategy() -> PortNamingStrategy:
    """Get the current port naming strategy."""
    return PORT_NAMING_STRATEGY


class PortNamer:
    """Port naming class to encapsulate port naming logic."""

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        strategy: PortNamingStrategy | None = None,
    ) -> None:
        """Initialize the PortNamer with the number of ports."""
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_ports = num_inputs + num_outputs
        self.strategy = (strategy or get_port_naming_strategy()).lower()

    def __repr__(self) -> str:
        return f"PortNamer({self.num_inputs}, {self.num_outputs}, '{self.strategy}')"

    def __getattr__(self, attr: str) -> sax.Name:
        idx = int(re.sub("[a-zA-Z_-]", "", attr))
        if attr.startswith("in"):
            if idx >= self.num_inputs:
                msg = (
                    f"{self.num_inputs}x{self.num_outputs} PortNamer does not "
                    f"have an input port {attr}."
                )
                raise AttributeError(msg)
                msg = f"Index {idx} out of range for {self.num_ports} ports."
                raise IndexError(msg)
            return self[idx]
        if attr.startswith("out"):
            if idx >= self.num_outputs:
                msg = (
                    f"{self.num_inputs}x{self.num_outputs} PortNamer does not "
                    f"have an output port {attr}."
                )
                raise AttributeError(msg)
            return self[self.num_ports - idx - 1]
        return self[idx - 1]

    def __getitem__(self, idx: int) -> sax.Name:
        if idx < 0:
            msg = "Index must be a non-negative integer."
            raise IndexError(msg)

        if idx >= self.num_ports:
            msg = f"Index {idx} out of range for {self.num_ports} ports."
            raise IndexError(msg)

        if self.strategy == "inout":
            if idx < self.num_inputs:
                return f"in{idx}"
            return f"out{self.num_ports - idx - 1}"

        return f"o{idx + 1:.0f}"

    def is_input_port(self, port: sax.Name) -> bool:
        """Check if a port is an input port."""
        port = port.lower()
        if port.startswith("in"):
            return True
        idx = int(re.sub("[a-zA-Z_-]", "", port))
        return idx < self.num_inputs

    def is_output_port(self, port: sax.Name) -> bool:
        """Check if a port is an output port."""
        return not self.is_input_port(port)

    def is_input_port_idx(self, idx: int) -> bool:
        """Check if a port index is an input port."""
        return self.is_input_port(self[idx])

    def is_output_port_idx(self, idx: int) -> bool:
        """Check if a port index is an output port."""
        return self.is_output_port(self[idx])
