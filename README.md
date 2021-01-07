# SAX

![Docs](https://readthedocs.org/projects/sax/badge/?version=latest)

Autograd and XLA for S-parameters - a scatter parameter circuit simulator and
optimizer for the frequency domain based on [JAX](https://github.com/google/jax).

The simulator was developed for simulating Photonic Integrated Circuits but in fact is
able to perform any S-parameter based circuit simulation. The goal of SAX is to be a
thin wrapper around JAX with some basic tools for S-parameter based circuit simulation
and optimization. Therefore, SAX does not define any special datastructures and tries to
stay as close as possible to the functional nature of JAX. This makes it very easy to
get started with SAX as you only need functions and standard python dictionaries. Let's
dive in...

## Quick Start

[Full Quick Start page](https://sax.readthedocs.io/en/latest/examples/01_quick_start.html) -
[Examples](https://sax.readthedocs.io/en/latest/examples.html) -
[Full Docs](https://sax.readthedocs.io/en/latest/index.html).

Let's first import the SAX library, along with JAX and the JAX-version of numpy:

```python
import sax
import jax
import jax.numpy as jnp
```

Define a model -- which is just a port combination -> function dictionary -- for your
component. For example a directional coupler:

```python
directional_coupler = {
    ("p0", "p1"): lambda params: (1 - params["coupling"]) ** 0.5,
    ("p1", "p0"): lambda params: (1 - params["coupling"]) ** 0.5,
    ("p2", "p3"): lambda params: (1 - params["coupling"]) ** 0.5,
    ("p3", "p2"): lambda params: (1 - params["coupling"]) ** 0.5,
    ("p0", "p2"): lambda params: 1j * params["coupling"] ** 0.5,
    ("p2", "p0"): lambda params: 1j * params["coupling"] ** 0.5,
    ("p1", "p3"): lambda params: 1j * params["coupling"] ** 0.5,
    ("p3", "p1"): lambda params: 1j * params["coupling"] ** 0.5,
    "default_params": {
        "coupling": 0.5
    },
}
```

Or a waveguide:

```python
def model_waveguide_transmission(params):
    neff = params["neff"]
    dwl = params["wl"] - params["wl0"]
    dneff_dwl = (params["ng"] - params["neff"]) / params["wl0"]
    neff = neff - dwl * dneff_dwl
    phase = jnp.exp(
        jnp.log(2 * jnp.pi * neff * params["length"]) - jnp.log(params["wl"])
    )
    return 10 ** (-params["loss"] * params["length"] / 20) * jnp.exp(1j * phase)

waveguide = {
    ("in", "out"): model_waveguide_transmission,
    ("out", "in"): model_waveguide_transmission,
    "default_params": {
        "length": 25e-6,
        "wl": 1.55e-6,
        "wl0": 1.55e-6,
        "neff": 2.34,
        "ng": 3.4,
        "loss": 0.0,
    },
}
```

These component model dictionaries can be combined into a circuit model dictionary:

```python
mzi = sax.circuit(
    models = {
        "dc1": directional_coupler,
        "top": waveguide,
        "dc2": directional_coupler,
        "btm": waveguide,
    },
    connections={
        "dc1:p2": "top:in",
        "dc1:p1": "btm:in",
        "top:out": "dc2:p3",
        "btm:out": "dc2:p0",
    },
    ports={
        "dc1:p3": "in2",
        "dc1:p0": "in1",
        "dc2:p2": "out2",
        "dc2:p1": "out1",
    },
)
```

Simulating this is as simple as modifying the default parameters:

```python
params = sax.copy_params(mzi["default_params"])
params["top"]["length"] = 2.5e-5
params["btm"]["length"] = 1.5e-5
mzi["in1", "out1"](params)
```

```
DeviceArray(-0.280701+0.10398856j, dtype=complex64)
```

Those are the basics. For more info, check out the **full**
[SAX Quick Start page](https://sax.readthedocs.io/en/latest/examples/01_quick_start.html),
the [Examples](https://sax.readthedocs.io/en/latest/examples.html)
or the
[Documentation](https://sax.readthedocs.io/en/latest/index.html).

## Installation

### Dependencies

- [JAX & JAXLIB](https://github.com/google/jax). Please read the JAX install
  instructions [here](https://github.com/google/jax/#installation). Alternatively, you can
  try running [jaxinstall.sh](jaxinstall.sh) to automatically pip-install the correct
  `jax` and `jaxlib` package for your python and cuda version (if that exact combination
  exists).

### Installation

```
pip install sax
```

## License

Copyright © 2021, Floris Laporte, [Apache-2.0 License](LICENSE)
