# SAX 0.3.1

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

Define a model function for your component. A SAX model is just a function that returns
an 'S-dictionary'. For example a directional coupler:

```python
def coupler(coupling=0.5):
    kappa = coupling**0.5
    tau = (1-coupling)**0.5
    sdict = sax.reciprocal({
        ("in0", "out0"): tau,
        ("in0", "out1"): 1j*kappa,
        ("in1", "out0"): 1j*kappa,
        ("in1", "out1"): tau,
    })
    return sdict
```

Or a waveguide:

```python
def waveguide(wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss=0.0):
    dwl = wl - wl0
    dneff_dwl = (ng - neff) / wl0
    neff = neff - dwl * dneff_dwl
    phase = 2 * jnp.pi * neff * length / wl
    transmission = 10 ** (-loss * length / 20) * jnp.exp(1j * phase)
    sdict = reciprocal({("in0", "out0"): transmission})
    return sdict
```

These component models can then be combined into a circuit:

```python
mzi = sax.circuit(
    instances = {
        "lft": coupler,
        "top": waveguide,
        "rgt": coupler,
    },
    connections={
        "lft:out0": "rgt:in0",
        "lft:out1": "top:in0",
        "top:out0": "rgt:in1",
    },
    ports={
        "lft:in0": "in0",
        "lft:in1": "in1",
        "rgt:out0": "out0",
        "rgt:out1": "out1",
    },
)
```

This mzi circuit is a model function in its own right. To simulate it, first obtain the
(possibly nested) dictionary of parameters, then modify the parameters and call the
function:

```python
params = sax.get_params(mzi)
params["top"]["length"] = 10e-5
S = mzi(**params)
S["in0", "out0"]
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
