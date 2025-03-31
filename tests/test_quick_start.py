import jax
import jax.example_libraries.optimizers as opt
import jax.numpy as jnp
import sax


def test_quick_start():
    """Runs the core parts of the quick start notebook.

    This does not use jax.jit, to support the cupy backend.
    """
    coupling = 0.5
    kappa = coupling**0.5
    tau = (1 - coupling) ** 0.5
    coupler_dict = {
        ("in0", "out0"): tau,
        ("out0", "in0"): tau,
        ("in0", "out1"): 1j * kappa,
        ("out1", "in0"): 1j * kappa,
        ("in1", "out0"): 1j * kappa,
        ("out0", "in1"): 1j * kappa,
        ("in1", "out1"): tau,
        ("out1", "in1"): tau,
    }

    coupler_dict = sax.reciprocal(
        {
            ("in0", "out0"): tau,
            ("in0", "out1"): 1j * kappa,
            ("in1", "out0"): 1j * kappa,
            ("in1", "out1"): tau,
        }
    )

    def coupler(coupling=0.5) -> sax.SDict:
        kappa = coupling**0.5
        tau = (1 - coupling) ** 0.5
        coupler_dict = sax.reciprocal(
            {
                ("in0", "out0"): tau,
                ("in0", "out1"): 1j * kappa,
                ("in1", "out0"): 1j * kappa,
                ("in1", "out1"): tau,
            }
        )
        return coupler_dict

    coupler(coupling=0.3)

    def waveguide(
        wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss=0.0
    ) -> sax.SDict:
        dwl = wl - wl0
        dneff_dwl = (ng - neff) / wl0
        neff = neff - dwl * dneff_dwl
        phase = 2 * jnp.pi * neff * length / wl
        transmission = 10 ** (-loss * length / 20) * jnp.exp(1j * phase)
        sdict = sax.reciprocal(
            {
                ("in0", "out0"): transmission,
            }
        )
        return sdict

    mzi, info = sax.circuit(
        netlist={
            "instances": {
                "lft": "coupler",
                "top": "waveguide",
                "btm": "waveguide",
                "rgt": "coupler",
            },
            "connections": {
                "lft,out0": "btm,in0",
                "btm,out0": "rgt,in0",
                "lft,out1": "top,in0",
                "top,out0": "rgt,in1",
            },
            "ports": {
                "in0": "lft,in0",
                "in1": "lft,in1",
                "out0": "rgt,out0",
                "out1": "rgt,out1",
            },
        },
        models={
            "coupler": coupler,
            "waveguide": waveguide,
        },
    )

    mzi()
    mzi(top={"length": 25.0}, btm={"length": 15.0})
    wl = jnp.linspace(1.51, 1.59, 1000)
    mzi(wl=wl, top={"length": 25.0}, btm={"length": 15.0})
