""" SAX thin-film models """

import jax.numpy as jnp
from ..utils import zero
from ..core import modelgenerator
from ..typing import Dict, ModelDict, ComplexFloat


#######################
## Fresnel interface ##
#######################

def r_fresnel_ij(params: Dict[str, float]) -> ComplexFloat:
    """
    Normal incidence amplitude reflection from Fresnel's equations
    ni : refractive index of the initial medium
    nj : refractive index of the final medium
    """
    return (params["ni"] - params["nj"]) / (params["ni"] + params["nj"])

def t_fresnel_ij(params: Dict[str, float]) -> ComplexFloat:
    """
    Normal incidence amplitude transmission from Fresnel's equations
    ni : refractive index of the initial medium
    nj : refractive index of the final medium
    """
    return 2 * params["ni"] / (params["ni"] + params["nj"])

fresnel_mirror_ij = {
    ("in", "in"): r_fresnel_ij,
    ("in", "out"): t_fresnel_ij,
    ("out", "in"): lambda params: (1 - r_fresnel_ij(params)**2)/t_fresnel_ij(params), # t_ji,
    ("out", "out"): lambda params: -1*r_fresnel_ij(params), # r_ji,
    "default_params": {
        "ni": 1.,
        "nj": 1.,
    }
}

#################
## Propagation ##
#################

def prop_i(params: Dict[str, float]) -> ComplexFloat:
    """
    Phase shift acquired as a wave propagates through medium i
    wl : wavelength (arb. units)
    ni : refractive index of medium (at wavelength wl)
    di : thickness of layer (same arb. unit as wl)
    """
    return jnp.exp(1j * 2*jnp.pi * params["ni"] / params["wl"] * params["di"])

propagation_i = {
    ("in", "out"): prop_i,
    ("out", "in"): prop_i,
    "default_params": {
        "ni": 1.,
        "di": 500.,
        "wl": 532.,
    }
}

#################################
## Lossless reciprocal element ##
#################################

def t_complex(params: Dict[str, float]) -> ComplexFloat:
    """
    Transmission coefficient (design parameter)
    """
    return params['t_amp']*jnp.exp(-1j*params['t_ang'])

def r_complex(params: Dict[str, float]) -> ComplexFloat:
    """
    Reflection coefficient, derived from transmission coefficient
    Magnitude from |t|^2 + |r|^2 = 1
    Phase from phase(t) - phase(r) = pi/2
    """
    r_amp = jnp.sqrt( ( 1. - params['t_amp']**2 ) )
    r_ang = params['t_ang'] - jnp.pi/2
    return r_amp*jnp.exp(-1j*r_ang)

mirror = {
    ("in", "in"): r_complex,
    ("in", "out"): t_complex,
    ("out", "in"): t_complex,
    ("out", "out"): r_complex,
    "default_params": {
        "t_amp": jnp.sqrt(0.5),
        "t_ang": 0.0,
    }
}