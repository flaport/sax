"""SAX KLU Backend."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import klujax
from natsort import natsorted
from typing_extensions import TypedDict

import sax

__all__ = [
    "KLUAnalyzed",
    "analyze_circuit_klu",
    "analyze_instances_klu",
    "evaluate_circuit_klu",
]


class KLUAnalyzed(TypedDict):
    """Pre-computed analysis data for the KLU sparse matrix backend."""

    n_col: int
    cs_s_indices: Any  # jax array: indices into Sx for CS product
    Si: Any  # jax array: row indices of block-diagonal S
    Sj: Any  # jax array: col indices of block-diagonal S
    Cext: Any  # jax array: external port coupling matrix (n_col x n_rhs)
    Cexti: Any  # jax array: row indices of Cext nonzeros
    Cextj: Any  # jax array: col indices of Cext nonzeros
    I_CSi: Any  # jax array: row indices of (I-CS) system matrix
    I_CSj: Any  # jax array: col indices of (I-CS) system matrix
    instance_names: tuple[str, ...]  # ordered instance names
    port_map: tuple[str, ...]  # external port names in order
    symbolic: klujax.KLUHandleManager  # pre-computed symbolic sparsity analysis


def analyze_instances_klu(
    instances: dict[sax.InstanceName, sax.Instance],
    models: dict[str, sax.Model],
) -> dict[str, sax.SCoo]:
    """Analyze circuit instances for the KLU backend.

    Prepares instance S-matrices for the KLU backend by converting all component
    models to SCoo (coordinate) format. The KLU backend uses sparse matrix
    techniques with the KLU solver for high-performance circuit evaluation.

    Args:
        instances: Dictionary mapping instance names to instance definitions
            containing component names and settings.
        models: Dictionary mapping component names to their model functions.

    Returns:
        Dictionary mapping instance names to their S-matrices in SCoo format.

    Note:
        The KLU backend is the recommended high-performance backend for most
        circuit simulations. It uses sparse matrix factorization and can handle
        large circuits efficiently with full bidirectional coupling and reflections.

    Example:
        ```python
        instances = {
            "wg1": {"component": "waveguide", "settings": {"length": 10.0}},
            "dc1": {"component": "coupler", "settings": {"coupling": 0.1}},
        }
        models = {"waveguide": waveguide_model, "coupler": coupler_model}
        analyzed = analyze_instances_klu(instances, models)
        ```
    """
    instances = sax.into[sax.Instances](instances)
    model_names = set()
    for i in instances.values():
        model_names.add(i["component"])
    dummy_models = {k: sax.scoo(models[k]()) for k in model_names}
    dummy_instances = {}
    for k, i in instances.items():
        dummy_instances[k] = dummy_models[i["component"]]
    return dummy_instances


def analyze_circuit_klu(
    analyzed_instances: dict[sax.InstanceName, sax.SCoo],
    nets: sax.Nets,
    ports: sax.Ports,
) -> KLUAnalyzed:
    """Analyze circuit topology for the KLU sparse matrix backend.

    Performs detailed circuit analysis to set up the sparse matrix system for
    the KLU solver. This includes building the connection and S-matrix index
    arrays, and running KLU's symbolic sparsity analysis on the (I-CS) system
    matrix. The symbolic analysis is cached so that subsequent evaluations
    only need numeric factorization.

    Args:
        analyzed_instances: Instance S-matrices from analyze_instances_klu in
            SCoo format.
        nets: List of net dictionaries with "p1" and "p2" keys defining
            internal circuit connections. Supports multiply connected ports.
        ports: Dictionary mapping external port names to instance ports.

    Returns:
        KLUAnalyzed TypedDict containing sparse matrix indices, connection
        mappings, external port information, and pre-computed symbolic
        analysis for the KLU solver.

    Example:
        ```python
        nets = [{"p1": "wg1,out", "p2": "dc1,in1"}, {"p1": "dc1,out1", "p2": "wg2,in"}]
        ports = {"in": "wg1,in", "out": "wg2,out"}
        analyzed = analyze_circuit_klu(analyzed_instances, nets, ports)
        ```
    """
    inverse_ports = {v: k for k, v in ports.items()}
    port_map = {k: i for i, k in enumerate(ports)}

    idx, Si, Sj, instance_ports = 0, [], [], {}
    for name, instance in analyzed_instances.items():
        si, sj, _, ports_map = sax.scoo(instance)
        Si.append(si + idx)
        Sj.append(sj + idx)
        instance_ports.update({f"{name},{p}": i + idx for p, i in ports_map.items()})
        idx += len(ports_map)

    n_col = idx
    n_rhs = len(port_map)

    Si = jnp.concatenate(Si, -1)
    Sj = jnp.concatenate(Sj, -1)

    pairs: set[tuple[int, int]] = set()
    for net in nets:
        p1_idx = int(instance_ports[net["p1"]])
        p2_idx = int(instance_ports[net["p2"]])
        pairs.add((p1_idx, p2_idx))
        pairs.add((p2_idx, p1_idx))
    sorted_pairs = sorted(pairs)
    Ci = jnp.array([p[0] for p in sorted_pairs], dtype=jnp.int32)
    Cj = jnp.array([p[1] for p in sorted_pairs], dtype=jnp.int32)

    Cextmap = {
        int(instance_ports[k]): int(port_map[v]) for k, v in inverse_ports.items()
    }
    Cexti = jnp.stack(list(Cextmap.keys()), 0)
    Cextj = jnp.stack(list(Cextmap.values()), 0)
    Cext = jnp.zeros((n_col, n_rhs), dtype=complex).at[Cexti, Cextj].set(1.0)

    match_2d = Cj[None, :] == Si[:, None]  # (len_Si, len_Cj)
    CSi = jnp.broadcast_to(Ci[None, :], match_2d.shape)[match_2d]
    s_idx_grid = jnp.broadcast_to(jnp.arange(len(Si))[:, None], match_2d.shape)
    cs_s_indices = s_idx_grid[match_2d]
    CSj = Sj[cs_s_indices]

    Ii = Ij = jnp.arange(n_col, dtype=jnp.int32)
    I_CSi = jnp.concatenate([CSi, Ii], -1).astype(jnp.int32)
    I_CSj = jnp.concatenate([CSj, Ij], -1).astype(jnp.int32)

    symbolic = klujax.analyze(I_CSi, I_CSj, n_col)

    return KLUAnalyzed(
        n_col=n_col,
        cs_s_indices=cs_s_indices,
        Si=Si,
        Sj=Sj,
        Cext=Cext,
        Cexti=Cexti,
        Cextj=Cextj,
        I_CSi=I_CSi,
        I_CSj=I_CSj,
        instance_names=tuple(analyzed_instances.keys()),
        port_map=tuple(port_map),
        symbolic=symbolic,
    )


def evaluate_circuit_klu(
    analyzed: KLUAnalyzed,
    instances: dict[sax.InstanceName, sax.SType],
) -> sax.SDense:
    """Evaluate circuit S-matrix using the KLU sparse matrix solver.

    Computes the circuit S-matrix by solving the sparse linear system
    (I - CS)x = C_ext using the high-performance KLU sparse matrix solver.
    Uses the pre-computed symbolic analysis from analyze_circuit_klu to skip
    redundant sparsity analysis on each evaluation.

    The algorithm:
    1. Assembles the sparse connection matrix C and S-matrix blocks
    2. Forms the system matrix (I - CS) where I is identity
    3. Solves the linear system using KLU factorization (with cached symbolic)
    4. Extracts the external port S-matrix from the solution

    Args:
        analyzed: KLUAnalyzed dict from analyze_circuit_klu containing
            pre-computed sparse matrix indices, mappings, and symbolic analysis.
        instances: Dictionary mapping instance names to their evaluated S-matrices
            in any SAX format (will be converted to SCoo).

    Returns:
        Circuit S-matrix in SDense format (dense matrix with port mapping).

    Example:
        ```python
        instances = {
            "wg1": {("in", "out"): 0.95 * jnp.exp(1j * 0.1)},
            "dc1": {
                ("in1", "out1"): 0.9,
                ("in1", "out2"): 0.1,
                ("in2", "out1"): 0.1,
                ("in2", "out2"): 0.9,
            },
        }
        circuit_s_matrix, port_map = evaluate_circuit_klu(analyzed, instances)
        ```
    """
    n_col = analyzed["n_col"]
    cs_s_indices = analyzed["cs_s_indices"]
    Si = analyzed["Si"]
    Sj = analyzed["Sj"]
    Cext = analyzed["Cext"]
    Cexti = analyzed["Cexti"]
    Cextj = analyzed["Cextj"]
    I_CSi = analyzed["I_CSi"]
    I_CSj = analyzed["I_CSj"]
    symbolic = analyzed["symbolic"]
    port_map = analyzed["port_map"]

    Sx = []
    batch_shape = ()
    for name in analyzed["instance_names"]:
        _, _, sx, _ = sax.scoo(instances[name])
        Sx.append(sx)
        if len(sx.shape[:-1]) > len(batch_shape):
            batch_shape = sx.shape[:-1]

    Sx = jnp.concatenate(
        [jnp.broadcast_to(sx, (*batch_shape, sx.shape[-1])) for sx in Sx], -1
    )
    CSx = Sx[..., cs_s_indices]
    Ix = jnp.ones((*batch_shape, n_col))
    I_CSx = jnp.concatenate([-CSx, Ix], -1)

    Sx = Sx.reshape(-1, Sx.shape[-1])  # n_lhs x N
    I_CSx = I_CSx.reshape(-1, I_CSx.shape[-1])  # n_lhs x M
    solve_klu = jax.vmap(klujax.solve_with_symbol, (None, None, 0, None, None), 0)
    inv_I_CS_Cext = solve_klu(I_CSi, I_CSj, I_CSx, Cext, symbolic)
    mul_coo = jax.vmap(klujax.dot, (None, None, 0, 0), 0)
    S_inv_I_CS_Cext = mul_coo(Si, Sj, Sx, inv_I_CS_Cext)

    CextT_S_inv_I_CS_Cext = S_inv_I_CS_Cext[..., Cexti, :][..., :, Cextj]

    _, n, _ = CextT_S_inv_I_CS_Cext.shape
    S = CextT_S_inv_I_CS_Cext.reshape(*batch_shape, n, n)

    return S, {p: i for i, p in enumerate(port_map)}


def _get_instance_ports(
    connections: dict[str, str], ports: dict[str, str]
) -> dict[str, list[str]]:
    instance_ports = {}
    for connection in connections.items():
        for ip in connection:
            i, p = ip.split(",")
            if i not in instance_ports:
                instance_ports[i] = set()
            instance_ports[i].add(p)
    for ip in ports.values():
        i, p = ip.split(",")
        if i not in instance_ports:
            instance_ports[i] = set()
        instance_ports[i].add(p)
    return {k: natsorted(v) for k, v in instance_ports.items()}


def _get_dummy_instances(
    connections: dict[str, str],
    ports: dict[str, str],
) -> dict[str, tuple[Any, Any, None, dict[str, int]]]:
    """No longer used. deprecated by analyze_instances_klu."""
    instance_ports = _get_instance_ports(connections, ports)
    dummy_instances = {}
    for name, _ports in instance_ports.items():
        num_ports = len(_ports)
        pm = {p: i for i, p in enumerate(_ports)}
        ij = jnp.mgrid[:num_ports, :num_ports]
        i = ij[0].ravel()
        j = ij[1].ravel()
        dummy_instances[name] = (i, j, None, pm)
    return dummy_instances
