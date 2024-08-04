""" SAX CUDA Backend """

from __future__ import annotations

from typing import Any, Dict

import cupy as cp
import cupyx
import cupyx.scipy.sparse.linalg
import jax.numpy as jnp
from natsort import natsorted

from ..netlist import Component
from ..saxtypes import Model, SCoo, SDense, SType, scoo


def scoo_cupy(S):
    Si, Sj, Sx, ports_map = scoo(S)
    return cp.asarray(Si), cp.asarray(Sj), cp.asarray(Sx), ports_map


def solve_cuda(Ai, Aj, Ax, B):
    """
    Custom solver using CuPy for sparse matrix solve.

    Args:
        Ai (array): Row indices of non-zero values in the sparse matrix.
        Aj (array): Column indices of non-zero values in the sparse matrix.
        Ax (array): Non-zero values of the sparse matrix.
        B (array): Right-hand side matrix to solve for.

    Returns: array: Solution matrix.
    """
    results = []
    for Ax_mat in Ax:
        # Create sparse matrix in COO format
        A_coo = cupyx.scipy.sparse.coo_matrix((Ax_mat, (Ai, Aj)))

        # Convert to CSR format for solving
        A_csr = A_coo.tocsr()

        # Solve the linear system
        results.append(cupyx.scipy.sparse.linalg.spsolve(A_csr, B))

    return cp.asarray(results)


def coo_mul_vec(Si, Sj, Sx, x):
    """
    COO matrix-vector multiplication using CuPy.

    Args:
        Si (array): Row indices of non-zero values in the sparse matrix.
        Sj (array): Column indices of non-zero values in the sparse matrix.
        Sx (array): Non-zero values of the sparse matrix.
        x (array): Dense vector to multiply with the sparse matrix.

    Returns:
        array: Resulting vector from the multiplication.
    """
    results = []
    for Sx_mat, x_vec in zip(Sx, x):
        # Create sparse matrix in COO format
        S_coo = cupyx.scipy.sparse.coo_matrix((Sx_mat, (Si, Sj)))

        # Perform the matrix-vector multiplication
        results.append(S_coo.dot(x_vec))

    return cp.asarray(results)


def analyze_instances_cuda(
    instances: Dict[str, Component],
    models: Dict[str, Model],
) -> Dict[str, SCoo]:
    instances, instances_old = {}, instances
    for k, v in instances_old.items():
        if not isinstance(v, Component):
            v = Component(**v)
        instances[k] = v
    model_names = set()
    for i in instances.values():
        if i.info and "model" in i.info and isinstance(i.info["model"], str):
            model_names.add(str(i.info["model"]))
        else:
            model_names.add(str(i.component))
    dummy_models = {k: scoo_cupy(models[k]()) for k in model_names}
    dummy_instances = {}
    for k, i in instances.items():
        if i.info and "model" in i.info and isinstance(i.info["model"], str):
            dummy_instances[k] = dummy_models[str(i.info["model"])]
        else:
            dummy_instances[k] = dummy_models[str(i.component)]
    return dummy_instances


def analyze_circuit_cuda(
    analyzed_instances: Dict[str, SCoo],
    connections: Dict[str, str],
    ports: Dict[str, str],
) -> Any:
    connections = {**connections, **{v: k for k, v in connections.items()}}
    inverse_ports = {v: k for k, v in ports.items()}
    port_map = {k: i for i, k in enumerate(ports)}

    idx, Si, Sj, instance_ports = 0, [], [], {}
    for name, instance in analyzed_instances.items():
        si, sj, _, ports_map = scoo_cupy(instance)
        Si.append(cp.asarray(si + idx))
        Sj.append(cp.asarray(sj + idx))
        instance_ports.update({f"{name},{p}": i + idx for p, i in ports_map.items()})
        idx += len(ports_map)

    n_col = idx
    n_rhs = len(port_map)

    Si = cp.concatenate(Si, -1)
    Sj = cp.concatenate(Sj, -1)

    Cmap = {
        int(instance_ports[k]): int(instance_ports[v]) for k, v in connections.items()
    }
    Ci = cp.array(list(Cmap.keys()), dtype=cp.int32)
    Cj = cp.array(list(Cmap.values()), dtype=cp.int32)

    Cextmap = {
        int(instance_ports[k]): int(port_map[v]) for k, v in inverse_ports.items()
    }
    Cexti = cp.asarray(list(Cextmap.keys()))
    Cextj = cp.asarray(list(Cextmap.values()))
    Cext = cp.zeros((n_col, n_rhs), dtype=complex)
    Cext[Cexti, Cextj] = 1.0

    mask = Cj[None, :] == Si[:, None]
    CSi = cp.broadcast_to(Ci[None, :], mask.shape)[mask]

    mask = (Cj[:, None] == Si[None, :]).any(0)
    CSj = Sj[mask]

    Ii = Ij = cp.arange(n_col)
    I_CSi = cp.concatenate([CSi, Ii], -1)
    I_CSj = cp.concatenate([CSj, Ij], -1)
    return (
        n_col,
        mask,
        Si,
        Sj,
        Cext,
        Cexti,
        Cextj,
        I_CSi,
        I_CSj,
        tuple((k, v[1]) for k, v in analyzed_instances.items()),
        tuple(port_map),
    )


def evaluate_circuit_cuda(analyzed: Any, instances: Dict[str, SType]) -> SDense:
    (
        n_col,
        mask,
        Si,
        Sj,
        Cext,
        Cexti,
        Cextj,
        I_CSi,
        I_CSj,
        dummy_pms,
        port_map,
    ) = analyzed

    idx = 0
    Sx = []
    batch_shape = ()
    for name, pm_ in dummy_pms:
        _, _, sx, ports_map = scoo_cupy(instances[name])
        Sx.append(sx)
        if len(sx.shape[:-1]) > len(batch_shape):
            batch_shape = sx.shape[:-1]
        idx += len(ports_map)

    Sx = cp.concatenate(
        [cp.broadcast_to(sx, (*batch_shape, sx.shape[-1])) for sx in Sx], -1
    )
    CSx = Sx[..., mask]
    Ix = cp.ones((*batch_shape, n_col))
    I_CSx = cp.concatenate([-CSx, Ix], -1)

    Sx = Sx.reshape(-1, Sx.shape[-1])  # n_lhs x N
    I_CSx = I_CSx.reshape(-1, I_CSx.shape[-1])  # n_lhs x M
    inv_I_CS_Cext = solve_cuda(I_CSi, I_CSj, I_CSx, Cext)
    S_inv_I_CS_Cext = coo_mul_vec(Si, Sj, Sx, inv_I_CS_Cext)

    CextT_S_inv_I_CS_Cext = S_inv_I_CS_Cext[..., Cexti, :][..., :, Cextj]

    _, n, _ = CextT_S_inv_I_CS_Cext.shape
    S = CextT_S_inv_I_CS_Cext.reshape(*batch_shape, n, n)

    return jnp.asarray(S), {p: i for i, p in enumerate(port_map)}
