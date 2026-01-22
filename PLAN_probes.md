# Design Plan: Probes Feature for SAX

## Overview

Add a `probes` keyword argument to the `circuit()` function that allows inserting ideal 4-port measurement taps at connection points. Each probe intercepts a connection and exposes forward (`_fwd`) and backward (`_bwd`) traveling wave ports.

**Design Decision:** Probes are passed as a `circuit()` argument rather than added to the netlist schema. This keeps the netlist schema compatible with GDSFactory and treats probes as a simulation-time debugging feature rather than part of the circuit definition.

## User-Facing API

### Circuit Call with Probes

```python
netlist = {
    "instances": {
        "wg1": "waveguide",
        "wg2": "waveguide",
    },
    "connections": {
        "wg1,out": "wg2,in",  # Connection to be probed
    },
    "ports": {
        "in": "wg1,in",
        "out": "wg2,out",
    },
}

circuit_fn, info = sax.circuit(
    netlist,
    models=models,
    probes={"mid": "wg1,out"},  # Probe named "mid" at connection "wg1,out"
)
```

### Result

The circuit will have ports: `in`, `out`, `mid_fwd`, `mid_bwd`

- `mid_fwd`: Signal traveling from `wg1,out` toward `wg2,in`
- `mid_bwd`: Signal traveling from `wg2,in` toward `wg1,out`

## The Ideal Probe Model (4-port)

```
          in ─────────────────── out
           │                     │
           │     (ideal tap)     │
           │                     │
         tap_bwd               tap_fwd
```

### S-Matrix Definition

An ideal directional coupler with 100% transmission AND 100% tap coupling (unphysical but useful for measurement):

```python
def ideal_probe() -> sax.SDict:
    """Ideal 4-port probe: 100% transmission, 100% tap coupling."""
    return {
        # Through path: full transmission
        ("in", "out"): 1.0,
        ("out", "in"): 1.0,

        # Forward tap: copies signal from in→out direction
        ("in", "tap_fwd"): 1.0,
        ("tap_fwd", "in"): 1.0,

        # Backward tap: copies signal from out→in direction
        ("out", "tap_bwd"): 1.0,
        ("tap_bwd", "out"): 1.0,

        # No cross-coupling between taps
        ("tap_fwd", "tap_bwd"): 0.0,
        ("tap_bwd", "tap_fwd"): 0.0,

        # No coupling from taps back to main path (unidirectional taps)
        ("tap_fwd", "out"): 0.0,
        ("tap_bwd", "in"): 0.0,

        # No reflections
        ("in", "in"): 0.0,
        ("out", "out"): 0.0,
        ("tap_fwd", "tap_fwd"): 0.0,
        ("tap_bwd", "tap_bwd"): 0.0,
    }
```

**Note:** This S-matrix is NOT unitary (violates energy conservation). This is intentional—it's a measurement tool, not a physical device.

## Internal Transformation

When processing the netlist, the `probes` section triggers a transformation:

### Before (user-provided):
```python
{
    "instances": {"wg1": "waveguide", "wg2": "waveguide"},
    "connections": {"wg1,out": "wg2,in"},
    "ports": {"in": "wg1,in", "out": "wg2,out"},
    "probes": {"mid": "wg1,out"},
}
```

### After (internal):
```python
{
    "instances": {
        "wg1": "waveguide",
        "wg2": "waveguide",
        "_probe_mid": {"component": "_ideal_probe"},
    },
    "connections": {
        "wg1,out": "_probe_mid,in",
        "_probe_mid,out": "wg2,in",
    },
    "ports": {
        "in": "wg1,in",
        "out": "wg2,out",
        "mid_fwd": "_probe_mid,tap_fwd",
        "mid_bwd": "_probe_mid,tap_bwd",
    },
    # probes section removed after processing
}
```

## Implementation Plan

### 1. Add Ideal Probe Model (`src/sax/models/probes.py`)

Create a new file with the `ideal_probe` model function.

### 2. Add Probe Expansion Logic (`src/sax/netlists.py`)

```python
def expand_probes(
    netlist: sax.Netlist,
    probes: dict[str, str],
) -> sax.Netlist:
    """Expand probes into ideal_probe instances and update connections/ports."""
```

This function:
1. For each probe name → instance_port mapping:
   - Validate the instance_port is part of a connection
   - Find the connection containing this instance_port
   - Insert `_probe_{name}` instance
   - Split the connection: `a,p1 → b,p2` becomes `a,p1 → _probe_{name},in` and `_probe_{name},out → b,p2`
   - Add ports: `{name}_fwd → _probe_{name},tap_fwd` and `{name}_bwd → _probe_{name},tap_bwd`
2. Return the modified netlist

### 3. Integrate into Circuit Compilation (`src/sax/circuits.py`)

Add `probes` parameter to the `circuit()` function signature:

```python
def circuit(
    netlist: sax.AnyNetlist,
    models: sax.Models | None = None,
    *,
    backend: sax.BackendLike = "default",
    return_type: Literal["SDict", "SDense", "SCoo"] = "SDict",
    top_level_name: str = "top_level",
    ignore_impossible_connections: bool = False,
    probes: dict[str, str] | None = None,  # NEW
) -> tuple[sax.Model, sax.CircuitInfo]:
```

In the function body, after `convert_nets_to_connections()` and before `resolve_array_instances()`:

```python
recnet = convert_nets_to_connections(recnet)
if probes:
    recnet = expand_probes(recnet, probes)  # NEW
    models = {"_ideal_probe": ideal_probe, **(models or {})}  # Inject probe model
recnet = resolve_array_instances(recnet)
```

### 4. Update Exports

- Add `ideal_probe` to `sax.models` exports
- Optionally add `expand_probes` to `sax.netlists` exports (could remain internal)

### 5. Add Tests (`src/tests/test_probes.py`)

Test cases:
1. Basic probe insertion on a simple 2-waveguide circuit
2. Multiple probes on different connections
3. Probe on a connection in a more complex circuit
4. Error case: probe on non-existent instance port
5. Error case: probe on instance port not part of any connection
6. Verify S-matrix values at probe ports

### 6. Add Documentation/Example

Consider adding an example notebook demonstrating probe usage.

## Files to Modify

| File | Changes |
|------|---------|
| `src/sax/models/probes.py` | NEW: `ideal_probe()` model |
| `src/sax/models/__init__.py` | Export `ideal_probe` |
| `src/sax/netlists.py` | Add `expand_probes()` function |
| `src/sax/circuits.py` | Add `probes` parameter, call `expand_probes()`, inject probe model |
| `src/tests/test_probes.py` | NEW: test cases |

## Edge Cases & Validation

1. **Probe on non-connected port**: Error - "Probe 'X' references instance port 'Y' which is not part of any connection"

2. **Duplicate probe names**: Handled naturally by dict (last one wins), but could warn

3. **Probe name conflicts with existing ports**: Error - "Probe 'X' would create ports 'X_fwd'/'X_bwd' which conflict with existing ports"

4. **Probe instance name conflicts**: Use `_probe_` prefix to minimize conflicts; if conflict exists, error

5. **Hierarchical netlists**: Probes are expanded on the top-level netlist only (after flattening conceptually). This is handled naturally since `expand_probes` operates on the recursive netlist before sub-circuits are resolved.

6. **Empty probes dict**: No-op, just return netlist unchanged

## Open Questions

1. **Naming convention**: `mid_fwd`/`mid_bwd` vs `mid@fwd`/`mid@bwd` vs `mid:fwd`/`mid:bwd`?
   - Recommendation: `_fwd`/`_bwd` suffix is simple and avoids special characters

2. **Should probes accept parameters?** (e.g., partial coupling for more realistic taps)
   - Recommendation: Start simple with ideal probes; can add `probe_settings` later if needed

3. **Multimode support**: Should probes work with multimode simulations?
   - Recommendation: Yes, the transformation happens before multimode expansion, so it should work automatically

4. **Should we allow probes on external ports?** (ports that connect to the outside)
   - Recommendation: No, only on internal connections. External ports are already observable.

## Benefits of This Approach

1. **GDSFactory compatibility**: Netlist schema is unchanged
2. **Separation of concerns**: Physical circuit definition vs simulation debugging
3. **Flexibility**: Same netlist can be simulated with or without probes
4. **Non-invasive**: Probes don't get saved to YAML/JSON files (they're transient)
5. **Simple API**: Just one new keyword argument to `circuit()`
