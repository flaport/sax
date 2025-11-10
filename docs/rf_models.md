# RF Models

This page documents the RF (Radio Frequency) circuit models available in SAX under the `sax.models.rf` module.

The RF models include impedance, admittance, capacitor, and inductor elements that are essential for microwave circuit simulations.

## Example Usage

```python
import sax.models.rf as rf

# Create an impedance element
s_params = rf.impedance(z=75, z0=50)

# Create a capacitor
import numpy as np
f = np.linspace(1e9, 10e9, 100)
s_cap = rf.capacitor(f=f, capacitance=1e-12, z0=50)
```

## References

The RF models are based on standard microwave engineering theory [@pozar2012].

\bibliography
