SAX
====

:ref:`genindex` · :ref:`modindex` · :ref:`search`

Autograd and XLA for S-parameters - a scatter parameter circuit simulator and
optimizer for the frequency domain based on `JAX <https://github.com/google/jax>`_

The simulator was developed for simulating Photonic Integrated Circuits but in fact is
able to perform any S-parameter based circuit simulation.  The goal of SAX is to be a
light wrapper around JAX with some basic tools for photonic component and circuit
simulation and optimization. Therefore, SAX does not define any special datastructures
and tries to stay as close as possible to the functional nature of JAX.  This makes it
very easy to get started with SAX as you only need functions and standard python
dictionaries. Let's dive in...

.. toctree::
   :maxdepth: 2

   examples
   sax
