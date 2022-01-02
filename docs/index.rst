SAX
====

:ref:`genindex` · :ref:`modindex` · :ref:`search`

Autograd and XLA for S-parameters - a scatter parameter circuit simulator and
optimizer for the frequency domain based on `JAX <https://github.com/google/jax>`_

The simulator was developed for simulating Photonic Integrated Circuits but in fact is
able to perform any S-parameter based circuit simulation. The goal of SAX is to be a
thin wrapper around JAX with some basic tools for S-parameter based circuit simulation
and optimization. Therefore, SAX does not define any special datastructures and tries to
stay as close as possible to the functional nature of JAX.  This makes it very easy to
get started with SAX as you only need functions and standard python dictionaries. Let's
dive in...

Table Of Contents
-----------------

.. toctree::
   :maxdepth: 2

   examples
   sax


Installation
------------


Dependencies
~~~~~~~~~~~~

-  `JAX & JAXLIB <https://github.com/google/jax>`__. Please read the JAX
   install instructions
   `here <https://github.com/google/jax/#installation>`__.
   Alternatively, you can try running `jaxinstall.sh <https://github.com/flaport/sax/blob/master/jaxinstall.sh>`__
   to automatically pip-install the correct ``jax`` and ``jaxlib``
   package for your python and cuda version (if that exact combination
   exists).


Installation
~~~~~~~~~~~~

::

   pip install sax


License
-------

Copyright © 2021, Floris Laporte, `Apache-2.0 License <https://github.com/flaport/sax/blob/master/LICENSE>`__
