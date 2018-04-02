========
SUFTware
========

.. image:: who.alcohol_consumption.png
   :height: 300px
   :width: 300 px
   :alt: Density estimation using alcohol consumption data from the WHO.
   :align: right

SUFTware (Statistics Using Field Theory) provides fast and lightweight Python
implementations of Bayesian Field Theory algorithms for low-dimensional
statistical inference. SUFTware currently supports the one-dimenstional
density estimation algorithm DEFT, which is described in [#Chen2018]_,
[#Kinney2015]_, and [#Kinney2014]_. As an example, the image on the right shows
DEFT applied to alcohol consumption data from the World Health Organization.
Code for this and other examples can be found here_.

Installation
------------

SUFTware is most easily installed from
`PyPI <https://pypi.python.org/pypi/suftware>`_ using the ``pip`` package
manager::

    $pip install suftware

The code for SUFTware is available on
`GitHub <https://github.com/jbkinney/suftware>`_.

Quick Start
-----------

.. code-block:: python

    import numpy as np
    import suftware as sw

    # Generate random data
    data = np.random.randn(100)

    # Perform one-dimensional density estimation using the DEFT algorithm
    density = sw.DensityEstimator(data)

    # Visualize results
    density.plot()

Documentation
-------------

.. toctree::

     DensityEstimator
     ExampleData
     SimulatedData
     enable_graphics


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
----------

.. [#Chen2018] Chen W, Tareen A, Kinney JB (2018) Density estimation on.
   small datasets *arxiv:XXXX [physics.data-an]*.

.. [#Kinney2015] Kinney JB (2015) Unification of field theory and maximum
   entropy methods for learning probability densities. *Phys Rev E* 92:032107.
   `PDF <../../Kinney2015.pdf>`_.

.. [#Kinney2014] Kinney JB (2014) Estimation of probability densities using
   scale-free field theories. *Phys Rev E* 90:011301(R).
   `PDF <../../Kinney2014.pdf>`_.