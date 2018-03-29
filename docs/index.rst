.. SUFTware documentation master file, created by
   sphinx-quickstart on Wed Mar 28 14:28:45 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

========
SUFTware
========

---------------------------------------
Statistics Using Field Theory
---------------------------------------

.. image:: who.alcohol_consumption.png
   :height: 300px
   :width: 300 px
   :alt: Example density estimate using WHO data.
   :align: right

SUFTware is a lightweight Python package that provides provides fast and robust implementations of Bayesian Field Theory (BFT) methods for low-dimensional statistical inference. BFT is a grid-based approach to Bayesian nonparametric inference. By using a grid in lieu of specific stochastic processes (such as Dirichlet processes or Gaussian processes), BFT allows certain types of problems to be solved in a fully Bayesian manner without requiring any large-data approximations.

Currently, SUFTware supports a one-dimensional density estimation called DEFT. DEFT has substantial advantages over standard density estimation methods, including, including kernel density estimation and Dirichlet process mixture modeling. See [Chen et al., 2018; Kinney 2015; Kinney 2014].


Installation
------------

.. parsed-literal::
    pip install suftware

Requirements

- Python >= 3.6.3
- numpy >= 1.13.3
- scipy >= 1.0.0
- matplotlib >= 2.1.0

Quick Start
-----------

.. code-block:: python

    import numpy as np
    import suftware as sw

    # Generate random data
    data = np.random.randn(100)

    # Perform one-dimensional density estimation using SUFTware
    density = sw.Density(data)

    # Visualize results
    density.plot()

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
