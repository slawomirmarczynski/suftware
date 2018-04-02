========
SUFTware
========

.. image:: docs/who.alcohol_consumption.png
   :height: 100px
   :width: 100 px
   :alt: Density estimation using alcohol consumption data from the WHO.
   :align: right

SUFTware (Statistics Using Field Theory) provides fast and lightweight Python
implementations of Bayesian Field Theory algorithms for low-dimensional
statistical inference. SUFTware currently supports the one-dimenstional
density estimation algorithm DEFT, described in [#Chen2018]_,
[#Kinney2015]_, and [#Kinney2014]_. 

The image on the right shows DEFT applied
to alcohol consumption data from
the World Health Organization. This computation takes about 0.25 seconds on
a standard laptop computer. 

References
----------

.. [#Chen2018] Chen W, Tareen A, Kinney JB (2018) Density estimation on.
   small datasets *arxiv:XXXX [physics.data-an]*.

.. [#Kinney2015] Kinney JB (2015) Unification of field theory and maximum
   entropy methods for learning probability densities. *Phys Rev E* 92:032107.
   :download:`PDF <docs/Kinney2015.pdf>`.

.. [#Kinney2014] Kinney JB (2014) Estimation of probability densities using
   scale-free field theories. *Phys Rev E* 90:011301(R).
   :download:`PDF <docs/Kinney2014.pdf>`.
