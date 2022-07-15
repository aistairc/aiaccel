.. aiaccel-dev documentation master file, created by
   sphinx-quickstart on Mon Jan 18 15:46:13 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

========
aiaccel
========
A hyperparameter optimization library for `AI Bridging Cloud Infrastructure (ABCI)`_.
This software solves hyperparameter optimizations related to AI technologies including deep learning and multi-agent simulation.
The software currently supports five optimization algorithms: random search, grid search, sobol sequence, nelder-mead method, and TPE.


.. _`AI Bridging Cloud Infrastructure (ABCI)`: https://abci.ai/

=============
Installation
=============
The software can be installed using `pip`.

.. code-block:: bash

   pip install git+https://github.com/aistairc/aiaccel.git


=============
User Guide
=============

.. toctree::
   :maxdepth: 2

   manual_jp.rst


=============
API Reference
=============

.. toctree::
   :maxdepth: 2

   aiaccel.rst
   aiaccel.abci.rst
   aiaccel.master.rst
   aiaccel.optimizer.rst
   aiaccel.scheduler.rst
   aiaccel.util.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
