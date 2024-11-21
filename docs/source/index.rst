.. aiaccel-dev documentation master file, created by
   sphinx-quickstart on Mon Jan 18 15:46:13 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


aiaccel Documentation
===================
This software is a hyperparameter optimization library designed for the AI Bridging Cloud Infrastructure (ABCI).
It addresses hyperparameter optimization challenges in AI technologies such as deep learning and multi-agent simulations.
Currently, it supports five optimization algorithms: Random Search, Grid Search, Sobol Sequence, Nelder-Mead Method, and TPE.


.. _`AI Bridging Cloud Infrastructure (ABCI)`: https://abci.ai/


.. toctree::
   :maxdepth: 2
   :caption: Installation:

   installation/installation.md


.. toctree::
   :maxdepth: 2
   :caption: Code Reference

   api_reference/modules


Index and Search
==========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Acknowledgments
====
* Part of this work was developed under a commissioned project of the New Energy and Industrial Technology Development Organization (NEDO).
* The TPE algorithm implementation uses Optuna.
