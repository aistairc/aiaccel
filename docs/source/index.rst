.. figure:: _static/logo_aiaccel.png
    :width: 600px

#######################
 Aiaccel Documantation
#######################

Aiaccel is a toolkit for accelerating machine learning research.

.. grid:: 1 1 2 2
    :gutter: 2
    :padding: 0
    :class-row: surface

    .. grid-item-card:: :octicon:`sliders;1.5em;sd-text-primary` Configuration management
       :link: user_guide/config.html

       OmegaConf-based config management.


    .. grid-item-card:: :octicon:`server;1.5em;sd-text-primary` Job management
       :link: user_guide/config.html

       HPC-oriented job abstraction.

    .. grid-item-card:: :octicon:`flame;1.5em;sd-text-primary` PyTorch/Lightning toolkit
       :link: user_guide/torch.html

       Training toolkit for HPC clusters.

    .. grid-item-card:: :octicon:`beaker;1.5em;sd-text-primary` Hyperparameter optimization
       :link: user_guide/hpo.html

       Ready-to-use algorithms/tools.

**************
 Key Features
**************

:octicon:`zap;1em` Research-Oriented
    Designed to accelerate your research cycles written in Python

:octicon:`cpu;1em` HPC Optimized
    Intended to use in high-performance computing (HPC) clusters, including AI Bridging
    Cloud Infrastructure (ABCI).

:octicon:`server;1em` Highly Modular
    Designed to let you pick up any part of aiaccel for your research project.

************************
 Aiaccel is used in ...
************************

- M3L: Multimodal machine listening toolkit (https://github.com/b-sigpro/m3l)
- SBSS: Scalable blind source separation toolkit (https://github.com/b-sigpro/sbss)

*****************
 Acknowledgments
*****************

- Part of this work was developed under a commissioned project of the New Energy and
  Industrial Technology Development Organization (NEDO).
- Part of this software was developed by using ABCI 3.0 provided by AIST and AIST
  Solutions.
- Part of this software was developed by using the TSUBAME4.0 supercomputer at Institute
  of Science Tokyo.

.. toctree::
    :hidden:

    user_guide/index.rst

.. toctree::
    :hidden:
    :maxdepth: 2

    api_reference/index.rst

.. toctree::
    :hidden:
    :maxdepth: 2

    contribution_guide/index.rst
