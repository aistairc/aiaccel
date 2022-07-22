.. aiaccel-dev documentation master file, created by
   sphinx-quickstart on Mon Jan 18 15:46:13 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


aiaccel
========
本ソフトウェアは、AI Bridging Cloud Infrastructure (ABCI)`_ のためのハイパーパラメータ最適化ライブラリです。
ディープラーニングやマルチエージェントシミュレーションなどのAI技術に関連するハイパーパラメータ最適化を解決します。
現在、ランダムサーチ、グリッドサーチ、ソボルシーケンス、ネルダーミード法、TPEの5つの最適化アルゴリズムに対応しています。


.. _`AI Bridging Cloud Infrastructure (ABCI)`: https://abci.ai/


Installation
=============
このソフトは `pip` を使ってインストールすることができます。

.. code-block:: bash

   pip install git+https://github.com/aistairc/aiaccel.git


User Guide
=============

* :doc:`./manual_jp`


Guide for adding an optimizer
=============================

* :doc:`./optimizer_jp`


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
