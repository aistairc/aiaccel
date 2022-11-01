.. aiaccel-dev documentation master file, created by
   sphinx-quickstart on Mon Jan 18 15:46:13 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


aiaccelドキュメント
===================
本ソフトウェアは、AI Bridging Cloud Infrastructure (ABCI)`_ のためのハイパーパラメータ最適化ライブラリです。
ディープラーニングやマルチエージェントシミュレーションなどのAI技術に関連するハイパーパラメータ最適化を解決します。
現在、ランダムサーチ、グリッドサーチ、ソボルシーケンス、ネルダーミード法、TPEの5つの最適化アルゴリズムに対応しています。


.. _`AI Bridging Cloud Infrastructure (ABCI)`: https://abci.ai/

.. toctree::
   :maxdepth: 2
   :caption: オーバービュー:

   overview/overview.md

..
   toctree::
   :maxdepth: 2
   :caption: インストール:

   installation/installation.md

..
   toctree::
   :maxdepth: 2
   :caption: 利用例:

   examples/examples.md

.. toctree::
   :maxdepth: 2
   :caption: ユーザーガイド:

   user_guide/basic_usage.md
.. user_guide/configuration_setting.md

.. toctree::
   :maxdepth: 2
   :caption: ディベロッパーガイド:

   developer_guide/custom_optimizer.md
.. developer_guide/architecture.md

..
   toctree::
   :maxdepth: 2
   :caption: コントリビューションガイド:

   contribution_guide/contribution_guide.md

..
   toctree::
   :maxdepth: 2
   :caption: リファレンス:

   references/references.md

.. toctree::
   :maxdepth: 2
   :caption: コードリファレンス

   api_reference/aiaccel

索引と検索
==========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

謝辞
====
* この成果の一部は、国立研究開発法人新エネルギー・産業技術総合開発機構(NEDO)の委託業務として開発されたものです。
* TPEアルゴリズムは Optuna を利用しました。
