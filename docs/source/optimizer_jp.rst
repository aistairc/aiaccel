(WIP:下書き以前の状態)オプティマイザを追加する手順について
===============================================================


1. 場所
=====================================

aiaccel/optimizerに新しくディレクトリを作成する.
デイレクトリの名前はオプティマイザの名前


2. ファイルの構成
=====================================

「__init__.py」「search.py」を作成する.
__init__pyは無記述


3. AbstractOptimizerの継承
=====================================

.. toctree::
    :maxdepth: 2

    optimizer_function.rst

**search.py**

.. code-block:: python

    from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer


    class MyOptimizer(AbstractOptimizer):

        def __init__(self, options: dict) -> None:
            super().__init__(options)

            # write your code here

        def pre_process(self) -> None:
            super().pre_process()

            # write your code here

        def generate_parameter(self, number: Optional[int] = 1) -> None:

            # write your code here

        def _serialize(self) -> None:
            self.serialize_datas = {
                'param': self.param
            }
            return super()._serialize()

        def _deserialize(self, dict_objects: dict) -> None:
            super()._deserialize(dict_objects)

            self.param = dict_objects['param']



4. aiaccel/commons.py
=======================

search_algorithm_myoptimizer = 'myoptimizer'
myoptimizerは任意の名前


5. aiaccel/config.py
=======================

.. code-block:: python

    def hps_format_check(self):
        algorithm = self.search_algorithm.get()
        hyperparameters = self.hyperparameters.get()

        if hyperparameters == []:
            Terminal().print_error("'hyperparameters' are empty.")
            sys.exit()

        # === item check (individual)===
        if algorithm.lower() == search_algorithm_random:
            self._check_random_setting_format(algorithm, hyperparameters)

        elif algorithm.lower() == search_algorithm_grid:
            self._check_grid_setting_format(algorithm, hyperparameters)

        elif algorithm.lower() == search_algorithm_sobol:
            self._check_sobol_setting_format(algorithm, hyperparameters)

        elif algorithm.lower() == search_algorithm_nelder_mead:
            self._check_neldermead_setting_format(algorithm, hyperparameters)

        elif algorithm.lower() == search_algorithm_tpe:
            self._check_tpe_setting_format(algorithm, hyperparameters)

        elif algorithm.lower() == search_algorithm_myoptimizer:
            self._check_myoptimizer_setting_format(algorithm, hyperparameters)

        else:
            Terminal().print_error(f"algorithm: {algorithm} is not suportted.")
            sys.exit()



New create
------------------

.. code-block:: python

    def _check_myoptimizer_setting_format(
        self,
        algorithm: str,
        hyperparameters: list
    ) -> None:

        hp_types = [
            'uniform_float',
            'uniform_int',
            'categorical',
            'ordinal'
        ]
        fmt = FormatChecker(algorithm, hp_types, hyperparameters)

        # int, float
        necessary_items = ["name", "type", "lower", "upper"]
        optional_items = ["initial", "comment"]
        fmt.check_uniform_int(necessary_items, optional_items)
        fmt.check_uniform_float(necessary_items, optional_items)

        # categorical
        necessary_items = ["name", "type", "choices"]
        optional_items = ["initial", "comment"]
        fmt.check_categorical(necessary_items, optional_items)

        # ordinal
        necessary_items = ["name", "type", "lower", "upper", "sequence"]
        optional_items = ["initial", "comment"]
        fmt.check_ordinal(necessary_items, optional_items)

        # initial check
        fmt.check_initial_type([int, float, str])

不要ならば．

.. code-block:: python

    def _check_myoptimizer_setting_format(
        self,
        algorithm: str,
        hyperparameters: list
    ) -> None:
        return None


6. aiaccel/__init__.py
===========================

.. code-block:: python

    from .common import search_algorithm_myoptimizer

    __all__ = [
        ...,
        search_algorithm_grid,
        search_algorithm_nelder_mead,
        search_algorithm_random,
        search_algorithm_sobol,
        search_algorithm_tpe,
        search_algorithm_myoptimizer,
        ...
    ]


7. start.py
===================

.. code-block:: python

    class CreationOptimizer:

        def __init__(self, config_path: str) -> None:
            config = Config(config_path)
            algorithm = config.search_algorithm.get()

            # grid search
            if algorithm.lower() == aiaccel.search_algorithm_grid:
                self.optimizer = GridSearchOptimizer

            # nelder-mead search
            elif algorithm.lower() == aiaccel.search_algorithm_nelder_mead:
                self.optimizer = NelderMeadSearchOptimizer

            # ramdom search
            elif algorithm.lower() == aiaccel.search_algorithm_random:
                self.optimizer = RandomSearchOptimizer

            # sobol search
            elif algorithm.lower() == aiaccel.search_algorithm_sobol:
                self.optimizer = SobolSearchOptimizer

            # tpe search
            elif algorithm.lower() == aiaccel.search_algorithm_tpe:
                self.optimizer = TpeSearchOptimizer

            # other (error)
            else:
                self.optimizer = None
            config = None


8. 再インストール
======================

.. code-block:: bash

    python setup.py install