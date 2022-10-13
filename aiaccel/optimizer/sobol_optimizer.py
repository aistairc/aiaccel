from typing import Optional

from scipy.stats import qmc

from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer


class SobolOptimizer(AbstractOptimizer):
    """An optimizer class with sobol algorithm.

    Attributes:
        generate_index (int): A number of generated hyper parameters.

    ToDo: Make it clear to resume this optimizer with Sobol sampler. Current code resume the sampler counts with a
        number of FINISHED PARAMETER FILES. Confirm whether the current code resumes for any timings of quits.
    """

    def __init__(self, options: dict) -> None:
        """Initial method of SobolOptimizer.

        Args:
            config (str): A file name of a configuration.
        """
        super().__init__(options)
        self.generate_index = None
        self.sampler = None

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        super().pre_process()

        finished = self.storage.trial.get_finished()
        self.generate_index = len(finished)
        self.sampler = qmc.Sobol(d=len(self.params.get_parameter_list(), scramble=False))
        self.sampler.fast_forward(self.generate_index)

    def generate_parameter(self, number: Optional[int] = 1) -> None:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.

        Returns:
            None
        """
        l_params = self.params.get_parameter_list()
        n_params = len(l_params)
        initial_parameter = self.generate_initial_parameter()

        if initial_parameter is not None:
            self.register_ready(initial_parameter)
            self._serialize()
            number -= 1

        for _ in range(number):
            new_params = []
            vec = self.sampler.random()
            self.generate_index += 1

            for i in range(0, n_params):
                min_value = l_params[i].lower
                max_value = l_params[i].upper
                value = (max_value - min_value) * vec[i] + min_value
                new_param = {
                    'parameter_name': l_params[i].name,
                    'type': l_params[i].type,
                    'value': float(value)
                }
                new_params.append(new_param)

            self.num_of_generated_parameter += 1
            self.register_ready({'parameters': new_params})
            self._serialize()

    def _serialize(self) -> dict:
        """Serialize this module.

        Returns:
            dict: The serialized objects.
        """
        self.serialize_datas = {
            'num_of_generated_parameter': self.num_of_generated_parameter,
            'loop_count': self.loop_count,
            'generate_index': self.generate_index
        }
        self.serialize.serialize(
            trial_id=self.trial_id.integer,
            optimization_variables=self.serialize_datas,
            native_random_state=self.get_native_random_state(),
            numpy_random_state=self.get_numpy_random_state()
        )

    def _deserialize(self, trial_id: int) -> None:
        """Deserialize this module.

        Args:
            dict_objects(dict): A dictionary including serialized objects.

        Returns:
            None
        """
        d = self.serialize.deserialize(trial_id)
        self.deserialize_datas = d['optimization_variables']
        self.set_native_random_state(d['native_random_state'])
        self.set_numpy_random_state(d['numpy_random_state'])

        self.generate_index = self.deserialize_datas['generate_index']
        self.loop_count = self.deserialize_datas['loop_count']
        self.num_of_generated_parameter = self.deserialize_datas['num_of_generated_parameter']
