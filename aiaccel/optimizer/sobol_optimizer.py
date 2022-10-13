from typing import Optional

from sobol_seq import i4_sobol

from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer


class SobolOptimizer(AbstractOptimizer):
    """An optimizer class with sobol algorithm.

    Attributes:
        generate_index (int): A number of generated hyper parameters.

    ToDo: The development of original library was stopped. It's recommended to
        be replaced with SciPy sobol module.
    """

    def __init__(self, options: dict) -> None:
        """Initial method of SobolOptimizer.

        Args:
            config (str): A file name of a configuration.
        """
        super().__init__(options)
        self.generate_index = None

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        super().pre_process()

        finished = self.storage.trial.get_finished()
        self.generate_index = len(finished)

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
            self._serialize(self.trial_id.integer)
            number -= 1

        for _ in range(number):
            new_params = []
            vec, seed = i4_sobol(n_params, self.generate_index)
            self.generate_index = seed

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
            self._serialize(self.trial_id.integer)
