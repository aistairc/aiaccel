from aiaccel.optimizer.abstract import AbstractOptimizer
from sobol_seq import i4_sobol
from typing import Optional


class SobolSearchOptimizer(AbstractOptimizer):
    """An optimizer class with sobol algorithm.

    Attributes:
        generate_index (int): A number of generated hyper parameters.

    ToDo: The development of original library was stopped. It's recommended to
        be replaced with SciPy sobol module.
    """

    def __init__(self, options: dict) -> None:
        """Initial method of SobolSearchOptimizer.

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
        returned_params = []
        l_params = self.params.get_parameter_list()
        n_params = len(l_params)
        initial_parameter = self.generate_initial_parameter()

        if initial_parameter is not None:
            returned_params.append(initial_parameter)
            number -= 1

        for n in range(number):
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

            returned_params.append({'parameters': new_params})
            self.num_of_generated_parameter += 1

        self.register_new_parameters(returned_params)

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
        super()._serialize()

    def _deserialize(self, trial_id: int) -> None:
        """Deserialize this module.

        Args:
            dict_objects(dict): A dictionary including serialized objects.

        Returns:
            None
        """
        super()._deserialize(trial_id)
        self.generate_index = self.deserialize_datas['generate_index']
        self.loop_count = self.deserialize_datas['loop_count']
        self.num_of_generated_parameter = self.deserialize_datas['num_of_generated_parameter']
