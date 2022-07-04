from aiaccel.optimizer.abstract import AbstractOptimizer
from typing import Optional


class RandomSearchOptimizer(AbstractOptimizer):
    """An optimizer class with a random algorithm.

    """

    def generate_parameter(self, number: Optional[int] = 1) -> None:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.

        Returns:
            None
        """

        returned_params = []
        self.get_each_state_count()
        initial_parameter = self.generate_initial_parameter()

        if initial_parameter is not None:
            returned_params.append(initial_parameter)
            number -= 1

        for i in range(number):
            new_params = []
            sample = self.params.sample()

            for s in sample:
                new_param = {
                    'parameter_name': s['name'],
                    'type': s['type'],
                    'value': s['value']
                }
                new_params.append(new_param)

            returned_params.append({'parameters': new_params})
            self.num_of_generated_parameter += 1

        self.register_new_parameters(returned_params)

    def _serialize(self) -> dict:
        """Serialize this module.

        Returns:
            dict: serialize data.
        """
        self.serialize_datas = {
            'num_of_generated_parameter': self.num_of_generated_parameter,
            'loop_count': self.loop_count
        }
        super()._serialize()

    def _deserialize(self, trial_id: int) -> None:
        """ Deserialize this module.

        Args:
            dict_objects(dict): A dictionary including serialized objects.

        Returns:
            None
        """
        super()._deserialize(trial_id)
        self.num_of_generated_parameter = self.deserialize_datas['num_of_generated_parameter']
        self.loop_count = self.deserialize_datas['loop_count']
