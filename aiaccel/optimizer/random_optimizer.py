from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from typing import Optional


class RandomOptimizer(AbstractOptimizer):
    """An optimizer class with a random algorithm.

    """

    def generate_parameter(self, number: Optional[int] = 1) -> None:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.

        Returns:
            None
        """

        self.get_each_state_count()
        initial_parameter = self.generate_initial_parameter()

        if initial_parameter is not None:
            self.register_ready(initial_parameter)
            self._serialize()
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

            self.num_of_generated_parameter += 1
            self.register_ready({'parameters': new_params})
            self._serialize()

    def _serialize(self) -> dict:
        """Serialize this module.

        Returns:
            dict: serialize data.
        """
        self.serialize_datas = {
            'num_of_generated_parameter': self.num_of_generated_parameter,
            'loop_count': self.loop_count
        }
        self.serialize.serialize(
            trial_id=self.trial_id.integer,
            optimization_variables=self.serialize_datas,
            native_random_state=self.get_native_random_state(),
            numpy_random_state=self.get_numpy_random_state()
        )

    def _deserialize(self, trial_id: int) -> None:
        """ Deserialize this module.

        Args:
            dict_objects(dict): A dictionary including serialized objects.

        Returns:
            None
        """
        d = self.serialize.deserialize(trial_id)
        self.deserialize_datas = d['optimization_variables']
        self.set_native_random_state(d['native_random_state'])
        self.set_numpy_random_state(d['numpy_random_state'])

        self.num_of_generated_parameter = self.deserialize_datas['num_of_generated_parameter']
        self.loop_count = self.deserialize_datas['loop_count']
