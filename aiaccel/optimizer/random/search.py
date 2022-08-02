from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from typing import Optional


class Optimizer(AbstractOptimizer):
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
        self.get_dict_state()
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
            self.generated_parameter += 1

        self.create_parameter_files(returned_params)

    def _serialize(self) -> dict:
        """Serialize this module.

        Returns:
            dict: serialize data.
        """
        self.serialize_datas = {
            'generated_parameter': self.generated_parameter,
            'loop_count': self.loop_count
        }
        return super()._serialize()

    def _deserialize(self, dict_objects: dict) -> None:
        """ Deserialize this module.

        Args:
            dict_objects(dict): A dictionary including serialized objects.

        Returns:
            None
        """
        super()._deserialize(dict_objects)
