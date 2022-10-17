from typing import Optional

from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer


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

        generated_params = []

        if initial_parameter is not None:
            generated_params.append(initial_parameter)
            number -= 1

        for _ in range(number):
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
            generated_params.append({'parameters': new_params})

        return generated_params
