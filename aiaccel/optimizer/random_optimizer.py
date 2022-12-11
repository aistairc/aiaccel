from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer


class RandomOptimizer(AbstractOptimizer):
    """An optimizer class with a random algorithm.

    """

    def generate_parameter(self) -> list:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.

        Returns:
            list: A list of created parameters.
        """

        new_params = []
        sample = self.params.sample(rng=self._rng)

        for s in sample:
            new_param = {
                'parameter_name': s['name'],
                'type': s['type'],
                'value': s['value']
            }
            new_params.append(new_param)

        return new_params
