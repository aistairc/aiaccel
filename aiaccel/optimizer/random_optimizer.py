from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer


class RandomOptimizer(AbstractOptimizer):
    """An optimizer class with a random algorithm.

    """

    def generate_parameter(self) -> None:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.

        Returns:
            None
        """

        new_params = []
        sample = self.params.sample()

        for s in sample:
            new_param = {
                'parameter_name': s['name'],
                'type': s['type'],
                'value': s['value']
            }
            new_params.append(new_param)

        return new_params

    def generate_initial_parameter(self) -> list:
        if super().generate_initial_parameter() is not None:
            self.logger.warning(
                "Initial values cannot be specified for random."
                "The set initial value has been invalidated."
            )
        return self.generate_parameter()
