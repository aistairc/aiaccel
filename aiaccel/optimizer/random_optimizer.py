from typing import Optional, Any

from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer


class RandomOptimizer(AbstractOptimizer):
    """An optimizer class with a random algorithm.

    """

    def __init__(self, options: dict) -> None:
        super().__init__(options)
        self.storage.variable.register(
            process_name=self.options['process_name'],
            labels=[
                'self_values',
                'self_keys',
                'self'
            ]
        )

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
            self._serialize(self.trial_id.integer)
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
            self._serialize(self.trial_id.integer)

    def _serialize(self, trial_id: int) -> dict:
        """Serialize this module.
        Returns:
            dict: serialize data.
        """

        obj = self.__dict__.copy()
        del obj['storage']
        _values = list(obj.values())
        _keys = list(obj.keys())
        self.storage.variable.d['self_values'].set(trial_id, _values)
        self.storage.variable.d['self_keys'].set(trial_id, _keys)

        # from aiaccel.util.snapshot import SnapShot
        # ss = SnapShot(self.ws, self.options['process_name'])
        # ss.save(trial_id=trial_id, obj=self)

    def _deserialize(self, trial_id: int) -> None:
        """ Deserialize this module.
        Args:
            dict_objects(dict): A dictionary including serialized objects.
        Returns:
            None
        """
        # obj = self.storage.variable.d['self'].get(trial_id)
        # print(obj)
        # self.__dict__.update(obj)

        _values = self.storage.variable.d['self_values'].get(trial_id)
        _keys = self.storage.variable.d['self_keys'].get(trial_id)
        self.__dict__.update(dict(zip(_keys, _values)))

    # def __reduce__(self):
    #     return ('storage', (self.storage,))
