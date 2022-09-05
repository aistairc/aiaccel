from pathlib import Path
from aiaccel.config import Config
from typing import Dict
from aiaccel.storage.storage import Storage


class Serializer:
    def __init__(self, config: Config, process_name: str, options):
        self.config = config
        self.ws = Path(self.config.workspace.get()).resolve()
        self.storage = Storage(
            self.ws,
            fsmode=options['fs'],
            config_path=options['config']
        )
        self.process_name = process_name.lower()
        if self.process_name not in ['master', 'optimizer', 'scheduler']:
            raise

    def _get_initialized_serialize_data(self):
        return {
            'num_of_generated_parameter': None,
            'loop_count': None,
            'ready_params': None,
            'generate_index': None,
            'parameter_pool': None,
            'study': None,
            'nelder_mead': None,
            'order': None,
            'all_trial_id': None
        }

    def serialize(
        self,
        trial_id: int,
        optimization_variables: Dict,
        native_random_state: tuple,
        numpy_random_state: tuple
    ):
        # optimization_variables
        serialize_data = self._get_initialized_serialize_data()
        for key in optimization_variables.keys():
            if key in serialize_data.keys():
                serialize_data[key] = optimization_variables[key]

        self.storage.serializer.set_any_trial_serialize(
            trial_id=trial_id,
            optimization_variable=serialize_data,
            process_name=self.process_name,
            native_random_state=native_random_state,
            numpy_random_state=numpy_random_state
        )

        # # random_state
        # self.storage.random_state.set_random_state(
        #     trial_id=trial_id,
        #     process_name=self.process_name,
        #     native_random_state=native_random_state,
        #     numpy_random_state=numpy_random_state
        # )

    def deserialize(self, trial_id: int) -> Dict:
        # optimization_variables, random_state
        optimization_variables, native_rnd_state, numpy_rnd_state = (
            self.storage.serializer.get_any_trial_serialize(
                trial_id=trial_id,
                process_name=self.process_name
            )
        )

        if optimization_variables is None:
            optimization_variables = self._get_initialized_serialize_data()

        # optimization_variables = (
        #     self.storage.serializer.get_any_trial_serialize(
        #         trial_id=trial_id,
        #         process_name=self.process_name
        #     )
        # )
        # if optimization_variables is None:
        #     optimization_variables = self._get_initialized_serialize_data()

        # # random_state
        # native_rnd_state, numpy_rnd_state = (
        #     self.storage.random_state.get_random_state(
        #         trial_id=trial_id,
        #         process_name=self.process_name
        #     )
        # )

        data = {
            "optimization_variables": optimization_variables,
            "native_random_state": native_rnd_state,
            "numpy_random_state": numpy_rnd_state
        }

        return data
