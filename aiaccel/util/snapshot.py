import shutil
import pathlib
import copy
from typing import Union
from typing import Any
from fasteners import InterProcessLock
from aiaccel.util import filesystem as fs
from aiaccel.util.retry import retry
import pickle


class _snapshot:
    def __init__(self, dir_path: pathlib.Path) -> None:
        self.dir_path = dir_path

    def get_text(self) -> dict:
        files = list(self.dir_path.glob("*.hp"))
        data = {}
        for file in files:
            data[str(file)] = fs.load_yaml(str(file.resolve()))
        return data


class SnapShot:
    """
        About:
            There are two main functions of this class.
            (1) save the contents of the hp directory in yaml format.
            (2) restore the hp directory from yaml.

        Example:
            ss = SnapShot("config.json")
            ss.save(trial_id)
            ss.load(trial_id)  # Replace the contents of the directory

        Attributes:
            process_name (str)  : A process name (master, optimizer, scheduler)
            ws (pathlib.Path)      : A path of workspace directory.
            ws_alive (pathlib.Path): A path of alive directory.
            ws_hp (pathlib.Path)   : A path of hp directory.
            ws_state (pathlib.Path): A path of state directory.
            lock_file (dict)       : A set of lock file path and lock name.
            hp_dirs (list)         : directory names in hp.
            snap_shot (dict)       : snapshot data of any directory.
            ss_hp (dict)                : snapshot data of hp directory.
            ss_loop_count (dict)        : loop count of master, optimizer,
                                            and scheduler.
            random_state_native (tuple) : nataive random state
            random_state_numpy (tuple)  : numpy random state
            optimization_variables (dict): store serialized memory objects of
                                            the process.
    """

    def __init__(
        self,
        workspace: pathlib.Path,
        # process_name: str   # Assume 'scheduler' or 'optimizer' or 'master'
    ) -> None:

        # self.process_name = process_name

        self.ws = workspace.resolve()
        self.ws_alive = self.ws / 'alive'
        self.ws_hp = self.ws / 'hp'
        self.ws_state = self.ws / 'state'
        self.lock_file = {
            'hp': str(self.ws / 'lock' / 'hp'),
            'ready': str(self.ws / 'lock' / 'ready'),
            'running': str(self.ws / 'lock' / 'running'),
            'finished': str(self.ws / 'lock' / 'finished'),
            'state': str(self.ws / 'lock' / 'state'),
            'serialize': str(self.ws / 'lock' / 'serialize'),
            'deserialize': str(self.ws / 'lock' / 'deserialize'),
            'num_gem_param_txt': str(self.ws / 'lock' / 'count'),
        }
        self.hp_dirs = [
            'ready',
            'running',
            'finished'
        ]
        self.snap_shot = {}
        for hp_dir in self.hp_dirs:
            self.snap_shot[hp_dir] = _snapshot(self.ws / 'hp' / hp_dir)
        self.ss_hp = {
            'ready': {},
            'running': {},
            'finished': {}
        }
        self.ss_loop_count = {
            'master': 0,
            'optimizer': 0,
            'scheduler': 0
        }
        self.random_state_native = None
        self.random_state_numpy = None
        self.optimization_variables = None

    def register_new_lock_file(self, label: str) -> None:
        """ Register a new lock file

        Args:
            label (str): lock name

        Returns:
            None
        """
        self.lock_file[label] = str(self.ws / 'lock' / label)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def shot(self) -> None:
        """ Get yaml files data in hp directory.

        Returns:
            None

        Note:
            If any error occurs, it will keep retrying for 60 seconds.
        """
        self.ss_hp = {
            'ready': {},
            'running': {},
            'finished': {}
        }

        # Get all hp-files in hp folders as text(yaml)
        for hp_dir in self.hp_dirs:
            with InterProcessLock(self.lock_file[hp_dir]):
                self.ss_hp[hp_dir] = self.snap_shot[hp_dir].get_text()

    def save(
        self,
        trial_id: int,
        process_name: str,
        optimization_variables: dict,
        native_random_object: tuple,
        numpy_random_object: tuple
    ) -> None:
        """ Save any data.
        Args:
            trial_id (int)     : Current trial number.
            native_random_object (tuple): Native random state.
            numpy_random_object (tuple) : Numpy random state.
            optimization_variables (dict): Intermediate data of optimize.

        Result:
            None
        """
        with InterProcessLock(self.lock_file['serialize']):
            self._save(
                trial_id,
                process_name,
                optimization_variables,
                native_random_object,
                numpy_random_object,
            )

    def _save(
        self,
        trial_id: int,
        process_name: str,
        optimization_variables: dict,
        native_random_object: tuple,
        numpy_random_object: tuple
    ) -> None:
        """
        Args:
            trial_id (int)     : Current trial number.
            optimization_variables (dict): Intermediate data of optimize.
            native_random_object (tuple): Native random state.
            numpy_random_object (tuple) : Numpy random state.

        Note:
            Save the contents of the Hp directory in yaml format.
        """
        # If self.process_name == 'scheduler':
        if process_name == 'master':
            self.shot()

        base_dir_path = self.ws / 'state' / str(trial_id)
        native_dir_path = base_dir_path / 'native_random'
        numpy_dir_path = base_dir_path / 'numpy_random'

        with InterProcessLock(self.lock_file['state']):
            if not base_dir_path.exists():
                base_dir_path.mkdir()

        # (loop_counter)
        loop_count = optimization_variables['loop_count']
        self._update_inner_loop_count(
            trial_id,
            process_name,
            loop_count
        )

        # (hp)
        save_file_path = base_dir_path / 'hp.yaml'
        if not save_file_path.exists():
            # Save only once per trial.
            with InterProcessLock(self.lock_file['state']):
                if process_name == 'master':
                    data = self.create_data()
                    self.hp_status_recoard_yml(save_file_path, data)
                self.write_inner_loop_count(trial_id)

        # native random file
        save_path = native_dir_path / (process_name + '.pickle')
        if not native_dir_path.exists():
            native_dir_path.mkdir()
        if not save_path.exists():
            # Save only once per trial.
            self._serialize_any_object(save_path, native_random_object)

        # numpy random file
        save_path = numpy_dir_path / (process_name + '.pickle')
        if not numpy_dir_path.exists():
            numpy_dir_path.mkdir()
        if not save_path.exists():
            # Save only once per trial.
            self._serialize_any_object(save_path, numpy_random_object)

        # Optuna DB file copy
        if process_name == 'master':
            src = self.ws / 'optuna_study.db'
            dst = base_dir_path / 'optuna_study.db'
            if src.exists():
                shutil.copy(src, dst)

        # Generete count file(count.txt) copy
        if process_name == 'master':
            src = self.ws_hp / 'count.txt'
            dst = base_dir_path / 'count.txt'
            if src.exists():
                shutil.copy(src, dst)

        # Write process memory objects
        pmo_file_path = base_dir_path / f'{process_name}.yaml'
        if not pmo_file_path.exists():
            with InterProcessLock(self.lock_file['state']):
                fs.create_yaml(pmo_file_path, optimization_variables)

    def _update_inner_loop_count(
        self,
        trial_id: int,
        process_name: str,
        inner_loop_count: int
    ) -> None:
        """ Update inner loop counter value.

        Args:
            curr_trial_numbe (int): Current trial number.
            inner_loop_count (int): Internal control variables.(not used)

        Returns:
            None

        Note:
            * The "inner loop counter" is not used for internal control.
            * See "module.py" for the meaning of "inner loop"
        """
        base_dir_path = self.ws / 'state' / str(trial_id)
        file_path = base_dir_path / 'hp.yaml'
        if not file_path.exists():
            self.ss_loop_count[process_name] = inner_loop_count
        else:
            self.ss_loop_count = fs.load_yaml(file_path)['loop_count']
            self.ss_loop_count[process_name] = inner_loop_count

    def write_inner_loop_count(self, trial_id: int) -> None:
        """ Write the inner loop count value to text.

        Args:
            curr_trial_numbe (int): Current trial number.
            inner_loop_count (int): Internal control variables.(not used)

        Returns:
            None
        """
        base_dir_path = self.ws / 'state' / str(trial_id)
        file_path = base_dir_path / 'hp.yaml'
        if not file_path.exists():
            return
        tmp = fs.load_yaml(file_path)
        tmp['loop_count'] = copy.deepcopy(self.ss_loop_count)
        fs.create_yaml(file_path, tmp)

    def create_data(self) -> dict:
        """ Creating saved data.

        Returns:
            data (dict)
        """
        data = {}
        data['hp'] = copy.deepcopy(self.ss_hp)
        data['loop_count'] = copy.deepcopy(self.ss_loop_count)
        return data

    def hp_status_recoard_yml(
        self,
        save_file_path: pathlib.Path,
        data: dict
    ) -> None:
        """ dict data to yaml file.

        Args:
            save_file_path (pathlib.Path): The yaml file path.
            data (dict): Subject to save.

        Returns:
            None
        """
        fs.create_yaml(save_file_path, data)

    def _serialize_any_object(
        self,
        save_path: pathlib.Path,
        obj: Any
    ) -> None:
        """ Serialize anything.

        Args:
            save_file_path (pathlib.Path): The yaml file path.
            obj (any): Subject to save.

        Returns:
            None
        """
        with open(save_path, 'wb') as f:
            pickle.dump(obj, f)

    def load(self, trial_id: int, process_name: str) -> bool:
        """ Load any trial data.

        Args:
            trial_id (int): target trial number.

        Returns:
            None
        """
        ss_files = list(self.ws_state.glob("*"))
        if len(ss_files) == 0:
            return False
        # ss_files.sort()
        with InterProcessLock(self.lock_file['deserialize']):
            return self._load(trial_id, process_name)

    def _load(self, trial_id: int, process_name: str) -> bool:
        """ Serialize anything.

        Args:
            save_file_path (pathlib.Path): The yaml file path.
            obj (any): Subject to save.

        Returns:
            bool

        Note:
            Replace the contents of the directory
        """
        base_dir_path = self.ws / 'state' / str(trial_id)
        native_dir_path = base_dir_path / 'native_random'
        numpy_dir_path = base_dir_path / 'numpy_random'

        if not base_dir_path.exists():
            return False

        if not native_dir_path.exists():
            return False

        if not numpy_dir_path.exists():
            return False

        # Load hp-file
        if process_name == 'master':
            filename = 'hp.yaml'
            load_file_path = base_dir_path / filename
            with InterProcessLock(self.lock_file['state']):
                if not load_file_path.exists():
                    return False
                load_data = fs.load_yaml(load_file_path)
                self.ss_hp = load_data['hp']

            # Replace hp dir
            with InterProcessLock(self.lock_file['hp']):
                self.replace_hp_dir()

        # native random file
        load_path = native_dir_path / (process_name + '.pickle')
        with InterProcessLock(self.lock_file['state']):
            if not load_path.exists():
                return False
            with open(load_path, 'rb') as f:
                self.random_state_native = pickle.load(f)

        # numpy random file
        load_path = numpy_dir_path / (process_name + '.pickle')
        with InterProcessLock(self.lock_file['state']):
            if not load_path.exists():
                return False
            with open(load_path, 'rb') as f:
                self.random_state_numpy = pickle.load(f)

        # Optuna DB file copy
        if process_name == 'master':
            src = base_dir_path / 'optuna_study.db'
            dst = self.ws / 'optuna_study.db'
            if dst.exists():
                dst.unlink()

            with InterProcessLock(self.lock_file['state']):
                if src.exists():
                    shutil.copy(src, dst)

        if process_name == 'master':
            src = base_dir_path / 'count.txt'
            dst = self.ws / 'hp' / 'count.txt'
            if dst.exists():
                dst.unlink()

            with InterProcessLock(self.lock_file['state']):
                if src.exists():
                    shutil.copy(src, dst)

        # Load process memory objects
        pmo_file_path = base_dir_path / f'{process_name}.yaml'
        with InterProcessLock(self.lock_file['state']):
            if not pmo_file_path.exists():
                return False
            self.optimization_variables = fs.load_yaml(pmo_file_path)

        print(f"loaded {process_name}")
        return True

    def get_base_dir_path(self, trial_id: int) -> pathlib.Path:
        """ Get the snapshot data save directory for each trial.

        Args:
            trial_id (int): Current trial number

        Returns:
            pathlib.Path: save directory path
        """
        return (self.ws / 'state' / str(trial_id))

    def get_inner_loop_counter(self, trial_id: int) -> Union[int, None]:
        """ Get the value of the inner loop counter for a given
            trial from a yaml file.

        Args:
            trial_id (int): The any trial number.

        Returns:
            Int value if target file exists, or None if it does not exist.
        """
        base_dir_path = self.ws / 'state' / str(trial_id)
        if not base_dir_path.exists():
            return None

        # Load hp-file
        load_file_path = base_dir_path / 'hp.yaml'
        if not load_file_path.exists():
            return None

        with InterProcessLock(self.lock_file['state']):
            load_data = fs.load_yaml(load_file_path)
            inner_loop_counter = load_data['loop_count']
        return inner_loop_counter

    def hp_clean(self):
        """ clean hp directory.

        Unlink all yaml file in 'ready' 'running'
        and 'finished' directries in hp directory.

        Returns:
            None
        """
        for hp_dir in self.hp_dirs:
            fp = self.ws / 'hp' / hp_dir
            files = fp.glob("*.*")
            for file in files:
                file.unlink()

    def replace_hp_dir(self):
        """ Replace hp directories.

        Replace the 'ready' 'running', and 'finished' directories in the
        hp directory with the contents of 'self.ss_hp'

        Returns:
            None
        """
        self.hp_clean()
        for hp_dir in self.hp_dirs:
            with InterProcessLock(self.lock_file[hp_dir]):
                files = list(self.ss_hp[hp_dir].keys())
                for file in files:
                    fs.create_yaml(file, self.ss_hp[hp_dir][file])

    def delete(self, trial_id: int):
        base_dir_path = self.ws / 'state' / str(trial_id)
        if base_dir_path.exists():
            shutil.rmtree(base_dir_path)
