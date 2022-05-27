from aiaccel.util.filesystem import interprocess_lock_file
from pathlib import Path
import aiaccel
import fasteners
import logging
import numpy as np
import pathlib
import pickle
import random
import re
import shutil
from aiaccel.util.filesystem import retry


def resub_loop_count(s: Path) -> int:
    """Find a loop count from a file name.

    Args:
        s (Path): A path of a serialized native random file.

    Returns:
        int: A loop count.
    """
    return int(re.sub("\\D", "", s.name))


def deserialize_state(
    dict_state: Path, class_name: str, dict_lock: Path
) -> int:

    """Deserialize a state of an execution from a state directory.

    Args:
        dict_state (Path): A path of a state directory.
        class_name (str): A class name of caller module.
        dict_lock (Path): A directory to store lock files.

    Returns:
        int: A loop count.
    """
    class_str = aiaccel.get_module_type_from_class_name(class_name)

    with fasteners.InterProcessLock(
        interprocess_lock_file(dict_state, dict_lock)
    ):
        native_pattern = '{}_{}_*.{}'.format(
            class_str, aiaccel.file_native_random, aiaccel.extension_pickle
        )
        numpy_pattern = '{}_{}_*.{}'.format(
            class_str, aiaccel.file_numpy_random, aiaccel.extension_pickle
        )
        native_random_files = sorted(dict_state.glob(native_pattern))
        numpy_random_files = sorted(dict_state.glob(numpy_pattern))
        logger = None

        if class_str == aiaccel.module_type_optimizer:
            logger = logging.getLogger('root.optimizer.random')
        if class_str == aiaccel.module_type_scheduler:
            logger = logging.getLogger('root.scheduler.random')
            # Recover the hp directory
            hpdict_pattern = '{}_*'.format(aiaccel.dict_hp)
            hpdict_files = sorted(dict_state.glob(hpdict_pattern))
            pdict = dict_state.parent
            hp_dict = pdict / aiaccel.dict_hp

            if hp_dict.is_dir():
                shutil.rmtree(hp_dict)

            print('\tDEBUG hpdict_files:', hpdict_files, 'hp_dict:', hp_dict)
            dsgs = dict_state.glob('*')
            for dsg in dsgs:
                print('\tDEBUG dict_state glob:', dsg)
            shutil.copytree(hpdict_files[-1], hp_dict)

        native_num = resub_loop_count(native_random_files[-1])
        numpy_num = resub_loop_count(numpy_random_files[-1])
        loop_count = min(native_num, numpy_num)

        if len(native_random_files) > 0:
            logger.info(
                'deserialize: {}'
                .format(native_random_files[loop_count - 1])
            )
            deserialize_native_random(native_random_files[loop_count - 1])

        if len(numpy_random_files) > 0:
            logger.info(
                'deserialize: {}'
                .format(numpy_random_files[loop_count - 1])
            )
            deserialize_numpy_random(numpy_random_files[loop_count - 1])

        return loop_count


def deserialize_native_random(filename: Path) -> None:
    """Deserialize a native random object.

    Args:
        filename (Path): A path of serialized random object.

    Returns:
        None

    Raises:
        FileNotFoundError: Causes when a serialized random object file is not
            found.
    """
    logger = logging.getLogger('root.scheduler.random')

    if not filename.is_file():
        logger.error(
            'Cannot find serialized random object: {}'
            .format(filename)
        )
        raise FileNotFoundError(
            'Cannot find serialized random object: {}'
            .format(filename)
        )

    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    random.setstate(obj)


def deserialize_numpy_random(filename: Path) -> None:
    """Deserialize a numpy random object.

    Args:
        filename (Path): A serialized numpy object.

    Returns:
        None

    Raises:
        FileNotFoundError: Causes when a serialized numpy random object file is
            not found.
    """
    logger = logging.getLogger('root.scheduler.random')

    if not filename.exists():
        logger.error(
            'Cannot find serialized numpy object: {}'.format(filename)
        )
        raise FileNotFoundError(
            'Cannot find serialized numpy object: {}'.format(filename)
        )

    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    np.random.set_state(obj)


def serialize_state(
    dict_state: Path, class_name: str, loop_count: int, dict_lock: Path
) -> None:
    """Serialize current state.

    Args:
        dict_state (Path): A path of a state directory.
        class_name (str): A class name of caller module.
        loop_count (int): A loop count
        dict_lock (Path): A directory to store lock files.

    Returns:
        None
    """
    class_str = aiaccel.get_module_type_from_class_name(class_name)

    with fasteners.InterProcessLock(
        interprocess_lock_file(dict_state, dict_lock)
    ):
        @retry(_MAX_NUM=60, _DELAY=1.0)
        def _copytree(_hp_dict, _copy_dict):
            shutil.copytree(_hp_dict, _copy_dict)

        if class_str == aiaccel.module_type_scheduler:
            parent = pathlib.Path(dict_state).parent
            hp_dict = parent / aiaccel.dict_hp
            copy_dict = dict_state / '{}_{:0=10}'.format(aiaccel.dict_hp, loop_count)

            if not copy_dict.exists():
                _copytree(hp_dict, copy_dict)

        serialize_native_random(
            aiaccel.get_file_random(
                dict_state, class_name, loop_count, aiaccel.file_native_random
            )
        )
        serialize_numpy_random(
            aiaccel.get_file_random(
                dict_state, class_name, loop_count, aiaccel.file_numpy_random
            )
        )


def serialize_native_random(filename: Path) -> None:
    """Serialize a current native random object.

    Args:
        filename (Path): A path to store a random object.

    Returns:
        None
    """
    obj = random.getstate()

    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def serialize_numpy_random(filename: Path) -> None:
    """Serialize a current numpy random object.

    Args:
        filename (Path): A path to store a numpy random object.

    Returns:
        None
    """
    obj = np.random.get_state()

    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
