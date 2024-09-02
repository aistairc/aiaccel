import logging
import os

from lightning.fabric.plugins.environments import ClusterEnvironment

log = logging.getLogger(__name__)


class ABCIEnvironment(ClusterEnvironment):
    def __init__(self) -> None:
        self._world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        self._rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        self._local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        self._local_size = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])

        self._main_address = os.environ["MAIN_ADDR"]
        self._main_port = int(os.environ["MAIN_PORT"])

    @property
    def creates_processes_externally(self) -> bool:
        return True

    @property
    def main_address(self) -> str:
        return self._main_address

    @property
    def main_port(self) -> int:
        return self._main_port
    
    @staticmethod
    def detect() -> bool:
        return True

    def world_size(self) -> int:
        return self._world_size

    def global_rank(self) -> int:
        return self._rank

    def node_rank(self) -> int:
        return self._rank // self._local_size

    def local_rank(self) -> int:
        return self._local_rank

    def set_world_size(self, size: int) -> None:
        log.debug("ABCIEnvironment.set_world_size was called. Ignored.")

    def set_global_rank(self, rank: int) -> None:
        log.debug("ABCIEnvironment.set_global_rank was called. Ignored.")

    def validate_settings(self, num_devices: int, num_nodes: int) -> None:
        if num_devices != self._local_size:
            raise ValueError("`num_devices` should match ${OMPI_COMM_WORLD_LOCAL_SIZE}")

        if num_devices * num_nodes != self._world_size:
            raise ValueError("`num_devices * num_nodes` should match ${OMPI_COMM_WORLD_SIZE}")
