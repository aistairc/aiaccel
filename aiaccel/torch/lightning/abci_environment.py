import logging
import os

from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment

log = logging.getLogger(__name__)


class ABCIEnvironment(ClusterEnvironment):
    """
    Environment class for ABCI.

    This class provides methods to interact with the ABCI environment,
    such as retrieving the world size, global rank, node rank, and local rank.

    Attributes:
        _world_size (int): The total number of processes in the ABCI.
        _rank (int): The global rank of the current process.
        _local_rank (int): The local rank of the current process.
        _local_size (int): The number of processes running on the local node.
        _main_address (str): The address of the main process.
        _main_port (int): The port number of the main process.
    """

    def __init__(self) -> None:
        self._world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        self._rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        self._local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        self._local_size = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])

        self._main_address = os.environ["MAIN_ADDR"]
        self._main_port = int(os.environ["MAIN_PORT"])

    @property
    def creates_processes_externally(self) -> bool:
        """
        Returns a boolean value indicating whether this environment creates processes externally.

        Returns:
            True if processes are created externally, False otherwise.
        """
        return True

    @property
    def main_address(self) -> str:
        """
        Returns the address of the main process.

        Returns:
            str: The address of the main process.
        """
        return self._main_address

    @property
    def main_port(self) -> int:
        """
        Returns the port number of the main process.

        Returns:
            int: The port number of the main process.
        """
        return self._main_port

    @staticmethod
    def detect() -> bool:
        return True

    def world_size(self) -> int:
        """
        Returns the total number of processes in the ABCI.

        Returns:
            int: The total number of processes in the ABCI.
        """
        return self._world_size

    def global_rank(self) -> int:
        """
        Returns the global rank of the current process.

        Returns:
            int: The global rank of the current process.
        """
        return self._rank

    def node_rank(self) -> int:
        """
        Returns the node rank of the current process.

        Returns:
            int: The node rank of the current process.
        """
        return self._rank // self._local_size

    def local_rank(self) -> int:
        """
        Returns the local rank of the current process.

        Returns:
            int: The local rank of the current process.
        """
        return self._local_rank

    def set_world_size(self, size: int) -> None:
        """
        Sets the world size for the environment.

        Args:
            size (int): The desired world size.

        Raises:
            ValueError: If the given size is not equal to the current world size.
        """

        if size != self.world_size():
            raise ValueError(f"`size` is expected to be {self.world_size()}, buf {size} is given.")

    def set_global_rank(self, rank: int) -> None:
        """
        Sets the global rank of the environment.

        Args:
            rank (int): The global rank to set.

        Raises:
            ValueError: If the given rank is not equal to the current global rank.
        """

        if rank != self.global_rank():
            raise ValueError(f"`rank` is expected to be {self.global_rank()}, buf {rank} is given.")

    def validate_settings(self, num_devices: int, num_nodes: int) -> None:
        """
        Validates the settings for the ABciEnvironment.

        Args:
            num_devices (int): The number of devices.
            num_nodes (int): The number of nodes.

        Raises:
            ValueError: If `num_devices` does not match the local size or
                if `num_devices * num_nodes` does not match the world size.
        """

        if num_devices != self._local_size:
            raise ValueError("`num_devices` should match ${OMPI_COMM_WORLD_LOCAL_SIZE}")

        if num_devices * num_nodes != self._world_size:
            raise ValueError("`num_devices * num_nodes` should match ${OMPI_COMM_WORLD_SIZE}")
