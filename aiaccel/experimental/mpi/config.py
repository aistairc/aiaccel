import aiaccel
from aiaccel.config import ConfileWrapper, ConfigEntry, Config


_DEFAULT_MPI_ENVIROMENT = aiaccel.resource_type_abci
_DEFAULT_MPI_NPERNODE = 1
_DEFAULT_MPI_BAT_RT_TYPE = 'C.small'
_DEFAULT_MPI_BAT_RT_NUM = 1
_DEFAULT_MPI_BAT_H_RT = '1:00:00'
_DEFAULT_MPI_BAT_VENV_DIR = '~/mpienv'
_DEFAULT_MPI_BAT_AIACCEL_DIR = '~/aiaccel'
_DEFAULT_MPI_BAT_CONFIG_DIR = '~/aiaccel/examples/experimental/mpi/sphere'
_DEFAULT_MPI_BAT_FILE = './qsub.sh'
_DEFAULT_MPI_HOSTFILE = './hostfile'
_DEFAULT_MPI_GPU_MODE = True
_DEFAULT_MPI_BAT_MAKE_FILE = False


class MpiConfig(Config):
    """ A Class for defining the configuration of a configuration file.
    """

    def define_items(self, config: ConfileWrapper, warn: bool):
        """ Define the configuration of the configuration file
        """
        super().define_items(config, warn)

        # === MPI defalt config ===
        self.mpi_enviroment = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_MPI_ENVIROMENT,
            warning=warn,
            group="resource",
            keys=("mpi_enviroment")
        )
        self.mpi_npernode = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_MPI_NPERNODE,
            warning=warn,
            group="resource",
            keys=("mpi_npernode")
        )
        self.mpi_bat_rt_type = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_MPI_BAT_RT_TYPE,
            warning=warn,
            group="resource",
            keys=("mpi_bat_rt_type")
        )
        self.mpi_bat_rt_num = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_MPI_BAT_RT_NUM,
            warning=warn,
            group="resource",
            keys=("mpi_bat_rt_num")
        )
        self.mpi_bat_h_rt = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_MPI_BAT_H_RT,
            warning=warn,
            group="resource",
            keys=("mpi_bat_h_rt")
        )
        self.mpi_bat_venv_dir = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_MPI_BAT_VENV_DIR,
            warning=warn,
            group="resource",
            keys=("mpi_bat_venv_dir")
        )
        self.mpi_bat_aiaccel_dir = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_MPI_BAT_AIACCEL_DIR,
            warning=warn,
            group="resource",
            keys=("mpi_bat_aiaccel_dir")
        )
        self.mpi_bat_config_dir = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_MPI_BAT_CONFIG_DIR,
            warning=warn,
            group="resource",
            keys=("mpi_bat_config_dir")
        )
        self.mpi_bat_file = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_MPI_BAT_FILE,
            warning=warn,
            group="resource",
            keys=("mpi_bat_file")
        )
        self.mpi_hostfile = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_MPI_HOSTFILE,
            warning=warn,
            group="resource",
            keys=("mpi_hostfile")
        )
        self.mpi_gpu_mode = ConfigEntry(
            config=config,
            type=[bool],
            default=_DEFAULT_MPI_GPU_MODE,
            warning=warn,
            group="resource",
            keys=("mpi_gpu_mode")
        )
        self.mpi_bat_make_file = ConfigEntry(
            config=config,
            type=[bool],
            default=_DEFAULT_MPI_BAT_MAKE_FILE,
            warning=warn,
            group="resource",
            keys=("mpi_bat_make_file")
        )
