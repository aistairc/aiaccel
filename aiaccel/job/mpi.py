from __future__ import annotations

mpi_enable = True
try:
    import mpi4py as m4p

    mpi4py = m4p
except ImportError:
    mpi_enable = False
