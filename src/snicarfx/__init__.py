import os

# force single-threaded BLAS/OpenMP execution (to prevent e.g. core
# oversubscription in HPC environments)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from importlib.metadata import version

from .core.session.config import Config
from .core.session.session import Session

__version__ = version("snicarfx")

__all__ = [
    "Config",
    "Session",
]
