from importlib.metadata import version

from .core.session.config import Config
from .core.session.session import Session

__version__ = version("snicarfx")

__all__ = [
    "Config",
    "Session",
]
