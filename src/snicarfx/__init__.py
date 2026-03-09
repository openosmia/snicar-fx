from .core.session.session import Session
from .core.session.config import Config
from importlib.metadata import version

__version__ = version("snicarfx")

__all__ = [
    "Session",
    "Confg",
]
