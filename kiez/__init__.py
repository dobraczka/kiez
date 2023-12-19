from importlib.metadata import version  # pragma: no cover

from kiez.kiez import Kiez
from kiez.new_kiez import NewKiez

__version__ = version(__package__)

__all__ = ["Kiez", "NewKiez"]
