import sys

if sys.version_info[1] > 7:
    from importlib.metadata import version  # pragma: no cover
else:
    from importlib_metadata import version  # pragma: no cover

from kiez.kiez import Kiez

__version__ = version(__package__)

__all__ = ["Kiez"]
