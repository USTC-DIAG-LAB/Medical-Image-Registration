"""Independent NumPy/SciPy implementation of the registration core."""
from .main import main, register_points, PACKAGE_VERSION as __version__

__all__ = ["main", "register_points", "__version__"]
