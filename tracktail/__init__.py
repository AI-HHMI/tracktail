"""tracktail: multi-view 3D point tracking without depth input.

This makes `tracktail` a regular package so it is discovered by
`setuptools.find_packages()` and importable from an (editable) install regardless of
the current working directory. Subpackages: `tracktail.model`, `tracktail.datasets`.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("tracktail")
except PackageNotFoundError:  # running from a source checkout, not installed
    __version__ = "0.0.0+unknown"
