from setuptools import setup, find_packages

# All package metadata (name, version, description, deps, etc.) lives in
# pyproject.toml [project]. setup.py is kept only to run find_packages() for
# package discovery of `tracktail` and its subpackages.
setup(packages=find_packages())
