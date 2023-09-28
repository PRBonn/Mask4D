### To not hack your paths
# install via `pip3 install -U -e .` in this dir
# https://stackoverflow.com/questions/6323860/sibling-package-imports/50193944#50193944

from setuptools import find_packages, setup

pkg = "mask_4d"
setup(name=pkg, version="1.0", packages=find_packages())
