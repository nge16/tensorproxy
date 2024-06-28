import os
from setuptools import find_packages, setup

BASEDIR = os.path.abspath(os.path.dirname(__file__))


with open("README.md", "r", encoding="utf8", errors="ignore") as fh:
    long_description = fh.read()


def required(requirements_file):
    with open(os.path.join(BASEDIR, requirements_file), "r") as f:
        requirements = f.read().splitlines()
        return [
            pkg
            for pkg in requirements
            if pkg.strip() and not pkg.startswith("#")
        ]


setup(
    name="tensorproxy",
    version="0.0.3",
    description="Proxy toolkit for engineering systems",
    packages=find_packages(),
    install_requires=required("requirements.txt"),
    author="Alexander Belinsky",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
