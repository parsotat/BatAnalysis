"""
steps to upload a distribution to PyPi is at: https://stackoverflow.com/questions/1471994/what-is-setup-py

"""
from setuptools import setup, find_packages
import os


# also need to include all subdirs in the data directory
def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


try:
    with open("README.md", "r") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ""

with open("requirements.txt") as f:
    required = f.read().splitlines()

extra_files = package_files("./batanalysis/data")

# get the version number from the _version file
with open("batanalysis/_version.py") as f:
    file_info = f.read().splitlines()
version = file_info[-1].split("=")[-1].split('"')[1]

setup(
    name="BatAnalysis",
    version=version,
    packages=["batanalysis"],
    url="https://github.com/parsotat/BatAnalysis",
    license="BSD-3-Clause",
    author="Tyler Parsotan and Sibasish Laha",
    author_email="tyler.parsotan@nasa.gov",
    description="Routines for analyzing data from BAT on the Neil Gehrels Swift Observatory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.9",
    install_requires=required,
    package_data={"": extra_files},
    include_package_data=True,
)
