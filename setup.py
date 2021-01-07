# imports
import sys
import sax
import warnings
import setuptools
import subprocess

# install sax:
setuptools.setup(
    name=sax.__name__,
    version=sax.__version__,
    description=sax.__doc__,
    long_description=open("README.md").read(),
    author=sax.__author__,
    author_email="floris.laporte@gmail.com",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License"
    ],
)
