# imports
import sys
import phax
import warnings
import setuptools
import subprocess

# jax/jaxlib installation
JAXLIB_URL='https://storage.googleapis.com/jax-releases'
PYTHON_VERSION="cp{}{}".format(*sys.version_info[:2])
if PYTHON_VERSION not in ("cp36", "cp37", "cp38"):
    raise RuntimeError(
        "phax only supports python 3.6, 3.7 and 3.8. "
        "Your version: {}.{}".format(*sys.version_info[:2])
    )
CUDA_VERSION="nocuda"
try:
    CUDA_VERSION=subprocess.check_output(["nvcc", "--version"]).decode()
    CUDA_VERSION=CUDA_VERSION.split("release ")[-1].split(",")[0].replace(".","")
    CUDA_VERSION=f"cuda{CUDA_VERSION}"
except subprocess.CalledProcessError:
    warnings.warn("no CUDA found. Falling back to phax CPU install.")
PLATFORM="manylinux2010_x86_64"
if not sys.platform == "linux":
    raise RuntimeError(f"phax is only supported on linux")
JAXLIB=f"{JAXLIB_URL}/{CUDA_VERSION}/jaxlib-0.1.55-{PYTHON_VERSION}-none-{PLATFORM}.whl"
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", JAXLIB])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "jax"])
except subprocess.CalledProcessError:
    raise RuntimeError(f"failed to install {JAXLIB}.\nInstallation of phax aborted.")

# install all other requirements
subprocess.call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# install phax:
setuptools.setup(
    name=phax.__name__,
    version=phax.__version__,
    description=phax.__doc__,
    long_description=phax.__doc__,
    author=phax.__author__,
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
