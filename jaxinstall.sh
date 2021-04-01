#!/bin/sh

[ -z "$PLATFORM_VERSION" ] && PLATFORM_VERSION="manylinux2010_x86_64"
[ -z "$PYTHON_VERSION" ] && PYTHON_VERSION="cp$(python --version | sed 's/^.*[ ]\([0-9]\)\.\([0-9]\).*/\1\2/g')"
[ -z "$CUDA_VERSION" ] && CUDA_VERSION="nocuda"
[ -z "$JAX_VERSION" ] && JAX_VERSION="$(grep "\bjax\b" requirements.txt)"
[ -z "$JAXLIB_VERSION" ] && JAXLIB_VERSION="$(grep "jaxlib" requirements.txt)"
JAXLIB_URL=$(echo $JAXLIB_VERSION | sed 's/==/-/g')
if which nvcc > /dev/null 2>&1; then
    [ "$CUDA_VERSION" == "nocuda" ] && CUDA_VERSION="cuda$(nvcc --version | grep release | sed 's/^.*release[ ]\([0-9]*\)\.\([0-9]*\).*$/\1\2/g')"
    JAXLIB_URL="https://storage.googleapis.com/jax-releases/$CUDA_VERSION/$JAXLIB_URL+$CUDA_VERSION-$PYTHON_VERSION-none-$PLATFORM_VERSION.whl"
fi
PYTHON=$(which python)

echo "python version: $PYTHON_VERSION @ $PYTHON"
echo "platform version: $PLATFORM_VERSION"
echo "cuda version: $CUDA_VERSION"
echo "jax version: $JAX_VERSION"
echo "jaxlib version: $JAXLIB_VERSION"
echo
echo "pip install --upgrade $JAX_VERSION"
echo "pip install --upgrade $JAXLIB_URL"
echo
echo

$PYTHON -m pip install --upgrade $JAX_VERSION
$PYTHON -m pip install --upgrade $JAXLIB_URL
