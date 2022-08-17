FROM debian:bullseye
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN ln -sf /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
    && apt-get update \
    && apt-get install --no-install-recommends --yes \
    build-essential \
    ca-certificates \
    curl \
    bzip2 \
    git \
    htop \
    neovim \
    openssh-client \
    patch \
    rsync \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ADD . /home/sax

RUN useradd sax -u 1000 -s /bin/bash \
    && mkdir -p "$(dirname ${CONDA_DIR})" \
    && chown -R sax:sax /home/sax "$(dirname ${CONDA_DIR})"

USER sax
WORKDIR /home/sax

RUN curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh --output /tmp/miniforge.sh --silent \
    && /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && sed -i "s/^[ ]*-[ ]*sax=\+.*//g" /home/sax/environment.yml \
    && mamba env update -n base -f /home/sax/environment.yml \
    && conda install pymeep=\*=mpi_mpich_\* gdsfactory suitesparse pybind11 \
    && pip install klujax klayout /home/sax[dev] \
    && conda run -n base python -m ipykernel install --user --name base --display-name base \
    && conda run -n base python -m ipykernel install --user --name sax --display-name sax
