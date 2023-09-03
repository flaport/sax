FROM condaforge/mambaforge

ENV TERM=xterm
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN ln -sf /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
 && apt-get update \
 && apt-get install --no-install-recommends --yes curl rsync zstd \
 && mamba install -y pymeep

COPY pyproject.toml .
RUN pip install --upgrade pip \
 && pip install toml \
 && python -c "import toml; p = toml.load('pyproject.toml')['project']; print('\n'.join(p['dependencies'])); print('\n'.join(p['optional-dependencies']['dev']))" \
  | xargs pip install --use-deprecated=legacy-resolver \
 && rm pyproject.toml

RUN useradd sax -u 1000 -s /bin/bash
USER sax
WORKDIR /home/sax

ADD . /home/sax

CMD ["/bin/bash"]
