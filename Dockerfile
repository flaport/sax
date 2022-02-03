FROM condaforge/mambaforge:4.11.0-0

COPY environment.yml /environment.yml
RUN sed -i "s/^[ ]*-[ ]*sax=\+.*//g" environment.yml
RUN mamba env update -n base -f /environment.yml
RUN conda run -n base python -m ipykernel install --user --name base --display-name base
RUN conda run -n base python -m ipykernel install --user --name sax --display-name sax
RUN rm -rf /environment.yml

COPY docs/nbdev_showdoc.patch /nbdev_showdoc.patch
RUN patch -R $(python -c "from nbdev import showdoc; print(showdoc.__file__)") < /nbdev_showdoc.patch
RUN rm -rf /nbdev_showdoc.patch

ADD . /sax
RUN pip install /sax
RUN rm -rf /sax
