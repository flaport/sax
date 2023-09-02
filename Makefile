.ONESHELL:
SHELL := /bin/bash
SRC = $(wildcard internals/*.ipynb)

all: sax docs

.SILENT: docker
docker:
	-docker build . -f Dockerfile.dev -t flaport/sax:latest

sax: $(SRC)
	nbdev_build_lib

lib:
	nbdev_build_lib --fname 'internals/*.ipynb'

sync:
	nbdev_update_lib

serve:
	cd docs && bundle exec jekyll serve

.PHONY: docs
docs: lib
	python -m sax.make_docs
	find docs/_build/html -name "*.html" | xargs sed -i 's|urlpath=tree/docs|urlpath=tree|g'

run:
	find . -name "*.ipynb" | grep -v ipynb_checkpoints | xargs -I {} papermill {} {} -k sax

test:
	pytest -s tests.py -n auto --cov=sax --cov-config=.coveragerc

release: pypi conda_release
	nbdev_bump_version

conda_release:
	fastrelease_conda_package

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python -m build --sdist --wheel

clean:
	nbdev_clean_nbs
	find . -name "*.ipynb" | xargs nbstripout
	find . -name "dist" | xargs rm -rf
	find . -name "build" | xargs rm -rf
	find . -name "builds" | xargs rm -rf
	find . -name "__pycache__" | xargs rm -rf
	find . -name "*.so" | xargs rm -rf
	find . -name "*.egg-info" | xargs rm -rf
	find . -name ".ipynb_checkpoints" | xargs rm -rf
	find . -name ".pytest_cache" | xargs rm -rf

reset:
	rm -rf sax
	rm -rf docs
	git checkout -- docs
	nbdev_build_lib
	make clean
