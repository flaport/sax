.ONESHELL:
SHELL := /bin/bash
SRC = $(wildcard nbs/*.ipynb)

all: sax docs

.SILENT: docker
docker:
	docker build . -t sax
	git checkout -- environment.yml

sax: $(SRC)
	nbdev_build_lib

lib:
	nbdev_build_lib

sync:
	nbdev_update_lib

serve:
	cd docs && bundle exec jekyll serve

.PHONY: docs
docs:
	jupyter nbconvert --execute --inplace index.ipynb
	nbdev_build_docs

run:
	find . -name "*.ipynb" | grep -v ipynb_checkpoints | xargs -I {} papermill {} {}

test:
	nbdev_test_nbs

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