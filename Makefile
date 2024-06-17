build:
	python -m build --sdist --wheel

docker:
	docker build . -t flaport/sax:latest

pre-commit:
	pre-commit install
	git config filter.nbstripout.extrakeys 'metadata.papermill'

nbrun:
	find . -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" | xargs parallel -j `nproc --all` papermill {} {} -k python3 :::
	rm -rf modes

dockerpush:
	docker push flaport/sax:latest

.PHONY: docs
docs:
	sphinx-apidoc --force --no-toc --no-headings --implicit-namespaces --module-first --maxdepth 1 --output-dir docs/source sax
	cd docs && make html

clean:
	find . -name "modes" | xargs rm -rf
	find . -name "dist" | xargs rm -rf
	find . -name "build" | xargs rm -rf
	find . -name "builds" | xargs rm -rf
	find . -name "__pycache__" | xargs rm -rf
	find . -name "*.so" | xargs rm -rf
	find . -name ".ipynb_checkpoints" | xargs rm -rf
	find . -name ".pytest_cache" | xargs rm -rf
	find . -name ".mypy_cache" | xargs rm -rf
