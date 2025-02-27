dev:
	uv venv --python 3.12
	uv sync --all-extras

build:
	uv run python -m build --sdist --wheel

docker:
	docker build . -t flaport/sax:latest -f Dockerfile.dev
	docker build . -t flaport/sax:0.14.2 -f Dockerfile.dev

pre-commit:
	pre-commit install

nbrun:
	find examples -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" -not -path "./.venv/*" | xargs parallel uv run papermill {} {} -k python3 --cwd examples ':::'
	find internals -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" -not -path "./.venv/*" | xargs parallel uv run papermill {} {} -k python3 --cwd internals ':::'

dockerpush:
	docker push flaport/sax:latest
	docker push flaport/sax:0.14.2

.PHONY: docs
docs:
	cd docs/source/ && make && cd .. && make html

clean:
	rm -rf .venv
	find . -name "modes" | xargs rm -rf
	find . -name "dist" | xargs rm -rf
	find . -name "build" | xargs rm -rf
	find . -name "builds" | xargs rm -rf
	find . -name "__pycache__" | xargs rm -rf
	find . -name "*.so" | xargs rm -rf
	find . -name ".ipynb_checkpoints" | xargs rm -rf
	find . -name ".pytest_cache" | xargs rm -rf
	find . -name ".mypy_cache" | xargs rm -rf
