# Changelog

## 0.14.0
### Added
- Backend information to `CircuitInfo`.

### Updated
- `klujax` package.
- Documentation.

## 0.13.5
### Added
- `settings` field to netlist.

### Updated
- GitHub Actions `publish.yml` workflow.
- Dependencies.

## 0.13.4
### Added
- `forward-only` backend (#39).
    - Example to demonstrate usage of forward-only backend.
    - Bug fixes related to forward-only backend.

### Fixed
- CI configuration.
- Pinned minimum version of `numpy` in dependencies.

## 0.13.3
### Added
- Validation for the number of ports.

### Fixed
- Issues in `variability` notebook.
- Layout-aware notebook.

## 0.13.2
### Added
- `extra_netlist_cleanup` argument.

### Updated
- Docker image.
- Makefile.

## 0.13.1
### Updated
- Makefile.
- Dockerfile.

## 0.13.0
### Added
- Documentation and examples.
- Coercion of nets into connections.
- Handle for callable instances in connections.
- Tests for cleanup and refactoring.

### Deprecated
- `info` subdirectory in instance dictionary.
- `sax.nn` module.

### Updated
- Proper transition to `pydantic v2`.
- Added docstrings across functions (#34).

## 0.12.2
### Updated
- Pinned minimum version of `flax`.

## 0.12.1
### Fixed
- Minor bug fixes.

## 0.12.0
### Added
- Config setting to override component in `info['model']`.

## 0.11.4
### Updated
- Bumped `klujax`.

## 0.11.3
### Improved
- Flatten netlist functionality.

## 0.11.2
### Fixed
- Ignored ports not following the correct format in `flatten_netlist`.

## 0.11.1
### Updated
- Notebooks.
- Bug fixes from 0.11 release.

## 0.11.0
### Added
- `flatten_netlist` function.

### Improved
- Error messages.
- Analysis of dummy models to prevent dense representation during evaluation.

## 0.10.4
### Added
- Handling of complex function arguments (#28).
- Support for omitting placements in recursive netlist (#27).

### Fixed
- Issues related to `'$'` sign in netlist component values.

## 0.10.3
### Added
- Changelog (#23).

### Fixed
- Changelog formatting.
- Updated `MANIFEST.in`.

## 0.10.2
### Fixed
- Evaluation of ports rather than tracing in models.

## 0.10.1
### Added
- `bump2version` as a development dependency.
- Test folder exposure as artifact.

### Fixed
- CI workflows, including Sphinx and Binderhub configurations.

## 0.10.0
### Added
- Type annotations and expanded documentation.
- Jupyter-book based documentation.

### Updated
- Notebooks.
- Improved multimode operations.

## 0.9.2
### Added
- Helper functions for component-to-instance mapping.

## 0.9.0
### Fixed
- Compatibility with both `pydantic` v1 and v2.

## 0.8.8
### Fixed
- Validation in `get_required_circuit_models` function.

### Added
- `get_required_circuit_models` function.

## 0.8.6
### Updated
- GitHub workflows for pypi publishing.

## 0.8.5
### Fixed
- Various minor cleanups in notebooks.

## 0.8.4
### Fixed
- Consolidation in multimode to singlemode transition.

## 0.8.3
### Added
- Support for circuit dependency DAGs.

## 0.8.2
### Fixed
- Handling of `None` settings in instances.

## 0.8.1
### Updated
- Notebooks and examples with new API.

### Added
- Support for dict-based circuits.

## 0.8.0
### Added
- Full type support for SAX.
- New documentation structure.

## 0.7.3
### Updated
- Layout-aware notebook title.
- Project logo.

## 0.7.2
### Fixed
- `circuit_from_gdsfactory` function.

## 0.7.1
### Updated
- Minimal version of SAX availability without JAX.

## 0.6.0 - 0.7.0
- Initial commits with foundational code, refactoring, and early versioning.
