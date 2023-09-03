import glob
import json
import os
import secrets
from os.path import abspath as _a
from os.path import dirname as _d
from typing import Dict, Union

ROOT = _d(_d(_d(_a(__file__))))

MAGIC_COMMENTS = {
    "default_exp": "remove-cell",
    "exporti": "remove-cell",
    "export": "hide-input",
    "exports": None,
    "hide": "remove-cell",
    "hide_input": "remove-input",
    "hide_output": "remove-output",
    "collapse_input": "hide-input",
    "collapse_output": "hide-output",
}


def load_nb(path: str) -> Dict:
    """load a jupyter notebook as dictionary

    Args:
        path: the path of the notebook to load

    Returns:
        the notebook represented as a dictionary
    """
    with open(path, "r") as file:
        nb = json.load(file)
    return nb


def repository_path(*path_parts: str, not_exist_ok: bool = False) -> str:
    """Get and validate a path in the repository

    Args:
        *path_parts: the path parts that will be joined together
            relative to the root of the repository.
        not_exist_ok: skip validation if True

    Returns:
        the absolute path of the referenced file.
    """
    if not (os.path.exists(path := os.path.join(ROOT, "docs"))):
        raise FileNotFoundError(f"docs path {path!r} not found!")
    if not (os.path.exists(path := os.path.join(ROOT, "internals"))):
        raise FileNotFoundError(f"internals path {path!r} not found!")
    if not (os.path.exists(path := os.path.join(ROOT, "sax"))):
        raise FileNotFoundError(f"sax path {path!r} not found!")

    path = os.path.join(ROOT, *path_parts)
    if not not_exist_ok and not os.path.exists(path):
        raise FileNotFoundError(f"Path {path!r} does not exist.")

    return path


def docs_path(*path_parts: str, not_exist_ok: bool = False) -> str:
    return repository_path("docs", "source", *path_parts, not_exist_ok=not_exist_ok)


def save_nb(nb: Dict, path: str) -> str:
    """save a dictionary as a jupyter notebook

    Args:
        nb: the dictionary to convert into an ipynb file
        path: the path to save the notebook under

    Returns:
        the path where the notebook was saved.
    """
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(nb, file, indent=2)
    return path


def strip_metadata(nb: Union[Dict, str]) -> Union[Dict, str]:
    path = ""
    if isinstance(nb, str):
        path = nb
        nb = load_nb(nb)
    for cell in nb["cells"]:
        if "metadata" not in cell:
            continue
        cell["metadata"] = {}
    if path:
        return save_nb(nb, path)
    return path


def iter_code_cells(nb):
    for cell in nb.get("cells", []):
        if not cell.get("cell_type", "") == "code":
            continue
        yield cell


def first_code_cell(nb):
    try:
        return next(iter_code_cells(nb))
    except StopIteration:
        return None


def get_default_exp(nb):
    first_cell = first_code_cell(nb) or {}
    first_source = first_cell.get("source", [])
    first_line = "" if not first_source else first_source[0]
    default_exp = first_line.split("default_exp")[-1].strip()
    return default_exp


def iter_function_names(source):
    for line in source:
        if not line.startswith("def "):
            continue
        if line.startswith("def _"):
            continue
        yield line.split("def ")[1].split("(")[0]


def docs_copy_nb(relpath, docsrelpath=None):
    """copy a single notebook from src to dst with modified docs metadata."""
    src = repository_path(relpath)
    dst = docs_path((docsrelpath or relpath), not_exist_ok=True)
    nb = load_nb(src)
    nb_new = {**nb}
    nb_new["cells"] = []
    module = get_default_exp(nb)

    for cell in nb.get("cells", []):
        if not cell.get("cell_type", "") == "code":
            nb_new["cells"].append(cell)
            continue

        cell_tags = cell.get("metadata", {}).get("tags", [])
        source = cell.get("source") or [""]
        line = source[0].strip()

        if not line.startswith("#"):
            nb_new["cells"].append(cell)
            continue

        keys = [k.strip() for k in line.split(" ")]
        keys = [k for k in keys if k in MAGIC_COMMENTS]
        if keys:
            del source[0]
        for key in keys:
            tag = MAGIC_COMMENTS[key]
            if tag:
                cell_tags.append(tag)
        if len(cell_tags) > 0:
            cell["metadata"]["tags"] = cell_tags

        if "remove-cell" not in cell_tags:
            for function_name in iter_function_names(source):
                extra_cell = {
                    "cell_type": "markdown",
                    "id": secrets.token_hex(8),
                    "metadata": {},
                    "source": [
                        ":::{eval-rst}\n",
                        f".. autofunction:: sax.{module}.{function_name}\n",
                        ":::\n",
                    ],
                }
                nb_new["cells"].append(extra_cell)

        cell["id"] = secrets.token_hex(8)
        nb_new["cells"].append(cell)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    save_nb(nb_new, dst)
    return dst


def list_notebooks(dir):
    return glob.glob(os.path.join(dir, "**/*.ipynb"), recursive=True)


def docs_copy_dir(relpath):
    main_src = repository_path(relpath)
    for src in list_notebooks(main_src):
        rel = os.path.relpath(src, repository_path())
        docs_copy_nb(rel)
