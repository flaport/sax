# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: sax
#     language: python
#     name: sax
# ---

# +
# default_exp make_docs
# -

# # SAX make docs
# > CLI: build jupyter-books based docs

# +
# exporti
from __future__ import annotations

import glob
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
from typing import Dict, Union

from fastcore.imports import IN_IPYTHON
# -

# export
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

# export
if IN_IPYTHON:
    ROOT = os.path.abspath('..')
else:
    ROOT = os.path.abspath('.')


# export
def load_nb(path: str) -> Dict:
    """ load a jupyter notebook as dictionary

    Args:
        path: the path of the notebook to load

    Returns:
        the notebook represented as a dictionary
    """
    with open(path, "r") as file:
        nb = json.load(file)
    return nb


# export
def repository_path(*path_parts: str, not_exist_ok: bool=False) -> str:
    """ Get and validate a path in the modelbuild repository

    Args:
        *path_parts: the path parts that will be joined together
            relative to the root of the repository.
        not_exist_ok: skip validation if True

    Returns:
        the absolute path of the referenced file.
    """
    if not (os.path.exists(path:=os.path.join(ROOT, "docs"))):
        raise FileNotFoundError(f"docs path {path!r} not found!")
    if not (os.path.exists(path:=os.path.join(ROOT, "nbs"))):
        raise FileNotFoundError(f"nbs path {path!r} not found!")
    if not (os.path.exists(path:=os.path.join(ROOT, "sax"))):
        raise FileNotFoundError(f"sax path {path!r} not found!")

    path = os.path.join(ROOT, *path_parts)
    if not not_exist_ok and not os.path.exists(path):
        raise FileNotFoundError(f"Path {path!r} does not exist.")

    return path


# export
def docs_path(*path_parts: str, not_exist_ok: bool=False) -> str:
    return repository_path('docs', *path_parts, not_exist_ok=not_exist_ok)


# export
def save_nb(nb: Dict, path: str) -> str:
    """ save a dictionary as a jupyter notebook

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


# export
def strip_metadata(nb: Union[Dict, str]) -> Union[Dict,str]:
    path = ''
    if isinstance(nb, str):
        path = nb
        nb = load_nb(nb)
    for cell in nb['cells']:
        if not 'metadata' in cell:
            continue
        cell['metadata'] = {}
    if path:
        return save_nb(nb, path)
    return path


# exporti
def iter_code_cells(nb):
    for cell in nb.get('cells', []):
        if not cell.get("cell_type", "") == "code":
            continue
        yield cell


# exporti
def first_code_cell(nb):
    try:
        return next(iter_code_cells(nb))
    except StopIteration:
        return None


# exporti
def get_default_exp(nb):
    first_cell = first_code_cell(nb) or {}
    first_source = first_cell.get('source', [])
    first_line = "" if not first_source else first_source[0]
    default_exp = first_line.split("default_exp")[-1].strip()
    return default_exp


# exporti
def iter_function_names(source):
    for line in source:
        if not line.startswith("def "):
            continue
        if line.startswith("def _"):
            continue
        yield line.split("def ")[1].split("(")[0]


# export
def docs_copy_nb(relpath, docsrelpath=None):
    """copy a single notebook from src to dst with modified docs metadata."""
    src = repository_path(relpath)
    dst = docs_path((docsrelpath or relpath), not_exist_ok=True)
    nb = load_nb(src)
    nb_new = {**nb}
    nb_new["cells"] = []
    module = get_default_exp(nb)

    for cell in nb.get('cells', []):
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

        if not 'remove-cell' in cell_tags:
            for function_name in iter_function_names(source):
                extra_cell = {
                    "cell_type": "markdown",
                    "id": secrets.token_hex(8),
                    "metadata": {},
                    "source": [
                        ":::{eval-rst}\n",
                        f".. autofunction:: sax.{module}.{function_name}\n",
                        ":::\n"
                    ],
                }
                nb_new["cells"].append(extra_cell)

        cell["id"] = secrets.token_hex(8)
        nb_new["cells"].append(cell)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    save_nb(nb_new, dst)
    return dst


# +
# export
def list_notebooks(dir):
    return glob.glob(os.path.join(dir, "**/*.ipynb"), recursive=True)

def list_zips(dir):
    return glob.glob(os.path.join(dir, "**/*.zip"), recursive=True)


# -

# export
def docs_copy_dir(relpath):
    main_src = repository_path(relpath)
    for src in list_notebooks(main_src):
        rel = os.path.relpath(src, repository_path())
        docs_copy_nb(rel)
    for src in list_zips(main_src):
        rel = os.path.relpath(src, repository_path())
        dst = docs_path("_build", "html", rel, not_exist_ok=True)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


# export
def docs_copy_simulations():
    with_results, without_results = {}, {}
    for fn in os.listdir(simulations_path()):
        sim, ext = os.path.splitext(fn)
        if ext != ".ipynb":
            continue
        try:
            with_results[sim] = get_nominal_result(sim)
        except FileNotFoundError:
            without_results[sim] = {'hash': None, 'params': get_default_params(simulations_path(f"{sim}.ipynb"))}
        except ValueError:
            without_results[sim] = {'hash': None, 'params': get_default_params(simulations_path(f"{sim}.ipynb"))}
            #raise
        
    for k, r in with_results.items():
        docs_copy_nb(f"results/{k}/{r['hash']}/_simulated.ipynb", f"simulations/{k}.ipynb")
        
    for k, r in without_results.items():
        docs_copy_nb(f"simulations/{k}.ipynb", f"simulations/{k}.ipynb")
        
    shutil.copytree(simulations_path("img"), docs_path("_build", "html", "simulations", "img", not_exist_ok=True), dirs_exist_ok=True)
    return with_results, without_results


# export
def get_toc_part(toc, caption):
    parts = [p for p in toc["parts"] if caption == p["caption"]]
    try:
        return parts[0]
    except IndexError:
        raise ValueError(f"No TOC part with caption {caption!r} found.")


# export
def make_docs():
    docs_copy_nb("index.ipynb")
    docs_copy_dir("nbs")
    docs_copy_dir("examples")
    os.chdir(docs_path())
    subprocess.check_call([sys.executable.replace("python", "jupyter-book"), "build", "."])


# export
if __name__ == '__main__' and not IN_IPYTHON:
    make_docs()
