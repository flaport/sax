import os
import pytest
import shutil
from papermill.engines import papermill_engines
from papermill.execute import raise_for_execution_errors
from papermill.iorw import load_notebook_node

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
NBS_DIR = os.path.join(TEST_DIR, "nbs")
NBS_FAIL_DIR = os.path.join(NBS_DIR, "failed")

def get_kernel():
    kernel = os.environ.get("CONDA_DEFAULT_ENV", "base")
    if kernel == "base":
        kernel = "python3"
    return kernel

shutil.rmtree(NBS_FAIL_DIR, ignore_errors=True)
os.mkdir(NBS_FAIL_DIR)

def _find_notebooks(*dir):
    dir = os.path.abspath(os.path.join(TEST_DIR, *dir))
    for root, _, files in os.walk(dir):
        for file in files:
            if ('checkpoint' in file) or (not file.endswith('.ipynb')):
                continue
            yield os.path.join(root, file)

@pytest.mark.parametrize('path', sorted(_find_notebooks('nbs')))
def test_nbs(path):
    fn = os.path.basename(path)
    nb = load_notebook_node(path)
    nb = papermill_engines.execute_notebook_with_engine(
        engine_name=None,
        nb=nb,
        kernel_name=get_kernel(),
        input_path=path,
        output_path=None,
    )
    raise_for_execution_errors(nb, os.path.join(NBS_FAIL_DIR, fn))

