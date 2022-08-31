import os
import pytest
from nbdev.test import _test_one

def _find_notebooks(dir):
    dir = os.path.abspath(os.path.expanduser(dir))
    for root, _, files in os.walk(dir):
        for file in files:
            if ('checkpoint' in file) or (not file.endswith('.ipynb')):
                continue
            yield os.path.join(root, file)

@pytest.mark.parametrize('fn', sorted(_find_notebooks('nbs')))
def test_nbs(fn):
    print(fn)
    assert _test_one(fn)
