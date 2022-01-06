""" A collection of useful constants for SAX simulations. """

from __future__ import annotations
import pathlib

import math

pi: float = math.pi
""" pi """

c: float = 299792458.0
"""speed of light"""


module_path = pathlib.Path(__file__).parent.absolute()
repo_path = module_path.parent


class Path:
    module = module_path
    repo = repo_path
    data = repo_path / "data"
    mmi_csv = data / "mmi1x2_si220n.csv"
    mmi_dat = data / "mmi1x2_si220n.dat"


PATH = Path()
