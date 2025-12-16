from pathlib import Path

import sax


def test_lum():
    path = Path(__file__).resolve().parent / "lum" / "spar_straight_dc_7_um_600_nm.dat"
    data = sax.parsers.parse_lumerical_dat(path)
