from pathlib import Path

import sax


def test_lum() -> None:
    path = Path(__file__).resolve().parent / "lum" / "spar_straight_dc_7_um_600_nm.dat"
    sax.parsers.parse_lumerical_dat(path)
