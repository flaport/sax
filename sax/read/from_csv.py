import numpy as np
import pandas as pd

# from scipy.interpolate import interp1d

from sax._typing import PathType, SDict, Float
from sax.constants import PATH


def from_csv(
    filepath: PathType, wl: Float = 1.55, xkey: str = "wavelength_nm", prefix: str = "S"
) -> SDict:
    """loads Sparameters from a CSV file

    Args:
        filepath:
        wl: wavelength
        prefix: for the sparameters
    """
    df = pd.read_csv(filepath)
    nsparameters = (len(df.keys()) - 1) // 2
    nports = int(nsparameters ** 0.5)

    # x = df[xkey]
    # wl_min = min(wl)
    # wl_max = max(wl)
    # df = df[df[xkey] >= wl_min][df[xkey] <= wl_max]

    return {
        (f"o{i}", f"o{j}"): df[f"{prefix}{i}{j}m"] * np.exp(1j * df[f"{prefix}{i}{j}a"])
        for i in range(1, nports + 1)
        for j in range(1, nports + 1)
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    filepath = PATH.mmi_csv
    df = pd.read_csv(filepath)
    s = from_csv(filepath=filepath)
    # s21 = s[("o1", "o2")]
    s21 = s[("o3", "o1")]
    w = np.linspace(2000, 1000, 500)
    plt.plot(w, np.abs(s21) ** 2)
    plt.show()
