import sax
import numpy as np
import matplotlib.pyplot as plt
wls = np.array([2.19999, 2.20001, 2.22499, 2.22501, 2.24999, 2.25001, 2.27499, 2.27501, 2.29999, 2.30001, 2.32499, 2.32501, 2.34999, 2.35001, 2.37499, 2.37501, 2.39999, 2.40001, 2.42499, 2.42501, 2.44999, 2.45001])
phis = np.array([5.17317336, 5.1219654, 4.71259842, 4.66252492, 5.65699608, 5.60817922, 2.03697377, 1.98936119, 6.010146, 5.96358061, 4.96336733, 4.91777933, 5.13912198, 5.09451137, 0.22347545, 0.17979684, 2.74501894, 2.70224092, 0.10403192, 0.06214664, 4.83328794, 4.79225525])
wl = np.linspace(wls.min(), wls.max(), 10000)
phi = np.array(sax.grouped_interp(wl, wls, phis))

_, ax = plt.subplots(2, 1, sharex=True, figsize=(14, 6))
plt.sca(ax[0])
plt.plot(1e3*wls, np.arange(wls.shape[0]), marker="o", ls="none")
plt.grid(True)
plt.ylabel("index")
plt.sca(ax[1])
plt.grid(True)
plt.plot(1e3*wls, phis, marker="o", c="C1")
plt.plot(1e3*wl, phi, c="C2")
plt.xlabel("λ [nm]")
plt.ylabel("φ")
plt.show()