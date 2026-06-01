#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

twiss = pd.read_csv("fccee_z.twiss", skiprows=52,
                    names=("name", "s", "betx", "alfx", "mux",
                           "bety", "alfy", "muy",
                           "dx", "dpx", "dy", "dpy"),
                    dtype={"a": str, "b": np.float64, "c": np.float64,
                           "d": np .float64, "e": np.float64,
                           "f": np.float64, "g": np.float64, "h": np.float64,
                           "i": np.float64, "j": np.float64, "k": np.float64,
                           "l": np.float64},
                    delimiter=r"\s+")

# print(twiss.shape)
# print(twiss.columns)

# print("s")
# print(twiss["s"][:20])

# print()
# print("betx")
# print(twiss["betx"][:20])

f, ax = plt.subplots(2, 1, sharex=True)

plt.suptitle(r"FCCee_z beta functions$")
ax[0].plot(twiss["s"], twiss["betx"], label=r"$\beta_x$")
ax[0].set_ylabel("beta x [m]")
ax[0].legend(loc="upper center")

ax[1].plot(twiss["s"], twiss["bety"], label=r"$\beta_y$")
ax[1].set_xlabel("s [m]")
ax[1].set_ylabel("beta y [m]")
ax[1].legend(loc="upper center")

# plt.figure()
# plt.title(r"$\beta_x$")
# plt.plot(twiss["s"], twiss["betx"])
# plt.xlabel("s [m]")
# plt.ylabel("beta x [m]")

# plt.figure()
# plt.title(r"$\beta_y$")
# plt.plot(twiss["s"], twiss["bety"])
# plt.xlabel("s [m]")
# plt.ylabel("beta y [m]")

plt.show()


