#!/usr/bin/env python3
#
# Copyright 2022-2026 ImpactX contributors
# Authors: Eric G. Stern, Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#

import argparse

import matplotlib.pyplot as plt
import openpmd_api as io

# options to run this script
parser = argparse.ArgumentParser(description="Plot action of the polygon aperture.")
parser.add_argument(
    "--save-png", action="store_true", help="non-interactive run: save to PNGs"
)
args = parser.parse_args()


# initial/final beam
series = io.Series("diags/openPMD/monitor.h5", io.Access.read_only)
last_step = list(series.iterations)[-1]
initial = series.iterations[1].particles["beam"].to_df()
final = series.iterations[last_step].particles["beam"].to_df()


f, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, constrained_layout=True)

axs[0].scatter(initial["position_x"] * 1.0e3, initial["position_y"] * 1.0e3)
axs[0].set_title("initial")
axs[0].set_xlabel(r"$x$ [mm]")
axs[0].set_ylabel(r"$y$ [mm]")

axs[1].scatter(final["position_x"] * 1.0e3, final["position_y"] * 1.0e3)
axs[1].set_title("final")
axs[1].set_xlabel(r"$x$ [mm]")
axs[1].set_ylabel(r"$y$ [mm]")


plt.tight_layout()
if args.save_png:
    plt.savefig("polygon_aperture.png")
else:
    plt.show()
