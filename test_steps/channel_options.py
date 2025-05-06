#!/usr/bin/env python

from math import pi
import synergia_workflow

opts = synergia_workflow.Options("foborodobo32_accel")

opts.add("seed", 12345791, "Pseudorandom number generator seed", int)

opts.add("momentum", 2.0, "proton beam momentum")
opts.add("enable_rf", True, "turn on the RF cavities")
opts.add("harmon", 32, "harmonic number for RF")
opts.add("rf_volt", 0.05, "RF voltage [MV]")
opts.add("lag", 0.0, "rf cavity phase in units of 2*pi")

opts.add("matching", "6dmoments", "matching procedure 6dmoments|uniform")

opts.add("emitx", 8.0e-6, "8 pi mm mr 90% not normalized")
opts.add("emity", 8.0e-6, "8 pi mm mr 90% not normalized")
opts.add("stddpop", 1.0e-4, "dp/p standard deviation for uniform beams")

opts.add("num_bunches", 1, "number of bunches in bunch train")
opts.add("macroparticles", 24, "number of macro particles")
opts.add("real_particles", 5.0e10, "number of real particles (bunch charge)")
opts.add("periodic", True, "make bunch periodic boundary conditions")
opts.add("turns", 1, "number of turns")

opts.add("spacecharge", None, "space charge [off|2d-openhockney|2d-bassetti-erskine|3d-openhockney", str)
opts.add("gridx", 32, "x grid size")
opts.add("gridy", 32, "y grid size")
opts.add("gridz", 128, "z grid size")
opts.add("comm_group_size", 1, "Communication group size for space charge solvers (must be 1 on GPUs), probably 16 on CPU", int)


opts.add("xtune", None, "adjust x tune", float)
opts.add("ytune", None, "adjust y tune", float)

# chromaticity adjustments
opts.add("set_hchrom", None, "adjust x chromaticity", float)
opts.add("set_vchrom", None, "adjust y chromaticity", float)

opts.add("stepper", "elements", "which stepper to use independent|elements|splitoperator")
opts.add("steps", 1, "# steps")

opts.add("step_diag", True, "diagnostics each step")
opts.add("tracks", 100, "number of particles to track")
opts.add("particles", True, "if True, save particles")
opts.add("particles_period", 0, "0: save every turn, n!=0, save particles every n turns")

job_mgr = synergia_workflow.Job_manager("channel.py", opts, ["channel.madx"])
