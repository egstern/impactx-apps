#!/usr/bin/env python

from math import pi
import synergia_workflow

opts = synergia_workflow.Options("booster-simple")

opts.add("seed", 12345791, "Pseudorandom number generator seed", int)

opts.add("lattice_file" , "sbbooster.madx", "lattice file to read")
opts.add("lattice_line", "booster", "which line in the lattice file to use")
opts.add("json_lattice_file", "cooked_booster.json", "json lattice file for running booster")
opts.add("momentum", 2.0, "proton beam momentum")
opts.add("enable_rf", True, "turn on the RF cavities")
opts.add("harmon", 32, "harmonic number for RF")
opts.add("rf_volt", 0.0002, "total RF voltage [GV]")

opts.add("matching", "6dmoments", "matching procedure 6dmoments|uniform|file")
opts.add("particles_file", "/home/egstern/impactx-apps/booster-simple/pip2/postinjection.02/pip-ii-injected-tenpercent.h5", "file of particles")

opts.add("emitx", 8.0e-6, "8 pi mm mr 90% not normalized")
opts.add("emity", 8.0e-6, "8 pi mm mr 90% not normalized")
opts.add("stddpop", 1.0e-3, "dp/p standard deviation for uniform beams")

opts.add("num_bunches", 1, "number of bunches in bunch train")
opts.add("macroparticles", 65536, "number of macro particles")
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
opts.add("set_xchrom", None, "adjust x chromaticity", float)
opts.add("set_ychrom", None, "adjust y chromaticity", float)

opts.add("stepper", "elements", "which stepper to use independent|elements|splitoperator")
opts.add("steps", 1, "# steps")

opts.add("step_diag", False, "diagnostics each step")
opts.add("tracks", 0, "number of particles to track")
opts.add("particles", False, "if True, save particles")
opts.add("particles_period", 0, "0: save every turn, n!=0, save particles every n turns")

opts.add("test_particles", False, "add test particles to the bunch")

job_mgr = synergia_workflow.Job_manager("booster-simple.py", opts, ["booster_simple_options.py", "sbbooster-cooked.madx"])
