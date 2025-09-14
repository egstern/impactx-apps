#!/usr/bin/env python

from math import pi
import synergia_workflow

opts = synergia_workflow.Options("run_booster_simple")

opts.add('enable_rf', True, 'turn on the RF cavities')
opts.add("emitx", 8.0e-6, "8 pi mm mr 90% not normalized")
opts.add("emity", 8.0e-6, "8 pi mm mr 90% not normalized")
opts.add("stddpop", 1.0e-4, "dp/p standard deviation for uniform beams")
opts.add("harmonic_number", 84, "harmonic number of booster")

opts.add('macroparticles', 32768, "number of macroparticles")
opts.add("turns", 1, "number of turns to run")

opts.add("rf_volt", 200.0e-6, "RF voltage [GV]")

opts.add("set_xtune", None, "adjust horizontal tune", float)
opts.add("set_ytune", None, "adjust vertical tune", float)
opts.add("set_xchrom", -8.0, "adjust horizontal chromatcity", float)
opts.add("set_ychrom", -8.0, "adjust vertical chromaticity", float)

opts.add("test_particles", False, "whether to include test particles")

opts.add("initial_monitor", False, "put monitor element at beginning of lattice")
opts.add("final_monitor", True, "put monitor element at end of lattice")

job_mgr = synergia_workflow.Job_manager("run_booster_simple.py", opts, ["sbbooster.madx", "syn2_to_impactx.py"])
