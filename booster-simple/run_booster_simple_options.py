#!/usr/bin/env python

from math import pi
import synergia_workflow

opts = synergia_workflow.Options("run_booster_simple")

opts.add('enable_rf', True, 'turn on the RF cavities')
opts.add("emitx", 8.0e-6, "8 pi mm mr 90% not normalized")
opts.add("emity", 8.0e-6, "8 pi mm mr 90% not normalized")
opts.add("stddpop", 1.0e-4, "dp/p standard deviation for uniform beams")


job_mgr = synergia_workflow.Job_manager("run_booster_simple.py", opts, ["sbbooster.madx", "syn2_to_impactx.py"])
