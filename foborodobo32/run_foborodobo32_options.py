#!/usr/bin/env python

import synergia_workflow

opts = synergia_workflow.Options("foborodobo32")

opts.add("macroparticles", 1048576, "number of macro particles")
opts.add("emitx", 8.0e-6, "horizontal emittance (geometric)")
opts.add("emity", 8.0e-6, "vertical emittance (geometric")
opts.add("stddpop", 1.0e-4, "width of dp/p distribution")

job_mgr = synergia_workflow.Job_manager("run_foborodobo32.py", opts, ["foborodobo32.madx", "syn2_to_impactx.py"])
