#!/usr/bin/env python

import synergia_workflow

opts = synergia_workflow.Options("booster")

opts.add("turns", 1500, "number of turns")
opts.add("injection_energy", 800.0, "injection energy (MeV)")
opts.add("final_energy", 8000.0, "extraction energy (MeV)")
opts.add("generate_bunch", False, "whether to generate a bunch or read it from a file")

# openPMD file converted from Synergia
# with linear transformation from the twiss functions at its generation
# to match the beginning of the sbbooster lattice.
opts.add("particles_file", "/pscratch/sd/e/egstern/pip2/pip-ii-injected/pip-ii-injected-583k-xform-opmd.h5", "file from which to read initial particle distribution")

opts.add("full_booster_charge", 6.7e12, "Charge of a fully loaded Booster")
opts.add("harmonic_number", 84, "The harmonic number of the Booster RF")
opts.add("full_buckets", 81, "Number of full buckets (3 buckets are empty for extraction)")

# not really options but useful values to have around
# as calculated by MAD-X. I especially need gamma_tr to know
# when transition occurs.
opts.add("alfa_x", -1.298673960026007664e-02, "alpha_x")
opts.add("beta_x", 3.373645362843065243e01, "beta_x")
opts.add("alfa_y", 6.089861210659328755e-03, "alpha_y")
opts.add("beta_y", 5.252517912567207681e00, "beta_y")

opts.add("disp_x", 3.785167992, "x dispersion")
opts.add("disp_px", 0.001377568703, "xp dispersion")
opts.add("gamma_tr", 5.449167323, "gamma transition")

opts.add("nancheck", False, "activate runtime checking for NaN in AmREX")

job_mgr = synergia_workflow.Job_manager("booster_accel.py", opts, ["sbbooster-cooked.madx", "booster_rf.py", "booster_set_rf.py", "booster_momentum.py", "get_lattice.py", "syn2_to_impactx.py"])

