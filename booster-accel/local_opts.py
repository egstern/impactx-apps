#!/usr/bin/env python

# numproc variable refers to the number of 16 core nodes that will
# be allocated and run with 16 threads each.

# The local_options file must be named local_opts.py and placed
# in the Synergia2 job manager search path.

from synergia_workflow import options

# Any instance of the Options class will be added as a suboption.
opts = options.Options('local')

# Any instance of the Override class will be used to override
#   the defaults of the existing options.
override = options.Override()
override.account = "m4272_g"
override.numproc = 4
override.procspernode=4
# The location of the setup.sh for your synergia build
override.template="job_perlmutter_gpu"
override.resumetemplate="resume_example"
override.queue="regular"
override.walltime="4:00:00"
