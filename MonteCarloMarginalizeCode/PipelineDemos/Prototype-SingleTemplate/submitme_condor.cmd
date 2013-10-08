Universe = vanilla
getenv = true

Executable = /usr/bin/python 
arguments = test_like_and_samp.py
error  = cndr.err
log = cndr.log
output = cndr.output

Queue 1

Executable = /usr/bin/python 
arguments = test_like_and_samp_margPsi.py
error  = cndr.err
log = cndr.log
output = cndr.output

Queue 1

