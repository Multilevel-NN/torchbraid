#!/bin/bash 

# This is an example shell script to be used as for profileing using the scalene profiler: https://github.com/plasma-umass/scalene
# The idea is that this script is used in place of calling python in an MPI command, for instance change:
# 
#   mpirun -n 3 python test.py --arg0 3
#
# to 
#
#   mpirun -n 3 run_scalene.sh test.py --arg0 3 
#
# this will then setup of the output files to include the number of processor and rank
#
# NOTE: As implemented this is only useful with OpenMPI
#

scalene --outfile profile-mpi${OMPI_COMM_WORLD_SIZE}-${OMPI_COMM_WORLD_RANK}.html $@
