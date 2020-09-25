# torchbraid

A braid interface to pytorch

## Requirements:
  + python libs:
    cython
    mpi4py
    pytorch
  + build of xbraid
  + MPI compiler

## Build xbraid:
  1. Download from git@github.com:eric-c-cyr/xbraid.git
  1. Checkout cython-adjoint branch: git checkout cython-adjoint
  1. From the xbraid directory run `make debug=no braid`

## Build torchbraid:

  1. Copy makefile.inc.example to makefile.inc 
  1. Modify makefile.inc to include your build specifics
  1. Type make
  1. You will need to add 
       `export PYTHONPATH=${TORCHBRAID_DIR}/torchbraid:${TORCHBRAID_DIR}:${PYTHONPATH}` to your 
     environment, this makes sure that the python search path
     for modules is setup

Take look at code in the examples directory.

### To Compile:
   `make clean`
   `make`

### To Run:

 1. Serial RNN (one node): `mpirun -n 1 python forward_scaling_RNN.py --serial 2`
 2. Parallel RNN (one node): `mpirun -n 1 python forward_scaling_RNN.py 2`
 Parallel RNN (two nodes): `mpirun -n 2 python forward_scaling_RNN.py 2`
 Parallel RNN (four nodes): `mpirun -n 4 python forward_scaling_RNN.py 4`
