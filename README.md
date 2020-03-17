# torchbraid

A braid interface to pytorch

Requirements:
  + python libs:
    cython
    mpi4py
    pytorch
  + build of xbraid
  + MPI compiler

To build:

  1) Modify setup.py to point to xbraid library (lib and includes)
  2) Type make
  3) You will need to add 
       `export PYTHONPATH=${TORCHBRAID_DIR}/python:${PYTHONPATH}` to your 
     environment

To test:

  python test.py -v 

To look at code in the examples directory.


