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

  1) makefile.inc to include your build specifics
  2) Type make
  3) You will need to add 
       `export PYTHONPATH=${TORCHBRAID_DIR}:${PYTHONPATH}` to your 
     environment

To test:

  make tests

To clean the directory:

   make clean

To uninstall python

   make uninstall


Take look at code in the examples directory.
