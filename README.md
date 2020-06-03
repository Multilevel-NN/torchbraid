# torchbraid

A braid interface to pytorch

Requirements:
  + python libs:
    cython
    mpi4py
    pytorch
  + build of xbraid
  + MPI compiler

To build xbraid:
  1. Download from git@github.com:eric-c-cyr/xbraid.git
  1. Checkout cython-adjoint branch: git checkout cython-adjoint
  1. From the xbraid directory run `make debug=no braid`

To build:

  1. Copy makefile.inc.example to makefile.inc 
  1. Modify makefile.inc to include your build specifics
  1. Type make
  1. You will need to add 
       `export PYTHONPATH=${TORCHBRAID_DIR}:${PYTHONPATH}` to your 
     environment, this makes sure that the python search path
     for modules is setup

To test:

  make tests
  make tests-serial

To clean the directory:

   make clean

To uninstall python (this may be a no-op)

   make uninstall


Take look at code in the examples directory.
