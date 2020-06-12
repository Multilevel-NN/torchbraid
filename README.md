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
       `export PYTHONPATH=${TORCHBRAID_DIR}:${PYTHONPATH}` to your 
     environment, this makes sure that the python search path
     for modules is setup

Take look at code in the examples directory.

### To test:

 1. `make tests`
 1. `make tests-serial`

### To clean the directory:

   `make clean`

### To uninstall python (this may be a no-op)

   `make uninstall`

## Publications

### Layer-Parallel

1. Cyr, Eric C., Stefanie Günther, and Jacob B. Schroder. "Multilevel Initialization for Layer-Parallel Deep Neural Network Training." arXiv preprint arXiv:1912.08974 (2019). [Link](https://arxiv.org/pdf/1912.08974)
1.  Günther, Stefanie, Lars Ruthotto, Jacob B. Schroder, Eric C. Cyr, and Nicolas R. Gauger. "Layer-parallel training of deep residual neural networks." SIAM Journal on Mathematics of Data Science 2, no. 1 (2020): 1-23. [Link](https://epubs.siam.org/doi/pdf/10.1137/19M1247620)
