# torchbraid

A braid interface to pytorch

## Requirements:
  + python libs:
    cython
    mpi4py
    pytorch
  + build of xbraid
  + MPI compiler

Conda environments can be found in 'torchbraid/env' directories. These can be used too get a consistent conda enviroonement
for using torchbraid. The one caveat, is that mpi4py should be installed consistenclty with the MPI compiler. In some cases
doing a 'pip install mpi4py' is to be preferred to installing it through conda (conda installs an alternate MPI compiler and
library. You might want mpi4py to use the native one on your platform).

Note, the cython version is pretty important, particularly if torch layers are shipped directly by braid.

### Setup for Conda (with native MPI support)
  
  ```
  conda env create -f ${TORCHBRAID_DIR}/env/py37.env
  conda activate py37
  MPICC=path/to/mpicc pip install mpi4py
  ```

## Build xbraid:
  1. Download from git@github.com:XBraid/xbraid.git
  1. The master branch should work fine
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

### To test:

 1. `make tests`
 1. `make tests-serial`

### To clean the directory:

   `make clean`

### To uninstall python (this may be a no-op)

   `make uninstall`

## Publications

1. Moon, Gordon Euhyun, and Eric C. Cyr. "Parallel Training of GRU Networks with a Multi-Grid Solver for Long Sequences." ICLR, 2022. [Arxiv Link](https://arxiv.org/abs/2203.04738)
1. Cyr, Eric C., Stefanie Günther, and Jacob B. Schroder. "Multilevel Initialization for Layer-Parallel Deep Neural Network Training." arXiv preprint arXiv:1912.08974 (2019). [Arxiv Link](https://arxiv.org/pdf/1912.08974)
1.  Günther, Stefanie, Lars Ruthotto, Jacob B. Schroder, Eric C. Cyr, and Nicolas R. Gauger. "Layer-parallel training of deep residual neural networks." SIAM Journal on Mathematics of Data Science 2, no. 1 (2020): 1-23. [Link](https://epubs.siam.org/doi/pdf/10.1137/19M1247620)
