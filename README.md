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
for using torchbraid. The one caveat, is that mpi4py should be installed consistently with the MPI compiler. In some cases
doing a 'pip install mpi4py' is to be preferred to installing it through conda (conda installs an alternate MPI compiler and
library. You might want mpi4py to use the native one on your platform).

Note, virtual environments can be used instead of Conda.

Note, the cython version is pretty important, particularly if torch layers are shipped directly by braid.

### Setup for Conda (with native MPI support)
  
  ```
  conda env create -f ${TORCHBRAID_DIR}/env/py37.env
  conda activate py37
  MPICC=path/to/mpicc pip install mpi4py
  ```
## Build torchbraid (pip):

1. Optional: create a new virtual environment

   `python -m venv pip-test`  
   `source pip-test/bin/activate`

1. Install using pip.  From inside torchbraid directory, do  
  `pip install .`

    If a development environment is desired, do  
    `pip install -e .`  
    Then all changes in the .py files are directly applicable in the
    installation. Changes to .pyx files require a re-installation.

1. Run unit tests (may need to install tox)  
  `tox`

1. Test run  
 `cd examples/mnist/`  
 `python download.py`  
 `mpirun -n 2 python main.py`


## Build torchbraid (Makefile):

### Build xbraid:
  1. Download from git@github.com:XBraid/xbraid.git
  1. The master branch should work fine
  1. From the xbraid directory run `make debug=no braid`


### Build torchbraid
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

## GPU direct communication

The default communication scheme of torchbraid using GPU's and layer parallel is given by: 

GPU0 -> CPU0 -> CPU1 -> GPU1

Copying memory between the GPUs and CPUs can be expensive, so torchbraid also supports direct GPU communication, where 
data is sent directly between GPUs without taking the longer route via CPUs. This feature can be enabled by a simple 
flag, but requires a CUDA-aware MPI version ( see [here](https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/)
or [here](https://www.open-mpi.org/faq/?category=runcuda) for more information). A simple first test to determine if 
your system supports CUDA-aware MPI is to execute the command

`ompi_info --parsable -l 9 --all | grep mpi_built_with_cuda_support:value`

This command returns a string with true or false at the end. However, in our experiments, it was not always sufficient 
to check that this value is true. One way to test whether direct GPU communication works on your system is to run:

`make tests-direct-gpu`

If the test works, your MPI version supports direct GPU communication. If the test throws an error (typically a 
segmentation fault), your MPI version does not support direct GPU communication.

## Publications

1. Moon, Gordon Euhyun, and Eric C. Cyr. "Parallel Training of GRU Networks with a Multi-Grid Solver for Long Sequences." ICLR, 2022. [Arxiv Link](https://arxiv.org/abs/2203.04738)
1. Cyr, Eric C., Stefanie Günther, and Jacob B. Schroder. "Multilevel Initialization for Layer-Parallel Deep Neural Network Training." arXiv preprint arXiv:1912.08974 (2019). [Arxiv Link](https://arxiv.org/pdf/1912.08974)
1.  Günther, Stefanie, Lars Ruthotto, Jacob B. Schroder, Eric C. Cyr, and Nicolas R. Gauger. "Layer-parallel training of deep residual neural networks." SIAM Journal on Mathematics of Data Science 2, no. 1 (2020): 1-23. [Link](https://epubs.siam.org/doi/pdf/10.1137/19M1247620)
