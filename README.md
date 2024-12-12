# TorchBraid

XBraid interface to PyTorch

## Build TorchBraid: pip (recommended):

1. Optional: create a new virtual environment

   `python -m venv tb-env`  
   `source tb-env/bin/activate`

1. Install using pip.  From inside TorchBraid directory, do  
  `pip install .`

    If a development environment is desired, do  
    `pip install -e .`  
    Then all changes in the .py files are directly applicable in the
    installation. Changes to .pyx files require a re-installation.

    You can also install directly from github using  
    `pip install git+ssh://git@github.com/Multilevel-NN/TorchBraid.git`  
    or the HTTP equivalent. 

## Build TorchBraid: Conda
  
With conda the easiest path is to use the pip install for TorchBraid

```
conda create -n tb-env python=3.10
conda activate tb-env
pip install git+ssh://git@github.com/Multilevel-NN/TorchBraid.git  # or local equivalent 
```

For testing the


## Run unit tests 

  * Make

    `make tests tests-serial`

  * TOX (may need to install tox)  
  `tox`

    The package tox is used for testing in a continuous integration sense and automatically
    creates and populates a new python environment. However, if you have an environment that
    already satisfies the dependency requirements you can run the test commands directly
    using `tox-direct`. 

     1. Install tox-direct
       `pip install tox-direct'

     1. Run commands
       `tox --direct`

1. Test run  
 `cd examples/mnist/`  
 `mpirun -n 2 python mnist_script.py --percent-data 0.01`

## GPU direct communication

Torchbraid uses direct GPU communication when running simulations on GPUs. For this, Torchbraid requires a 
CUDA-aware MPI version ( see [here](https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/)
or [here](https://www.open-mpi.org/faq/?category=runcuda) for more information). A simple first test to determine if 
your system supports CUDA-aware MPI is to execute the command

`ompi_info --parsable -l 9 --all | grep mpi_built_with_cuda_support:value`

This command returns a string with true or false at the end. However, in our experiments, it was not always sufficient 
to check that this value is true. One way to test whether direct GPU communication works on your system is to run:

`make tests-direct-gpu`

If the test works, your MPI version supports direct GPU communication. If the test throws an error (typically a 
segmentation fault), your MPI version does not support direct GPU communication.


## Build TorchBraid: Makefile (advanced):

### Requirements:
  + python libs:
    cython
    mpi4py
    pytorch
  + build of xbraid
  + MPI compiler

Note, the cython version is pretty important, particularly if torch layers are shipped directly by braid.

### Setup for Conda (with native MPI support)

### Build xbraid:
  1. Download from git@github.com:XBraid/xbraid.git
  1. The master branch should work fine
  1. From the xbraid directory run `make debug=no braid`

### Build TorchBraid
  1. Copy makefile.inc.example to makefile.inc 
  1. Modify makefile.inc to include your build specifics
  1. Type make
  1. You will need to add the TorchBraid directory to your python path. E.g.:
     1. `export PYTHONPATH=${PYTHONPATH}:/path/to/TorchBraid/src`
    This makes sure that the python search path for modules is setup.

Take look at code in the examples directory.

### To clean the directory:

   `make clean`

## Publications

1. Moon, Gordon Euhyun, and Eric C. Cyr. "Parallel Training of GRU Networks with a Multi-Grid Solver for Long Sequences." ICLR, 2022. [Arxiv Link](https://arxiv.org/abs/2203.04738)
1. Cyr, Eric C., Stefanie Günther, and Jacob B. Schroder. "Multilevel Initialization for Layer-Parallel Deep Neural Network Training." arXiv preprint arXiv:1912.08974 (2019). [Arxiv Link](https://arxiv.org/pdf/1912.08974)
1.  Günther, Stefanie, Lars Ruthotto, Jacob B. Schroder, Eric C. Cyr, and Nicolas R. Gauger. "Layer-parallel training of deep residual neural networks." SIAM Journal on Mathematics of Data Science 2, no. 1 (2020): 1-23. [Link](https://epubs.siam.org/doi/pdf/10.1137/19M1247620)
