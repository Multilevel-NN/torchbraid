<p align="center">
<img src="TorchBraid-v0.1.png" width="300">
</p>

# TorchBraid
This package implements a layer-parallel approach to training neural ODEs, and neural networks. 
Algorithmically multigrid-in-time is used to expose parallelism in the forward and backward
propagation phases used to compute the gradient. The neural network interface is build on
PyTorch, while the backend uses XBraid (a C library) for multigrid-in-time.

If you are having trouble with the GPU/MPI interaction, including low communication as a result
of unneccessary host-device transfers (e.g. not using GPU aware MPI), please 
see [GPU Direct Communication](#gpu-direct-communication).

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

  1. Test the installation: See [Run Unit Tests](#run-unit-tests)

## Build TorchBraid: Conda
  
With conda the easiest path is to use the pip install for TorchBraid

```
conda create -n tb-env python=3.10
conda activate tb-env
pip install git+ssh://git@github.com/Multilevel-NN/TorchBraid.git  # or local equivalent 
```

For testing see [Run Unit Tests](#run-unit-tests)

## Run Unit Tests

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

TorchBraid uses direct GPU communication when running simulations on GPUs. For this, Torchbraid requires a 
CUDA-aware MPI version ( see [here](https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/)
or [here](https://www.open-mpi.org/faq/?category=runcuda) for more information). A simple first test to determine if 
your system supports CUDA-aware MPI is to execute the command

`ompi_info --parsable -l 9 --all | grep mpi_built_with_cuda_support:value`

This command returns a string with true or false at the end. However, in our experiments, it was not always sufficient 
to check that this value is true. One way to test whether direct GPU communication works on your system is to run:

`make tests-direct-gpu`

If the test works, your MPI version supports direct GPU communication. You should see output that looks like (note
that the output header that tries to explain how the test result should be interpreted).

```

******************************************************************
* This script is to be run with two MPI ranks and                *
* tests the availability of GPU/MPI direct                       *
* communication. This is _required_ for TorchBraid when          *
* GPUs are used. This test will fail if either:                  *
*                                                                *
*    1. Torch was not built with GPUs, or GPUs are unavailable   * 
*    2. GPU aware MPI is not available (NVLINK with Nvidia)      *
*                                                                *
* If the test is successful, the last line on rank 0 will output *
*                                                                *
*    "PASSED: GPU aware MPI is available"                        *
*                                                                *
* While failures are indicated by:                               *
*                                                                *
*    "FAILED: GPU aware MPI is NOT available"                    *
*                                                                *
* Followed by a brief explaination of the type of failure seen.  *
* It's possible that a segfault can occur on some untested.      *
* platforms. That should be viewed as GPU aware MPI not being    *
* available.                                                     *
******************************************************************

Check For GPU-Direct Support
-- compile time: This MPI library has CUDA-aware support.
-- run time:This MPI library has CUDA-aware support.

Check For GPU-Direct Support
-- compile time: This MPI library has CUDA-aware support.
-- run time:This MPI library has CUDA-aware support.

PASSED: GPU aware MPI is available
```

If the final line says FAILED then your MPI version does not support direct GPU communication.
For instance, if you don't have CUDA enabled, then the error will look like:

```

******************************************************************
* This script is to be run with two MPI ranks and                *
* tests the availability of GPU/MPI direct                       *
* communication. This is _required_ for TorchBraid when          *
* GPUs are used. This test will fail if either:                  *
*                                                                *
*    1. Torch was not built with GPUs, or GPUs are unavailable   * 
*    2. GPU aware MPI is not available (NVLINK with Nvidia)      *
*                                                                *
* If the test is successful, the last line on rank 0 will output *
*                                                                *
*    "PASSED: GPU aware MPI is available"                        *
*                                                                *
* While failures are indicated by:                               *
*                                                                *
*    "FAILED: GPU aware MPI is NOT available"                    *
*                                                                *
* Followed by a brief explaination of the type of failure seen.  *
* It's possible that a segfault can occur on some untested.      *
* platforms. That should be viewed as GPU aware MPI not being    *
* available.                                                     *
******************************************************************

Check For GPU-Direct Support
-- compile time: This MPI library does NOT have CUDA-aware support.
-- run time:This MPI library does not have CUDA-aware support.

Check For GPU-Direct Support
-- compile time: This MPI library does NOT have CUDA-aware support.
-- run time:This MPI library does not have CUDA-aware support.

FAILED: GPU aware MPI is NOT available - "MPIX_Query_cuda_support" test failed.
```

We also check the `MPIX_Query_cuda_support` command available in most MPI libraries. Finally, it's 
possible, due to the range of implementations, that the script will raise a seg fault
if the GPU direct communication is not supported. If such a case arises, then feel free to reach out
with a description of the MPI implementation and version, CUDA version, and the platform being run on.

## Build TorchBraid: Makefile (advanced):

[Link to the Make instructions](MAKEINSTRUCTIONS.md)

## Publications

1. Moon, Gordon Euhyun, and Eric C. Cyr. "Parallel Training of GRU Networks with a Multi-Grid Solver for Long Sequences." ICLR, 2022. [Arxiv Link](https://arxiv.org/abs/2203.04738)
1. Cyr, Eric C., Stefanie Günther, and Jacob B. Schroder. "Multilevel Initialization for Layer-Parallel Deep Neural Network Training." arXiv preprint arXiv:1912.08974 (2019). [Arxiv Link](https://arxiv.org/pdf/1912.08974)
1.  Günther, Stefanie, Lars Ruthotto, Jacob B. Schroder, Eric C. Cyr, and Nicolas R. Gauger. "Layer-parallel training of deep residual neural networks." SIAM Journal on Mathematics of Data Science 2, no. 1 (2020): 1-23. [Link](https://epubs.siam.org/doi/pdf/10.1137/19M1247620)
