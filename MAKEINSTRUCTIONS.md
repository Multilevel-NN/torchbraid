# Building with Make (advanced)

TorchBraid can also be built with Make, though this is more advanced and opens potential for unforeseen problems. As a result
its not the recommended approach. Regardless, here are some partial instructions! 

If you want a recommended way please see the [PIP Install](README.md#build-torchbraid-pip-recommended)

## Requirements:

Here are the requirements as of 12/12/2024. It is possible that these were not updated, please check [setpy.py](setup.py)
for the most up to date (or, again use [PIP Install](README.md#build-torchbraid-pip-recommended))

  + python libs:
    cython>=0.29.32,
    mpi4py,
    torch>=2.0.1,
    torchvision>=0.15.2,
    matplotlib
  + build of xbraid
  + MPI compiler

Note, the cython version is pretty important, particularly if torch layers are shipped directly by braid.

## Build xbraid:
  1. Download from git@github.com:XBraid/xbraid.git
  1. The master branch should work fine
  1. From the xbraid directory run `make debug=no braid`

## Build TorchBraid
  1. Copy makefile.inc.example to makefile.inc 
  1. Modify makefile.inc to include your build specifics
  1. Type make
  1. You will need to add the TorchBraid directory to your python path. E.g.:
     1. `export PYTHONPATH=${PYTHONPATH}:/path/to/TorchBraid/src`
    This makes sure that the python search path for modules is setup.

Take look at code in the examples directory.

## To clean the directory:

   `make clean`
