## UCI Human Activity Recognition Examples Directory

### Jupyter Notebook Examples:
1. First see the `examples/mnist/start0_install_mpi_notebook.ipynb` and `examples/mnist/start1_simple_mpi_notebook.ipynb` notebooks for instructions on installing, verifying MPI in the Jupyter setting

1. `UCI_HAR_GRU_notebook_simple.ipynb`
   Simple UCI HAR learning example in MPI+Jupyter form

1. `UCI_HAR_GRU_notebook.ipynb`
   More complicated UCI HAR example in MPI+Jupyter form, with more options for the layer-parallel solver

### Command-line Python scripts for training
1. `UCI_HAR_GRU_script.py`
   Script equivalent to `UCI_HAR_GRU_notebook.ipynb`, most full featured MNIST example in this directory.

   Depends on some network definitions in `network_architecture.py` and data download/parsing utilities in `data.py`
