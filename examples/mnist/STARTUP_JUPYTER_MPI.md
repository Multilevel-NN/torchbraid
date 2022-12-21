The notebook requires some setup: start0_install_mpi_notebook.ipynb

### This is a simple "Hello World" example for MPI with iPython
For references, these websites are not terrible
- https://ipython.org/ipython-doc/3/parallel/magics.html
- https://charlesreid1.com/wiki/Jupyter/MPI

#### Before running this example, do the following
1) Install dependencies (assumes dependencies, e.g. mpi4py have  been install)

        pip3 install ipyparallel
        jupyter serverextension enable --py ipyparallel

2)  Start local ipython cluster independently of jupyter
- Create mpi ipython profile

        $ ipython profile create --parallel --profile=mpi
   
- Note, MPI should be listed as installed in the file

        ~/.ipython/profile_mpi/ipcluster_config.py`


3) Start ipython cluster from new terminal

        $ ipcluster start --n=2 --engines=mpi --profile=mpi
   
   or depending, may need to use
   
        $ ipcluster start --n=2 --engines=MPIEngineSetLauncher --profile=mpi

4) Leave that ipython cluster running in its own terminal
- Issues:  This page is helpful https://charlesreid1.com/wiki/Jupyter/MPI


5) Connect to ipython cluster in this Jupyter notebook by running the below code blocks
- Note how you need to first connect to the ipython cluster, and then synchronize imports of packages
