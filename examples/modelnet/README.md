## ModelNet Example Directory
Trains a ResNet CNN on the ModelNet10 dataset. The network can be trained in serial and with layer parallel and spatial coarsening.
Spatial coarsening improves the performance of layer parallel training by reducing the cost of each iteration.

### Command-line Python scripts for training
1. Install required package:
    $ pip3 install open3d
   
2. Download the dataset: (This will also process the data, which may take a few minutes)
    $ python3 downloadModelNet.py

3. Visualize dataset and spatial coarsen and refine functions with coarse_refine_test.ipynb

4. Train the network in serial:
    $ python3 ModelNet_script.py --percent-data 1. --lp-max-levels 1

5. Train the network with layer parallel and spatial coarsening:
    $ mpirun -n 32 python3 ModelNet_script.py --percent-data 1. --lp-max-levels 3 --lp-sc-levels 0

6. Test the saved network models/nx31_nt128_ml3_sc0_78percent.pt:
    $ mpirun -n 32 python3 test_parallel.py --percent-data 1. --lp-max-levels 3 --lp-sc-levels 0 --retrained-network

