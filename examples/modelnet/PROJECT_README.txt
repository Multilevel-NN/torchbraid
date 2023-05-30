1. Install torchbraid according to ../../README.md

2. Download the dataset: (This will download the full dataset, but only process the test set, which may take a few minutes)
    $ python3 downloadModelNet.py

3. Test the network trained in serial, saved in models/nx31_nt128_ml1_scNone.pt:
    i. in serial
        $ python3 test_parallel.py --percent-data 1. --lp-max-levels 1
    ii. with layer parallel
        $ mpirun -n 32 python3 test_parallel.py --percent-data 1. --lp-max-levels 3
    iii. with layer parallel and spatial coarsening (coarsening from level 0)
        $ mpirun -n 32 python3 test_parallel.py --percent-data 1. --lp-max-levels 3 --lp-sc-levels 0

4. Test the network trained with layer parallel and spatial coarsening, saved in models/nx31_nt128_ml3_sc0_78percent.pt:
    $ mpirun -n 32 python3 test_parallel.py --percent-data 1. --lp-max-levels 3 --lp-sc-levels 0 --retrained-network