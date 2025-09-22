#!/bin/bash
#SBATCH --job-name=vit-2-parallel
#SBATCH --time=90:99:00
#SBATCH --output=vit_2_ti-parallel.out
#SBATCH --error=vit_2_ti-parallel.err
#SBATCH --nodelist=sn[5-16]

#SBATCH --nodes=2
#SBATCH --partition=ampere
#SBATCH --ntasks-per-node=1

module purge

module load gcc/14.2.0
module load openmpi/4.1.7
module load cuda/12.6.2
module load cudnn/9.2.0.82-12
module load ucx/1.17.0
module load python/3.12.5


source ~/braids_v3/pip-test/bin/activate


# BATCH_SIZE=256
BATCH_SIZE=128
EPOCHS=30
PDATA=300000  #20000 # This is not used 

#mpirun -n 2 python main.py --lr 5e-5 --percent-data=$PDATA --steps 64 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lp-max-levels "(1, 2)" --lp-bwd-max-iters 1 --lp-fwd-max-iters 1 --lp-cfactor 4 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1 --log-interval 1 --seq-len 512  --model_dimension 768
#mpirun -n 2 python main.py --lr 5e-4 --percent-data=$PDATA --steps 32 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lp-max-levels "(1, 2)" --lp-bwd-max-iters 1 --lp-fwd-max-iters 1 --lp-cfactor 4 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1 --log-interval 1

# For cifar10
mpirun -n 2 python main.py --lr 5e-5 --percent-data=$PDATA --steps 12 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lp-max-levels "(2, 2)" --lp-bwd-max-iters 1 --lp-fwd-max-iters 1 --lp-cfactor 3 --lp-print-level 1 --lp-braid-print-level 1 --Tf .12 --log-interval 1 --serial-file True
mpirun -n 2 python main.py --lr 5e-5 --percent-data=$PDATA --steps 12 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lp-max-levels "(2, 2)" --lp-bwd-max-iters 1 --lp-fwd-max-iters 1 --lp-cfactor 3 --lp-print-level 1 --lp-braid-print-level 1 --Tf .12 --log-interval 1
