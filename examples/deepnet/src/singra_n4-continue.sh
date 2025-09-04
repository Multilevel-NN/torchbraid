#!/bin/bash
#SBATCH --job-name=cont22-serial
#SBATCH --time=90:99:00
#SBATCH --output=ml_multi_4-22-serial.out
#SBATCH --error=ml_multi_4-22-serial.err
#SBATCH --nodelist=sn[5-16]

#SBATCH --nodes=4
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
BATCH_SIZE=32
EPOCHS=1 # This is useless
PDATA=300000  #20000 # This is not used 

#mpirun -n 2 python main.py --lr 5e-5 --percent-data=$PDATA --steps 64 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lp-max-levels "(1, 2)" --lp-bwd-max-iters 1 --lp-fwd-max-iters 1 --lp-cfactor 4 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1 --log-interval 1 --seq-len 512  --model_dimension 768
mpirun -n 4 python main-continuing.py --lr 5e-4 --percent-data=$PDATA --steps 128 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lp-max-levels "(1, 1)" --lp-bwd-max-iters 1 --lp-fwd-max-iters 1 --lp-cfactor 4 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1 --log-interval 1 --seq-len 224  --model_dimension 768
