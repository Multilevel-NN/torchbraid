#!/bin/bash
#SBATCH --job-name=vit-2-postprocessing
#SBATCH --time=90:99:00
#SBATCH --output=vit-2-postprocessing.out
#SBATCH --error=vit-2-postprocessing.err
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

BATCH_SIZE=32
EPOCHS=1
PDATA=30000

mpirun -n 2 python residual_postprocessing.py --lr 5e-4 --percent-data=$PDATA --steps 32 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lp-max-levels "(2, 2)" --lp-bwd-max-iters 8 --lp-fwd-max-iters 8 --lp-cfactor 4 --lp-print-level 1 --lp-braid-print-level 1 --Tf 1 --log-interval 1 --seq-len 224 --model_dimension 768


