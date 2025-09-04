#!/bin/bash
#SBATCH --job-name=post-processing
#SBATCH --time=90:99:00
#SBATCH --output=ml_multi_2-postprocessing.out
#SBATCH --error=ml_multi_2-postprocessing.err
#SBATCH --nodelist=sn[5-16]

#SBATCH --nodes=2
#SBATCH --partition=ampere
#SBATCH --ntasks-per-node=1

module purge

module load gcc/12.2.0
module load openmpi/4.1.4
module load cuda/11.8.0   
module load cudnn/8.4.0.27-11.6 
module load ucx/1.13.1 
module load python/3.10.8

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/braids_v2/pip-test/bin/activate
BATCH_SIZE=256
EPOCHS=1
PDATA=30000

mpirun -n 2 python residual_postprocessing.py --lr 5e-5 --percent-data=$PDATA --steps 32 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lp-max-levels "(1, 2)" --lp-bwd-max-iters 1 --lp-fwd-max-iters 1 --lp-cfactor 4 --lp-print-level 1 --lp-braid-print-level 1 --Tf 1 --log-interval 1 --seq-len 96


