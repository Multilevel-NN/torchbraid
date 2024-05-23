#!/bin/bash
#SBATCH --job-name=lang-model
#SBATCH --time=08:30:00
#SBATCH --output=ml.out
#SBATCH --error=ml.err
#SBATCH --nodelist=sn[5-16]

#SBATCH --nodes=1
#SBATCH --partition=ampere
#SBATCH --ntasks-per-node=1

module purge

module load gcc/12.2.0
module load openmpi/4.1.4
module load cuda/11.8.0   
module load cudnn/8.4.0.27-11.6 
module load ucx/1.13.1 
module load python/3.10.8

source ~/braids/pip-test/bin/activate

mpirun -n 1 python main.py --percent-data .01 --steps 32 --epochs 4 --batch-size 32 --lp-max-levels 1 --lp-bwd-max-iters 1 --lp-fwd-max-iters 3 --lp-cfactor 4 --model_dimension 384 --num_heads 6 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1
