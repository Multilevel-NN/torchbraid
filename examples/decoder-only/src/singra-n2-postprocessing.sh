#!/bin/bash
#SBATCH --job-name=gpt-2-parallel-pp
#SBATCH --time=90:99:00
#SBATCH --output=gpt-2-parallel-pp.out
#SBATCH --error=gpt-2-parallel-pp.err
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


mpirun -n 2 python residual_postprocessing.py --lr 1e-4 --steps 48 --batch-size=32 --lp-max-levels "(1, 2)" --lp-bwd-max-iters 5 --lp-fwd-max-iters 2 --lp-cfactor 4 --lp-print-level 1 --lp-braid-print-level 1 --Tf 1 --log-interval 8 --seq-len 512
