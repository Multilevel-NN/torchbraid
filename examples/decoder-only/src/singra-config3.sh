#!/bin/bash
#SBATCH --job-name=gpt-2-config3
#SBATCH --time=90:99:00
#SBATCH --output=full-gpt-saves/gpt-parallel.out
#SBATCH --error=full-gpt-saves/gpt-parallel.err
#SBATCH --nodelist=sn[5-16]

#SBATCH --nodes=5
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

# Parallel case with buffer; steps 4 smaller due to cf; changed n=5
# mpirun -n 5 python main.py --lr 2e-4 --steps 20 --batch-size=32 --lp-max-levels "(1, 2)" --lp-bwd-max-iters 1 --lp-fwd-max-iters 1 --lp-cfactor 4 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1 --log-interval 16 --seq-len 1024 --buffer

mpirun -n 5 python main.py --lr 2e-4 --steps 20 --batch-size=32 --lp-max-levels "(1, 2)" --lp-bwd-max-iters 2 --lp-fwd-max-iters 1 --lp-cfactor 4 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1 --log-interval 16 --seq-len 1024 --buffer
