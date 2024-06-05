#!/bin/bash
#SBATCH --job-name=lang-model-serial
#SBATCH --time=20:30:00
#SBATCH --output=ml_serial.out
#SBATCH --error=ml_serial.err
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
	
python main_serial.py --percent-data .01 --steps 32 --epochs 3 --batch-size 32 --model_dimension 384 --num_heads 6

# mpirun -n 2 python main.py

# Following is runs for getting losses for comparisons
# Baseline (no iterations)
# python main.py --epochs 10 --lp-max-levels 1 --num_heads 1 --steps 32

# Run different ones
# python main.py --epochs 10 --lp-max-levels 2 --lp-bwd-max-iters 1 --lp-fwd-max-iters 1 --num_heads 1 --steps 32
# python main.py --epochs 10 --lp-max-levels 2 --lp-bwd-max-iters 2 --lp-fwd-max-iters 2 --num_heads 1 --steps 32

# python main.py --epochs 10 --lp-max-levels 3 --lp-bwd-max-iters 2 --lp-fwd-max-iters 2 --num_heads 1 --steps 32
# python main.py --epochs 10 --lp-max-levels 3 --lp-bwd-max-iters 1 --lp-fwd-max-iters 2 --num_heads 1 --steps 32




