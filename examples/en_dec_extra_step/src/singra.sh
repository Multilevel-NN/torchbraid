#!/bin/bash
#SBATCH --job-name=mt-transformer
#SBATCH --time=00:10:00
#SBATCH --output=mt.out
#SBATCH --error=mt.err

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

source ~/braids/pip-test/bin/activate

mpirun -n 2 python main_noDP.py

