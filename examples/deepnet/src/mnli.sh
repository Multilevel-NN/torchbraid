#!/bin/bash
#SBATCH --job-name=mnli-search
#SBATCH --time=90:00:00
#SBATCH --output=mnli.out
#SBATCH --error=mnli.err
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

source ~/braids_v2/pip-test/bin/activate

python mnli.py --bsize 32 --lr 5e-5
python mnli.py --bsize 32 --lr 8e-5


