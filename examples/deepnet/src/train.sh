#!/usr/bin/bash 
#SBATCH --job-name=bert-pretrain
#SBATCH --output=pretrain_%j.out
#SBATCH --error=pretrain_%j.err
#SBATCH --time=95:00:00
#SBATCH --partition=ampere
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

module purge

source ~/braids/pip-test/bin/activate

python c4-training.py

